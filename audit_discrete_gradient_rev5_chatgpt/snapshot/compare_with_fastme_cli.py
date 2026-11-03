"""
compare_with_fastme_cli.py
--------------------------
Compares:
  1. Phylo2Vec BME Optimizer (C99 / ctypes via DiscreteGradient.optimize_phylo2vec_bme)
  2. In-memory C99 FastME bmeSPR implementation (DiscreteGradient.optimize_fastme_bme)
  3. External `fastme` CLI binary (if installed via `conda install -c bioconda fastme` or `brew install fastme`)
     using `fastme -i D.phy -u init.nwk -s -o out.nwk`, parsing `out.nwk`, and independently evaluating
     its exact Pauplin BME tree length and Robinson-Foulds (RF) distance.
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
import numpy as np
from discrete_gradient import DiscreteGradient, v_to_ancestry_matrix


def v_to_newick(v: np.ndarray) -> str:
    """
    Convert a Phylo2Vec vector v (0 <= v_i <= 2i) into an unrooted Newick string with taxon names T0..T{N-1}.
    Note: v_to_ancestry_matrix(v) returns an (N-1, 2) matrix M where row k represents internal node (N + k)
    with children M[k, 0] and M[k, 1].
    """
    M = v_to_ancestry_matrix(np.asarray(v, dtype=np.int64))
    n_pairs = len(v)
    n_leaves = n_pairs + 1
    sub = {i: f"T{i}" for i in range(n_leaves)}
    for k in range(n_pairs - 1):
        c1, c2 = int(M[k, 0]), int(M[k, 1])
        p = n_leaves + k
        sub[p] = f"({sub[c1]},{sub[c2]})"
    # Unrooted trifurcation at the top merger (k = n_pairs - 1)
    c1_top, c2_top = int(M[n_pairs - 1, 0]), int(M[n_pairs - 1, 1])
    rc_max = max(c1_top, c2_top)
    rc_min = min(c1_top, c2_top)
    row_max = rc_max - n_leaves
    u0, u1 = int(M[row_max, 0]), int(M[row_max, 1])
    return f"({sub[u0]},{sub[u1]},{sub[rc_min]});"


def parse_newick_and_evaluate(nwk_str: str, D: np.ndarray) -> tuple[float, set[frozenset[int]]]:
    """
    Parse an unrooted (or rooted) Newick string with leaves T0..T{N-1} (ignoring branch lengths),
    and independently compute:
      1. Exact Pauplin BME tree length L(T) = sum_{x < y} D[x, y] * 2^{-(d_T(x, y) - 1)}
      2. Canonical non-trivial split set (each split normalized to contain leaf 0) for RF distance.
    """
    # Strip branch lengths, whitespace, and trailing semicolon
    cleaned = re.sub(r":[-+0-9.eE]+", "", nwk_str).strip()
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()

    n_leaves = D.shape[0]
    adj: dict[int, list[int]] = {}
    next_int_id = n_leaves

    def _add_edge(u: int, w: int) -> None:
        adj.setdefault(u, []).append(w)
        adj.setdefault(w, []).append(u)

    # Recursive descent parser over comma-separated subtrees
    pos = 0

    def _parse_node() -> int:
        nonlocal pos, next_int_id
        if cleaned[pos] == "(":
            pos += 1  # consume '('
            children = []
            while True:
                children.append(_parse_node())
                if cleaned[pos] == ",":
                    pos += 1
                elif cleaned[pos] == ")":
                    pos += 1
                    break
                else:
                    raise ValueError(f"Unexpected character '{cleaned[pos]}' at pos {pos}")
            # Optional internal node label
            while pos < len(cleaned) and cleaned[pos] not in (",", ")"):
                pos += 1
            u = next_int_id
            next_int_id += 1
            for ch in children:
                _add_edge(u, ch)
            return u
        else:
            start = pos
            while pos < len(cleaned) and cleaned[pos] not in (",", ")"):
                pos += 1
            tok = cleaned[start:pos].strip()
            if not tok.startswith("T"):
                raise ValueError(f"Expected taxon label T<id>, got '{tok}'")
            leaf_id = int(tok[1:])
            adj.setdefault(leaf_id, [])
            return leaf_id

    root_id = _parse_node()
    # If the Newick root is bifurcating (degree 2), suppress it to obtain the unrooted topology
    if len(adj[root_id]) == 2:
        c0, c1 = adj[root_id]
        adj[c0].remove(root_id)
        adj[c1].remove(root_id)
        _add_edge(c0, c1)
        del adj[root_id]

    # Compute topological distances between all pairs of leaves via BFS and extract splits
    total_bme = 0.0
    for src in range(n_leaves):
        dist = {src: 0}
        queue = [src]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for nb in adj[u]:
                if nb not in dist:
                    dist[nb] = dist[u] + 1
                    queue.append(nb)
        for dst in range(src + 1, n_leaves):
            total_bme += np.ldexp(float(D[src, dst]), -(dist[dst] - 1))

    # Extract canonical splits by rooting at leaf 0's neighbor
    start_u = adj[0][0]
    parent = {start_u: -1}
    order = [start_u]
    for u in order:
        for nb in adj[u]:
            if nb != parent[u]:
                parent[nb] = u
                order.append(nb)
    sub_leaves: dict[int, set[int]] = {}
    all_leaves = set(range(n_leaves))
    splits: set[frozenset[int]] = set()
    for u in reversed(order):
        if u < n_leaves:
            sub_leaves[u] = {u}
        else:
            s: set[int] = set()
            for nb in adj[u]:
                if parent.get(nb) == u:
                    s |= sub_leaves[nb]
            sub_leaves[u] = s
        if u != start_u:
            s = sub_leaves[u]
            if 1 < len(s) < n_leaves - 1:
                canon = frozenset(s if (0 in s) else (all_leaves - s))
                splits.add(canon)

    return total_bme, splits


def write_phylip_matrix(D: np.ndarray, path: str) -> None:
    """Write an N x N distance matrix D in standard PHYLIP square format with 12 decimal places."""
    n = D.shape[0]
    with open(path, "w") as f:
        f.write(f"   {n}\n")
        for i in range(n):
            name = f"T{i}".ljust(10)
            vals = " ".join(f"{D[i, j]:.12f}" for j in range(n))
            f.write(f"{name}{vals}\n")


def main():
    dg = DiscreteGradient()
    fastme_bin = shutil.which("fastme")

    print("=" * 96)
    print("COMPARISON: Phylo2Vec BME Optimizer vs. C99 FastME-SPR vs. External `fastme` CLI")
    print("=" * 96)

    # Self-test of v_to_newick and Newick parser/evaluator on a random N=20 tree
    rng = np.random.default_rng(2026)
    v_test = np.array([0] + [rng.integers(0, 2 * i + 1) for i in range(1, 19)], dtype=np.int64)
    X_test = rng.normal(size=(20, 8))
    D_test = np.sqrt(((X_test[:, None, :] - X_test[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(D_test, 0.0)
    nwk_test = v_to_newick(v_test)
    L_nwk, splits_nwk = parse_newick_and_evaluate(nwk_test, D_test)
    assert len(splits_nwk) == 20 - 3 and np.isfinite(L_nwk), "v_to_newick self-test failed"
    print(f"[PASS] v_to_newick + Newick split/BME round-trip self-test (N=20, {len(splits_nwk)} internal splits)\n")

    if fastme_bin:
        print(f"Found external fastme binary at: {fastme_bin}\n")
    else:
        print("Note: External `fastme` CLI binary not found on PATH.")
        print("      Running in-memory C99 `optimize_fastme_bme` (exact Desper & Gascuel 2002 bmeSPR)")
        print("      vs. `optimize_phylo2vec_bme`. To also run the external CLI binary, install via:")
        print("          conda install -c bioconda fastme   OR   brew install fastme\n")

    for N in [20, 50, 100, 150, 200]:
        v_init = np.array([0] + [rng.integers(0, i + 1) for i in range(1, N - 1)], dtype=np.int64)
        X = rng.normal(size=(N, 12))
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(D, 0.0)

        v_p2v, L_p2v, evals_p2v, ms_p2v = dg.optimize_phylo2vec_bme(v_init, D)
        v_fme, L_fme, iters_fme, ms_fme = dg.optimize_fastme_bme(v_init, D)

        # Independently verify BME length via v_to_newick -> parse_newick_and_evaluate
        L_p2v_check, splits_p2v = parse_newick_and_evaluate(v_to_newick(v_p2v), D)
        L_fme_check, splits_fme = parse_newick_and_evaluate(v_to_newick(v_fme), D)
        assert abs(L_p2v - L_p2v_check) < 1e-9, f"P2V length mismatch: {L_p2v} vs {L_p2v_check}"
        assert abs(L_fme - L_fme_check) < 1e-9, f"FME length mismatch: {L_fme} vs {L_fme_check}"

        msg = (
            f"N={N:3d} | C99 FastME-SPR: L={L_fme:9.5f} ({iters_fme:3d} iters, {ms_fme:6.2f} ms) | "
            f"Phylo2Vec-BME: L={L_p2v:9.5f} ({evals_p2v:2d} evals, {ms_p2v:6.2f} ms) | "
            f"Speedup: {ms_fme / ms_p2v:.2f}x"
        )
        print(msg)

        if fastme_bin:
            with tempfile.TemporaryDirectory() as tmpdir:
                phy_path = os.path.join(tmpdir, "D.phy")
                nwk_path = os.path.join(tmpdir, "init.nwk")
                out_path = os.path.join(tmpdir, "out.nwk")
                write_phylip_matrix(D, phy_path)
                # FastME (-u) initializes T0->weight from the sum of Newick branch lengths
                # before running bmeSPR and reverts to T0 if T_spr->weight > T0->weight.
                # Setting each of the (2N-3) unrooted edges to L_init / (2N-3) sets T0->weight = L_init.
                L_init, _ = parse_newick_and_evaluate(v_to_newick(v_init), D)
                edge_bl = L_init / max(1, 2 * N - 3)
                init_nwk_bl = re.sub(r"(T\d+|\))(?=[^;]*;)", f"\\1:{edge_bl:.8f}", v_to_newick(v_init))
                with open(nwk_path, "w") as f:
                    f.write(init_nwk_bl + "\n")
                t0 = time.perf_counter()
                subprocess.run(
                    [fastme_bin, "-i", phy_path, "-u", nwk_path, "-s", "-T", "1", "-o", out_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                cli_ms = (time.perf_counter() - t0) * 1000.0
                with open(out_path, "r") as f:
                    cli_nwk = f.read().strip()
                L_cli, splits_cli = parse_newick_and_evaluate(cli_nwk, D)
                rf_vs_fme = len(splits_cli.symmetric_difference(splits_fme)) / (2.0 * (N - 3))
                rf_vs_p2v = len(splits_cli.symmetric_difference(splits_p2v)) / (2.0 * (N - 3))
                print(
                    f"      | External `fastme -u init.nwk -s`: L={L_cli:9.5f}, "
                    f"RF(CLI,FME)={rf_vs_fme:.4f}, RF(CLI,P2V)={rf_vs_p2v:.4f}, "
                    f"wall time (incl. I/O)={cli_ms:6.2f} ms"
                )


if __name__ == "__main__":
    main()
