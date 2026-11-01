"""
compare_with_fastme_cli.py
--------------------------
Compares:
  1. Phylo2Vec BME Optimizer (C99 / ctypes via DiscreteGradient.optimize_phylo2vec_bme)
  2. In-memory C99 FastME bmeSPR implementation (DiscreteGradient.optimize_fastme_bme)
  3. External `fastme` CLI binary (if installed via `conda install -c bioconda fastme` or `brew install fastme`)
     using both:
       - `fastme -i D.phy -u init.nwk -s` (SPR search from the exact same random starting tree)
       - `fastme -i D.phy -m B -s`        (Default BMEAdd greedy initialization + SPR refinement)
"""

import os
import shutil
import subprocess
import tempfile
import time
import numpy as np
from discrete_gradient import DiscreteGradient, v_to_ancestry_matrix


def v_to_newick(v: np.ndarray) -> str:
    """Convert a Phylo2Vec vector v (0 <= v_i <= 2i) into an unrooted Newick string with taxon names T0..T{N-1}."""
    M = v_to_ancestry_matrix(np.asarray(v, dtype=np.int64))
    n_pairs = len(v)
    n_leaves = n_pairs + 1
    sub = {i: f"T{i}" for i in range(n_leaves)}
    for k in range(n_pairs - 1):
        c1, c2, p = int(M[k, 0]), int(M[k, 1]), int(M[k, 2])
        sub[p] = f"({sub[c1]},{sub[c2]})"
    # Unrooted trifurcation at the top merger
    c1_top, c2_top, _ = int(M[n_pairs - 1, 0]), int(M[n_pairs - 1, 1]), int(M[n_pairs - 1, 2])
    rc_max = max(c1_top, c2_top)
    rc_min = min(c1_top, c2_top)
    # Expand rc_max into its two children for an unrooted trifurcating Newick root
    row_max = rc_max - n_leaves
    u0, u1 = int(M[row_max, 0]), int(M[row_max, 1])
    return f"({sub[u0]},{sub[u1]},{sub[rc_min]});"


def write_phylip_matrix(D: np.ndarray, path: str) -> None:
    """Write an N x N distance matrix D in standard PHYLIP lower/square format."""
    n = D.shape[0]
    with open(path, "w") as f:
        f.write(f"   {n}\n")
        for i in range(n):
            name = f"T{i}".ljust(10)
            vals = " ".join(f"{D[i, j]:.8f}" for j in range(n))
            f.write(f"{name}{vals}\n")


def main():
    dg = DiscreteGradient()
    fastme_bin = shutil.which("fastme")

    print("=" * 88)
    print("COMPARISON: Phylo2Vec BME Optimizer vs. C99 FastME-SPR vs. External `fastme` CLI")
    print("=" * 88)
    if fastme_bin:
        print(f"Found external fastme binary at: {fastme_bin}")
    else:
        print("Note: External `fastme` CLI binary not found on PATH.")
        print("      Running in-memory C99 `optimize_fastme_bme` (exact Desper & Gascuel 2002 bmeSPR)")
        print("      vs. `optimize_phylo2vec_bme`. To also run the external CLI binary, install via:")
        print("          conda install -c bioconda fastme   OR   brew install fastme\n")

    rng = np.random.default_rng(2026)
    for N in [20, 50, 100, 150, 200]:
        v_true = np.array([0] + [rng.integers(0, i + 1) for i in range(1, N - 1)], dtype=np.int64)
        # Generate random initial tree v_init
        v_init = np.array([0] + [rng.integers(0, i + 1) for i in range(1, N - 1)], dtype=np.int64)
        X = rng.normal(size=(N, 12))
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(D, 0.0)

        v_p2v, L_p2v, evals_p2v, ms_p2v = dg.optimize_phylo2vec_bme(v_init, D)
        v_fme, L_fme, iters_fme, ms_fme = dg.optimize_fastme_bme(v_init, D)

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
                with open(nwk_path, "w") as f:
                    f.write(v_to_newick(v_init) + "\n")
                t0 = time.perf_counter()
                subprocess.run(
                    [fastme_bin, "-i", phy_path, "-u", nwk_path, "-s", "-o", out_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                cli_ms = (time.perf_counter() - t0) * 1000.0
                print(f"      | External `fastme -u init.nwk -s` CLI wall time (incl. I/O): {cli_ms:6.2f} ms")


if __name__ == "__main__":
    main()

