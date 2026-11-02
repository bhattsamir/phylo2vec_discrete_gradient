"""
verify_spr_coverage.py
----------------------
Brute-force checks of how the Phylo2Vec BME optimizer covers the unrooted SPR neighbourhood:
  1. Coverage: fraction of the 2(N-3)(2N-7) distinct unrooted SPR neighbours of a tree that are
     reached by single-coordinate moves 0 <= j <= 2i in the flip_mode = 0 and flip_mode = 1
     cherry-first gauges (Theorem 2.1), i.e. what Stages A-B can see before Stage C.
  2. Local optimality: every unrooted SPR neighbour of the optimize_phylo2vec_bme_c (and
     optimize_fastme_bme_c) output is scored with an independent Pauplin evaluation; the output
     must have no improving neighbour, and the reported BME length must match.
Exits non-zero if any coordinate move is not a single SPR, if the enumerated neighbourhood size
differs from 2(N-3)(2N-7), or if an optimizer output is not an exact SPR local optimum.
"""

import random
import sys
from collections import defaultdict

import numpy as np

from discrete_gradient import DiscreteGradient, v_to_ancestry_matrix


def rooted_clades(v: np.ndarray) -> list[frozenset[int]]:
    n_leaves = len(v) + 1
    M = v_to_ancestry_matrix(np.asarray(v, dtype=np.int64))
    clade = {x: frozenset([x]) for x in range(n_leaves)}
    for k, (c1, c2) in enumerate(M):
        clade[n_leaves + k] = clade[int(c1)] | clade[int(c2)]
    return list(clade.values())


def canonical_splits(clades, n_leaves: int, relabel=None) -> frozenset:
    """Unrooted topology as its set of non-trivial splits, each stored as the side without leaf 0."""
    full = frozenset(range(n_leaves))
    out = set()
    for c in clades:
        if relabel is not None:
            c = frozenset(int(relabel[x]) for x in c)
        if 2 <= len(c) <= n_leaves - 2:
            out.add(c if 0 not in c else full - c)
    return frozenset(out)


def unrooted_adjacency(v: np.ndarray) -> dict[int, set[int]]:
    n_leaves = len(v) + 1
    M = v_to_ancestry_matrix(np.asarray(v, dtype=np.int64))
    adj = defaultdict(set)
    for k, (c1, c2) in enumerate(M):
        p = n_leaves + k
        for c in (int(c1), int(c2)):
            adj[p].add(c)
            adj[c].add(p)
    root = 2 * n_leaves - 2
    a, b = adj.pop(root)
    adj[a].discard(root)
    adj[b].discard(root)
    adj[a].add(b)
    adj[b].add(a)
    return adj


def adjacency_splits(adj: dict[int, set[int]], n_leaves: int) -> frozenset:
    clades = []
    for u in adj:
        for w in adj[u]:
            if u < w:
                side, seen, stack = set(), {u, w}, [w]
                while stack:
                    x = stack.pop()
                    if x < n_leaves:
                        side.add(x)
                    for y in adj[x]:
                        if y not in seen:
                            seen.add(y)
                            stack.append(y)
                clades.append(frozenset(side))
    return canonical_splits(clades, n_leaves)


def spr_neighbours(v: np.ndarray) -> set[frozenset]:
    """All distinct unrooted trees one SPR move away from v (excluding v itself)."""
    n_leaves = len(v) + 1
    base = unrooted_adjacency(v)
    result = set()
    for p in list(base):
        if p < n_leaves:
            continue
        for c in list(base[p]):
            # Prune the c-side subtree at internal node p and suppress p
            adj = {k: set(s) for k, s in base.items()}
            adj[p].discard(c)
            adj[c].discard(p)
            x, y = adj.pop(p)
            adj[x].discard(p)
            adj[y].discard(p)
            adj[x].add(y)
            adj[y].add(x)
            rest, stack = {x}, [x]
            while stack:
                a = stack.pop()
                for b in adj[a]:
                    if b not in rest:
                        rest.add(b)
                        stack.append(b)
            # Regraft onto every edge of the remaining tree
            for a, b in {(a, b) for a in rest for b in adj[a] if a < b}:
                g = {k: set(s) for k, s in adj.items()}
                g[a].discard(b)
                g[b].discard(a)
                g[p] = {a, b, c}
                g[a].add(p)
                g[b].add(p)
                g[c].add(p)
                result.add(adjacency_splits(g, n_leaves))
    result.discard(canonical_splits(rooted_clades(v), n_leaves))
    return result


def coordinate_neighbours(v_cf: np.ndarray, perm: np.ndarray) -> set[frozenset]:
    """Unrooted trees reached by single-coordinate moves on v_cf, in the original leaf labels."""
    n_leaves = len(v_cf) + 1
    result = set()
    for i in range(n_leaves - 1):
        for j in range(2 * i + 1):
            if j == v_cf[i]:
                continue
            w = v_cf.copy()
            w[i] = j
            result.add(canonical_splits(rooted_clades(w), n_leaves, relabel=perm))
    return result


def pauplin_length(splits: frozenset, D: np.ndarray) -> float:
    """L(T) = sum_{x<y} D_xy 2^(1 - tau_xy), with tau_xy = 2 + #non-trivial splits separating x, y."""
    n_leaves = D.shape[0]
    member = np.zeros((max(len(splits), 1), n_leaves), dtype=bool)
    for k, s in enumerate(splits):
        member[k, list(s)] = True
    tau = 2 + (member[:, :, None] ^ member[:, None, :]).sum(axis=0)
    iu = np.triu_indices(n_leaves, 1)
    return float((D[iu] * 2.0 ** (1 - tau[iu])).sum())


def noisy_tree_distances(n_leaves: int, sigma: float, rng: random.Random) -> np.ndarray:
    """Additive distances on a random tree (branch lengths ~ U(0.05, 0.55)) with log-normal noise."""
    v = np.array([rng.randint(0, 2 * i) for i in range(n_leaves - 1)], dtype=np.int64)
    M = v_to_ancestry_matrix(v)
    adj = defaultdict(list)
    for k, (c1, c2) in enumerate(M):
        for c in (int(c1), int(c2)):
            w = rng.uniform(0.05, 0.55)
            adj[n_leaves + k].append((c, w))
            adj[c].append((n_leaves + k, w))
    D = np.zeros((n_leaves, n_leaves))
    for s in range(n_leaves):
        dist, stack = {s: 0.0}, [s]
        while stack:
            a = stack.pop()
            for b, w in adj[a]:
                if b not in dist:
                    dist[b] = dist[a] + w
                    stack.append(b)
        D[s] = [dist[t] for t in range(n_leaves)]
    Z = np.triu(np.array([[rng.gauss(0.0, 1.0) for _ in range(n_leaves)] for _ in range(n_leaves)]), 1)
    return D * np.exp(sigma * (Z + Z.T))


def main() -> int:
    dg = DiscreteGradient(backend="c")
    failures = 0

    print("1. Unrooted SPR neighbourhood reached by single-coordinate moves (30 random ordered trees per N)")
    rng = random.Random(3)
    for n_leaves in (8, 12, 16, 20):
        n_trials = 30
        expected = 2 * (n_leaves - 3) * (2 * n_leaves - 7)
        n_spr = n_g0 = n_g1 = n_both = n_full = n_outside = 0
        for _ in range(n_trials):
            v = np.array([rng.randint(0, i) for i in range(n_leaves - 1)], dtype=np.int64)
            spr = spr_neighbours(v)
            if len(spr) != expected:
                print(f"   FAIL: enumerated {len(spr)} SPR neighbours, expected {expected}")
                failures += 1
            gauges = [coordinate_neighbours(*dg.relabel_cherry_first(v, flip_mode=m)) for m in (0, 1)]
            orig = canonical_splits(rooted_clades(v), n_leaves)
            g0, g1 = (g - {orig} for g in gauges)
            both = g0 | g1
            n_spr += len(spr)
            n_g0 += len(g0 & spr)
            n_g1 += len(g1 & spr)
            n_both += len(both & spr)
            n_full += spr <= both
            n_outside += len(both - spr)
        failures += n_outside > 0
        print(
            f"   N={n_leaves:2d}: {n_spr // n_trials:5d} SPR neighbours | flip_mode=0 {n_g0 / n_spr:6.1%}  "
            f"flip_mode=1 {n_g1 / n_spr:6.1%}  both {n_both / n_spr:6.1%} | "
            f"fully covered {n_full}/{n_trials} | coordinate moves that are not one SPR: {n_outside}"
        )

    print("2. Exact SPR local optimality of optimizer outputs (15 runs per row, random v_init)")
    rng = random.Random(7)
    for sigma in (0.15, 0.5):
        for n_leaves in (10, 16, 24):
            n_trials = 15
            not_opt = {"p2v": 0, "fme": 0}
            mismatch = wins = ties = losses = 0
            for _ in range(n_trials):
                D = noisy_tree_distances(n_leaves, sigma, rng)
                v_init = np.array([rng.randint(0, 2 * i) for i in range(n_leaves - 1)], dtype=np.int64)
                lengths = {}
                for key, optimize in (("p2v", dg.optimize_phylo2vec_bme), ("fme", dg.optimize_fastme_bme)):
                    v_out, L_reported = optimize(v_init, D)[:2]
                    L = pauplin_length(canonical_splits(rooted_clades(v_out), n_leaves), D)
                    mismatch += abs(L - L_reported) > 1e-8
                    best_neighbour = min(pauplin_length(t, D) for t in spr_neighbours(v_out))
                    not_opt[key] += best_neighbour < L - 1e-9
                    lengths[key] = L
                wins += lengths["p2v"] < lengths["fme"] - 1e-9
                losses += lengths["p2v"] > lengths["fme"] + 1e-9
                ties += abs(lengths["p2v"] - lengths["fme"]) <= 1e-9
            failures += mismatch + not_opt["p2v"] + not_opt["fme"]
            print(
                f"   sigma={sigma:4.2f} N={n_leaves:2d}: not SPR-optimal P2V {not_opt['p2v']}/{n_trials}, "
                f"FastME {not_opt['fme']}/{n_trials} | reported-L mismatches {mismatch} | "
                f"P2V vs FastME win/tie/loss {wins}/{ties}/{losses}"
            )

    print("All checks passed." if failures == 0 else f"{failures} check(s) FAILED.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
