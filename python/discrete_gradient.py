import ctypes
import os
import subprocess
import sys
import numpy as np

try:
    import numba as nb
except ImportError:
    class _DummyNumba:
        def njit(self, *args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            def decorator(fn):
                return fn
            return decorator
    nb = _DummyNumba()


# ==============================================================================
# 1. PURE MATRIX-BASED PHYLO2VEC DISCRETE GRADIENT (NUMBA, V2 WITH L1-CACHE ND_U)
# ==============================================================================


@nb.njit(fastmath=True)
def v_to_ancestry_matrix(v: np.ndarray) -> np.ndarray:
    """
    Decode Phylo2Vec vector v (length N - 1) into the (N - 1, 2) merge matrix M,
    where row k corresponds to internal node (N + k) with children M[k, 0], M[k, 1].
    """
    n_pairs = len(v)
    n_leaves = n_pairs + 1
    n_nodes_r = 2 * n_leaves - 1

    pairs = np.empty((n_pairs, 2), dtype=np.int32)
    pairs[0, 0] = 0
    pairs[0, 1] = 1
    curr_len = 1
    for i in range(1, n_pairs):
        next_leaf = i + 1
        vi = v[i]
        if vi <= i:
            for k in range(curr_len, 0, -1):
                pairs[k, 0] = pairs[k - 1, 0]
                pairs[k, 1] = pairs[k - 1, 1]
            pairs[0, 0] = vi
            pairs[0, 1] = next_leaf
        else:
            idx = vi - curr_len
            c0 = pairs[idx - 1, 0]
            for k in range(curr_len, idx, -1):
                pairs[k, 0] = pairs[k - 1, 0]
                pairs[k, 1] = pairs[k - 1, 1]
            pairs[idx, 0] = c0
            pairs[idx, 1] = next_leaf
        curr_len += 1

    M = np.empty((n_pairs, 2), dtype=np.int32)
    parents_uf = np.arange(n_nodes_r, dtype=np.int32)

    for k in range(n_pairs):
        u = n_leaves + k
        p1 = pairs[k, 0]
        p2 = pairs[k, 1]
        while parents_uf[p1] != p1:
            parents_uf[p1] = parents_uf[parents_uf[p1]]
            p1 = parents_uf[p1]
        while parents_uf[p2] != p2:
            parents_uf[p2] = parents_uf[parents_uf[p2]]
            p2 = parents_uf[p2]
        M[k, 0] = p2
        M[k, 1] = p1
        parents_uf[p1] = u
        parents_uf[p2] = u

    return M


@nb.njit(fastmath=True)
def get_moves_and_destinations_matrix(M: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Single O(N) forward pass over the rows of M to compute:
      - v_map[leaf]: root of the subtree pruned when v[leaf - 1] changes
      - destinations[i, j]: target node where v_map[i + 1] regrafts when v[i] -> j
    """
    n_pairs = len(M)
    n_leaves = n_pairs + 1
    n_nodes_r = 2 * n_leaves - 1

    min_leaf = np.arange(n_nodes_r, dtype=np.int32)
    v_map = np.zeros(n_leaves, dtype=np.int32)
    destinations = np.empty((n_pairs, n_leaves), dtype=np.int32)
    for i in range(1, n_pairs):
        for j in range(i):
            destinations[i, j] = j

    for k in range(n_pairs):
        u = n_leaves + k
        c1 = M[k, 0]
        c2 = M[k, 1]
        ml1 = min_leaf[c1]
        ml2 = min_leaf[c2]
        if ml1 < ml2:
            l_min, l_max, c_move = ml1, ml2, c2
        else:
            l_min, l_max, c_move = ml2, ml1, c1
        min_leaf[u] = l_min
        v_map[l_max] = c_move
        if k < n_pairs - 1:
            for row in range(l_min + 1, l_max):
                destinations[row, l_min] = u

    return v_map, destinations


@nb.njit(fastmath=True)
def build_unrooted_matrices_and_subdist(
    M: np.ndarray, D: np.ndarray
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Pure forward and backward row passes over M (zero stacks, zero recursion) to compute:
      - parent_u: parent of each node in the unrooted tree
      - adj_u: (2N - 2, 3) adjacency matrix of the unrooted tree
      - subdist: (2N - 2, 2N - 2) matrix of balanced inter-subtree distances
      - nd_u: (2N - 2, 3) precomputed neighbor-pair distances for each internal node
    """
    n_pairs = len(M)
    n_leaves = n_pairs + 1
    n_nodes_u = 2 * n_leaves - 2
    root_u = n_nodes_u - 1

    rc1 = M[n_pairs - 1, 0]
    rc2 = M[n_pairs - 1, 1]
    other_root_child = rc1 if rc1 < rc2 else rc2
    root_u_c0 = M[n_pairs - 2, 0]
    root_u_c1 = M[n_pairs - 2, 1]
    root_u_c2 = other_root_child

    parent_u = np.empty(n_nodes_u, dtype=np.int32)
    sub_size = np.ones(n_nodes_u, dtype=np.int32)
    adj_u = np.full((n_nodes_u, 3), -1, dtype=np.int32)

    for k in range(n_pairs - 2):
        u = n_leaves + k
        c1 = M[k, 0]
        c2 = M[k, 1]
        parent_u[c1] = u
        parent_u[c2] = u
        sub_size[u] = 1 + sub_size[c1] + sub_size[c2]
        adj_u[u, 0] = c1
        adj_u[u, 1] = c2

    parent_u[root_u_c0] = root_u
    parent_u[root_u_c1] = root_u
    parent_u[root_u_c2] = root_u
    parent_u[root_u] = other_root_child

    for u in range(n_leaves):
        adj_u[u, 0] = parent_u[u]
    for u in range(n_leaves, root_u):
        adj_u[u, 2] = parent_u[u]
    adj_u[root_u, 0] = root_u_c0
    adj_u[root_u, 1] = root_u_c1
    adj_u[root_u, 2] = root_u_c2

    tin = np.empty(n_nodes_u, dtype=np.int32)
    dfs_order = np.empty(n_nodes_u, dtype=np.int32)
    tin[root_u] = 0
    tin[root_u_c0] = 1
    tin[root_u_c1] = 1 + sub_size[root_u_c0]
    tin[root_u_c2] = 1 + sub_size[root_u_c0] + sub_size[root_u_c1]

    for k in range(n_pairs - 3, -1, -1):
        u = n_leaves + k
        c1 = M[k, 0]
        c2 = M[k, 1]
        tin[c1] = tin[u] + 1
        tin[c2] = tin[u] + 1 + sub_size[c1]

    for u in range(n_nodes_u):
        dfs_order[tin[u]] = u

    subdist = np.zeros((n_nodes_u, n_nodes_u), dtype=np.float64)
    subdist[:n_leaves, :n_leaves] = D

    for k in range(n_pairs - 2):
        u = n_leaves + k
        c1 = M[k, 0]
        c2 = M[k, 1]
        for v_node in range(u):
            subdist[u, v_node] = 0.5 * (subdist[c1, v_node] + subdist[c2, v_node])
        for v_node in range(u):
            subdist[v_node, u] = subdist[u, v_node]

    for idx in range(tin[root_u_c0], tin[root_u_c0] + sub_size[root_u_c0]):
        v_node = dfs_order[idx]
        subdist[root_u, v_node] = 0.5 * (subdist[root_u_c1, v_node] + subdist[root_u_c2, v_node])
    for idx in range(tin[root_u_c1], tin[root_u_c1] + sub_size[root_u_c1]):
        v_node = dfs_order[idx]
        subdist[root_u, v_node] = 0.5 * (subdist[root_u_c0, v_node] + subdist[root_u_c2, v_node])
    for idx in range(tin[root_u_c2], tin[root_u_c2] + sub_size[root_u_c2]):
        v_node = dfs_order[idx]
        subdist[root_u, v_node] = 0.5 * (subdist[root_u_c0, v_node] + subdist[root_u_c1, v_node])
    for v_node in range(root_u):
        subdist[v_node, root_u] = subdist[root_u, v_node]

    for k in range(n_pairs - 3, -1, -1):
        p = n_leaves + k
        g = parent_u[p]
        c1 = M[k, 0]
        c2 = M[k, 1]
        for idx in range(tin[c1], tin[c1] + sub_size[c1]):
            v_node = dfs_order[idx]
            subdist[p, v_node] = 0.5 * (subdist[c2, v_node] + subdist[g, v_node])
        for idx in range(tin[c1], tin[c1] + sub_size[c1]):
            v_node = dfs_order[idx]
            subdist[v_node, p] = subdist[p, v_node]
        for idx in range(tin[c2], tin[c2] + sub_size[c2]):
            v_node = dfs_order[idx]
            subdist[p, v_node] = 0.5 * (subdist[c1, v_node] + subdist[g, v_node])
        for idx in range(tin[c2], tin[c2] + sub_size[c2]):
            v_node = dfs_order[idx]
            subdist[v_node, p] = subdist[p, v_node]

    for i in range(n_nodes_u):
        subdist[i, i] = 0.0

    # Precompute the 3 neighbor-pair distances for every internal node u >= n_leaves
    nd_u = np.empty((n_nodes_u, 3), dtype=np.float64)
    for u in range(n_leaves, root_u + 1):
        n0 = adj_u[u, 0]
        n1 = adj_u[u, 1]
        n2 = adj_u[u, 2]
        nd_u[u, 0] = subdist[n1, n2]
        nd_u[u, 1] = subdist[n0, n2]
        nd_u[u, 2] = subdist[n0, n1]

    return parent_u, adj_u, subdist, nd_u


@nb.njit(fastmath=True)
def _run_half_tree_bfs_nb(
    start_node: int,
    orig_node: int,
    parent_node: int,
    n_leaves: int,
    adj_u: np.ndarray,
    nd_u: np.ndarray,
    row_root: np.ndarray,
    row_orig: np.ndarray,
    edge_delta: np.ndarray,
    q_node: np.ndarray,
    q_prev: np.ndarray,
    q_pow: np.ndarray,
    q_dkc: np.ndarray,
    q_dist: np.ndarray,
):
    if start_node < n_leaves:
        return

    n0, n1, n2 = adj_u[start_node, 0], adj_u[start_node, 1], adj_u[start_node, 2]
    k_in = 0 if n0 == parent_node else (1 if n1 == parent_node else 2)
    k0 = 1 if k_in == 0 else 0
    k1 = 1 if k_in == 2 else 2

    B0 = adj_u[start_node, k0]
    B1 = adj_u[start_node, k1]
    D_AB = nd_u[start_node, k_in]
    d_kc_init = row_root[orig_node]

    head = 0
    tail = 0

    # Branch 0 (initial == 1)
    new_dist0 = 0.25 * (row_orig[B1] + row_root[B0] - D_AB - d_kc_init)
    edge_delta[start_node if start_node < B0 else B0] = new_dist0
    if B0 >= n_leaves:
        q_node[tail] = B0
        q_prev[tail] = start_node
        q_pow[tail] = 0.25
        q_dkc[tail] = 0.5 * (d_kc_init + row_root[B1])
        q_dist[tail] = new_dist0
        tail += 1

    # Branch 1 (initial == 1)
    new_dist1 = 0.25 * (row_orig[B0] + row_root[B1] - D_AB - d_kc_init)
    edge_delta[start_node if start_node < B1 else B1] = new_dist1
    if B1 >= n_leaves:
        q_node[tail] = B1
        q_prev[tail] = start_node
        q_pow[tail] = 0.25
        q_dkc[tail] = 0.5 * (d_kc_init + row_root[B0])
        q_dist[tail] = new_dist1
        tail += 1

    while head < tail:
        u = q_node[head]
        C = q_prev[head]
        pow_neg_len = q_pow[head]
        d_kc = q_dkc[head]
        curr_dist = q_dist[head]
        head += 1

        ki = 0 if adj_u[u, 0] == C else (1 if adj_u[u, 1] == C else 2)
        idx0 = 1 if ki == 0 else 0
        idx1 = 1 if ki == 2 else 2

        b0 = adj_u[u, idx0]
        b1 = adj_u[u, idx1]
        d_ab = nd_u[u, ki]
        r_b0 = row_root[b0]
        r_b1 = row_root[b1]

        # Branch 0: B = b0, A = b1
        d_ac0 = nd_u[u, idx0] + (row_orig[b1] - r_b1) * pow_neg_len
        ndist0 = curr_dist + 0.25 * (d_ac0 + r_b0 - d_ab - d_kc)
        edge_delta[u if u < b0 else b0] = ndist0
        if b0 >= n_leaves:
            q_node[tail] = b0
            q_prev[tail] = u
            q_pow[tail] = pow_neg_len * 0.5
            q_dkc[tail] = 0.5 * (d_kc + r_b1)
            q_dist[tail] = ndist0
            tail += 1

        # Branch 1: B = b1, A = b0
        d_ac1 = nd_u[u, idx1] + (row_orig[b0] - r_b0) * pow_neg_len
        ndist1 = curr_dist + 0.25 * (d_ac1 + r_b1 - d_ab - d_kc)
        edge_delta[u if u < b1 else b1] = ndist1
        if b1 >= n_leaves:
            q_node[tail] = b1
            q_prev[tail] = u
            q_pow[tail] = pow_neg_len * 0.5
            q_dkc[tail] = 0.5 * (d_kc + r_b0)
            q_dist[tail] = ndist1
            tail += 1


@nb.njit(fastmath=True)
def evaluate_spr_gradient_matrix(
    v: np.ndarray,
    v_map: np.ndarray,
    destinations: np.ndarray,
    parent_u: np.ndarray,
    adj_u: np.ndarray,
    subdist: np.ndarray,
    nd_u: np.ndarray,
) -> np.ndarray:
    """
    L1-cache-resident O(N^2) evaluation of BME objective changes for all Phylo2Vec perturbations.
    """
    n_pairs = len(v)
    n_leaves = n_pairs + 1
    n_nodes_u = 2 * n_leaves - 2

    edge_delta = np.zeros(n_nodes_u, dtype=np.float64)
    delta = np.zeros((n_pairs, n_pairs - 1), dtype=np.float64)

    q_node = np.empty(n_nodes_u, dtype=np.int32)
    q_prev = np.empty(n_nodes_u, dtype=np.int32)
    q_pow = np.empty(n_nodes_u, dtype=np.float64)
    q_dkc = np.empty(n_nodes_u, dtype=np.float64)
    q_dist = np.empty(n_nodes_u, dtype=np.float64)

    for v_index in range(2, n_leaves):
        tree_root = v_map[v_index]
        parent = parent_u[tree_root]

        other0, other1 = -1, -1
        for k in range(3):
            nb_node = adj_u[parent, k]
            if nb_node != -1 and nb_node != tree_root:
                if other0 == -1:
                    other0 = nb_node
                else:
                    other1 = nb_node
        if other0 == -1:
            continue

        edge_delta[tree_root] = 0.0
        edge_delta[parent if parent < other0 else other0] = 0.0
        edge_delta[parent if parent < other1 else other1] = 0.0

        row_root = subdist[tree_root]
        _run_half_tree_bfs_nb(
            other0, other1, parent, n_leaves, adj_u, nd_u,
            row_root, subdist[other1], edge_delta,
            q_node, q_prev, q_pow, q_dkc, q_dist,
        )
        _run_half_tree_bfs_nb(
            other1, other0, parent, n_leaves, adj_u, nd_u,
            row_root, subdist[other0], edge_delta,
            q_node, q_prev, q_pow, q_dkc, q_dist,
        )

        row_i = v_index - 1
        vi_val = v[row_i]
        for j in range(row_i):
            if vi_val == j:
                continue
            delta[row_i, j] = edge_delta[destinations[row_i, j]]

    return delta


@nb.njit(fastmath=True)
def calculate_discrete_gradient_numba(v: np.ndarray, D: np.ndarray) -> np.ndarray:
    for i in range(len(v)):
        if v[i] < 0 or v[i] > i:
            raise ValueError("Invalid Phylo2Vec vector entries; required 0 <= v[i] <= i.")
    M = v_to_ancestry_matrix(v)
    v_map, destinations = get_moves_and_destinations_matrix(M)
    parent_u, adj_u, subdist, nd_u = build_unrooted_matrices_and_subdist(M, D)
    return evaluate_spr_gradient_matrix(v, v_map, destinations, parent_u, adj_u, subdist, nd_u)


# ==============================================================================
# 2. UNIFIED CLASS WITH C99 SHARED LIBRARY BACKEND
# ==============================================================================


class DiscreteGradient:
    """
    Matrix-based Phylo2Vec BME Discrete Gradient calculator.
    Supports both 'c' (compiled C99 shared library via ctypes, fastest)
    and 'numba' (pure Python/Numba JIT).
    """

    def __init__(self, backend: str = "auto"):
        if backend not in ("auto", "c", "numba"):
            raise ValueError(f"Unknown backend '{backend}'; expected 'auto', 'c', or 'numba'.")
        self._c_lib = None
        here = os.path.dirname(os.path.abspath(__file__))
        c_src = os.path.join(here, "phylo2vec_fastme.c")
        if not os.path.exists(c_src):
            c_src = os.path.abspath(os.path.join(here, "..", "src", "phylo2vec_fastme.c"))
        is_macos = sys.platform == "darwin"
        lib_name = "libphylo2vec_fastme.dylib" if is_macos else "libphylo2vec_fastme.so"
        build_dir = os.path.abspath(os.path.join(here, "..", "build"))
        os.makedirs(build_dir, exist_ok=True)
        dylib_path = os.path.join(build_dir, lib_name)

        def _build_dylib() -> None:
            if is_macos:
                cmd = [
                    "/usr/bin/clang",
                    "-O3",
                    "-ffast-math",
                    "-arch",
                    "arm64",
                    "-arch",
                    "x86_64",
                    "-shared",
                    "-fPIC",
                    c_src,
                    "-o",
                    dylib_path,
                ]
            else:
                cc = os.environ.get("CC", "cc")
                cmd = [
                    cc,
                    "-O3",
                    "-march=native",
                    "-ffast-math",
                    "-shared",
                    "-fPIC",
                    c_src,
                    "-lm",
                    "-lpthread",
                    "-o",
                    dylib_path,
                ]
            subprocess.run(cmd, check=True)

        if (
            not os.path.exists(dylib_path)
            or (os.path.exists(c_src) and os.path.getmtime(c_src) > os.path.getmtime(dylib_path))
        ) and os.path.exists(c_src):
            _build_dylib()
        if os.path.exists(dylib_path):
            try:
                lib = ctypes.CDLL(dylib_path)
            except OSError:
                if os.path.exists(c_src):
                    _build_dylib()
                    lib = ctypes.CDLL(dylib_path)
                else:
                    raise
            lib.calculate_discrete_gradient_c.argtypes = [
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.calculate_discrete_gradient_c.restype = None

            lib.calculate_discrete_gradient_full_c.argtypes = [
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.calculate_discrete_gradient_full_c.restype = None

            lib.relabel_cherry_first_c.argtypes = [
                ctypes.POINTER(ctypes.c_int64),
                ctypes.c_int32,
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int32),
            ]
            lib.relabel_cherry_first_c.restype = None

            lib.optimize_phylo2vec_bme_c.argtypes = [
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.optimize_phylo2vec_bme_c.restype = ctypes.c_double

            lib.optimize_fastme_bme_c.argtypes = [
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int64),
                ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.optimize_fastme_bme_c.restype = ctypes.c_double
            self._c_lib = lib

        if backend == "c" and self._c_lib is None:
            raise RuntimeError("Requested backend='c', but libphylo2vec_fastme.dylib could not be loaded.")
        self.backend = "c" if (self._c_lib is not None and backend != "numba") else "numba"

    @staticmethod
    def _validate_v(v: np.ndarray, cherry_first_only: bool = False) -> np.ndarray:
        v_raw = np.asarray(v)
        if v_raw.ndim != 1 or len(v_raw) < 3:
            raise ValueError(f"Expected 1D Phylo2Vec vector of length N-1 >= 3, got shape {v_raw.shape}.")
        if np.issubdtype(v_raw.dtype, np.bool_) or not np.issubdtype(v_raw.dtype, np.integer):
            raise ValueError(f"Expected integer dtype for Phylo2Vec vector v, got dtype {v_raw.dtype}.")
        v_arr = np.ascontiguousarray(v_raw, dtype=np.int64)
        i_idx = np.arange(len(v_arr), dtype=np.int64)
        upper = i_idx if cherry_first_only else (2 * i_idx)
        if np.any(v_arr < 0) or np.any(v_arr > upper):
            bound_str = "0 <= v[i] <= i" if cherry_first_only else "0 <= v[i] <= 2*i"
            raise ValueError(f"Invalid Phylo2Vec vector entries; required {bound_str}.")
        return v_arr

    @classmethod
    def _validate_v_and_D(
        cls, v: np.ndarray, D: np.ndarray, cherry_first_only: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        v_arr = cls._validate_v(v, cherry_first_only=cherry_first_only)
        D_arr = np.ascontiguousarray(D, dtype=np.float64)
        n_leaves = len(v_arr) + 1
        if D_arr.ndim != 2 or D_arr.shape != (n_leaves, n_leaves):
            raise ValueError(f"Expected D of shape ({n_leaves}, {n_leaves}), got {D_arr.shape}.")
        if not np.all(np.isfinite(D_arr)):
            raise ValueError("Distance matrix D contains NaN or Inf entries.")
        if not np.allclose(D_arr, D_arr.T, rtol=1e-7, atol=1e-8):
            raise ValueError("Distance matrix D must be symmetric.")
        return v_arr, D_arr

    def _require_c_lib(self) -> ctypes.CDLL:
        if self._c_lib is None:
            raise RuntimeError("C shared library backend (libphylo2vec_fastme.dylib) is not available.")
        return self._c_lib

    def calculate_discrete_gradient(self, v: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Lower-triangular j < i discrete gradient for cherry-first Phylo2Vec vector v (0 <= v[k] <= k),
        shape: (N-1, N-2). For general vectors with v[k] > k, first relabel via relabel_cherry_first().
        """
        v_arr, D_arr = self._validate_v_and_D(v, D, cherry_first_only=True)
        if self.backend == "c":
            n_leaves = D_arr.shape[0]
            n_pairs = n_leaves - 1
            delta = np.empty((n_pairs, n_pairs - 1), dtype=np.float64)
            self._c_lib.calculate_discrete_gradient_c(
                v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                D_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int32(n_leaves),
                delta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            return delta
        return calculate_discrete_gradient_numba(v_arr, D_arr)

    def calculate_discrete_gradient_full(self, v: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Full 0 <= j <= 2i discrete gradient for cherry-first Phylo2Vec vector v (v_k <= k).
        Returns array of shape (N-1, 2N-3) where row i has valid entries at columns 0 <= j <= 2i.
        """
        v_arr, D_arr = self._validate_v_and_D(v, D, cherry_first_only=True)
        lib = self._require_c_lib()
        n_leaves = D_arr.shape[0]
        n_pairs = n_leaves - 1
        delta_full = np.empty((n_pairs, 2 * n_pairs - 1), dtype=np.float64)
        lib.calculate_discrete_gradient_full_c(
            v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            D_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int32(n_leaves),
            delta_full.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return delta_full

    def relabel_cherry_first(self, v: np.ndarray, flip_mode: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        O(N) cherry-first taxon relabeling: converts any v in V (0 <= v_i <= 2i) into
        an isomorphic cherry-first vector v_new (0 <= v_new[i] <= i) and leaf permutation perm.
        """
        v_arr = self._validate_v(v, cherry_first_only=False)
        lib = self._require_c_lib()
        n_pairs = len(v_arr)
        n_leaves = n_pairs + 1
        v_out = np.empty(n_pairs, dtype=np.int64)
        perm_out = np.empty(n_leaves, dtype=np.int32)
        lib.relabel_cherry_first_c(
            v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int32(n_leaves),
            ctypes.c_int32(flip_mode),
            v_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            perm_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
        return v_out, perm_out

    def optimize_phylo2vec_bme(self, v_init: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, float, int, float]:
        """
        Full Phylo2Vec BME tree optimizer:
        combines full 0 <= j <= 2i discrete gradients, multi-coordinate K-SPR batch updates,
        O(N) cherry-first taxon relabeling, and 2-SPR local-minimum escape kicks.
        Returns (v_opt, bme_length, n_grad_evals, elapsed_ms).
        """
        v_arr, D_arr = self._validate_v_and_D(v_init, D, cherry_first_only=False)
        lib = self._require_c_lib()
        n_leaves = D_arr.shape[0]
        v_out = np.empty(n_leaves - 1, dtype=np.int64)
        n_evals = ctypes.c_int32(0)
        elapsed_ms = ctypes.c_double(0.0)
        bme_len = lib.optimize_phylo2vec_bme_c(
            v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            D_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int32(n_leaves),
            v_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.byref(n_evals),
            ctypes.byref(elapsed_ms),
        )
        return v_out, float(bme_len), int(n_evals.value), float(elapsed_ms.value)

    def optimize_fastme_bme(self, v_init: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, float, int, float]:
        """
        Full unrooted FastME SPR hill-climber in C99 (evaluates all 2N-3 subtrees per step).
        Returns (v_opt, bme_length, n_iters, elapsed_ms).
        """
        v_arr, D_arr = self._validate_v_and_D(v_init, D, cherry_first_only=False)
        lib = self._require_c_lib()
        n_leaves = D_arr.shape[0]
        v_out = np.empty(n_leaves - 1, dtype=np.int64)
        n_iters = ctypes.c_int32(0)
        elapsed_ms = ctypes.c_double(0.0)
        bme_len = lib.optimize_fastme_bme_c(
            v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            D_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int32(n_leaves),
            v_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.byref(n_iters),
            ctypes.byref(elapsed_ms),
        )
        return v_out, float(bme_len), int(n_iters.value), float(elapsed_ms.value)
