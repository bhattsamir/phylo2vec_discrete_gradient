"""
eigen_distance.py

Python / NumPy / ctypes interface to the C99 + Multi-GPU CUDA implementation
of Penn et al. (2025) "Bayesian Inference of Phylogenetic Distances: Revisiting
the Eigenvalue Approach" (Bull. Math. Biol. 87:32, 2025).

Supports:
  - "paper_orig"    (0): Original Eq. (5)-(6) with (+1/2, -3/2) exponents
  - "symmetric_svd" (1): Canonical (-1/2, -1/2) contingency SVD + rank-1 deflation
  - "symmetric_wls" (2): Canonical (-1/2, -1/2) SVD + inverse-variance WLS weighting
"""

import ctypes
import os
import subprocess
from typing import Dict, List, Optional, Tuple
import numpy as np


class _DnaBitplaneAlignmentC(ctypes.Structure):
    _fields_ = [
        ("n_taxa", ctypes.c_int32),
        ("n_sites", ctypes.c_int32),
        ("n_words", ctypes.c_int32),
        ("bitplanes", ctypes.POINTER(ctypes.c_uint64)),
        ("taxon_names", ctypes.POINTER(ctypes.c_char_p)),
    ]


class _EigenGtrParamsC(ctypes.Structure):
    _fields_ = [
        ("s_rates", ctypes.c_double * 6),
        ("alpha", ctypes.c_double),
        ("log_post", ctypes.c_double),
        ("n_iters", ctypes.c_int32),
    ]


MODE_MAP = {
    "paper_orig": 0,
    "symmetric_svd": 1,
    "symmetric_wls": 2,
}


class EigenDistanceEngine:
    def __init__(self, lib_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if lib_path is None:
            lib_path = os.path.join(base_dir, "libphylo2vec_eigen.so")
        if not os.path.exists(lib_path):
            self._build_shared_lib(base_dir, lib_path)
        self.lib = ctypes.CDLL(lib_path)
        self._bind_functions()

    @staticmethod
    def _build_shared_lib(base_dir: str, out_so: str) -> None:
        c_src = os.path.join(base_dir, "phylo2vec_eigen_dist.c")
        cu_src = os.path.join(base_dir, "phylo2vec_eigen_dist_cuda.cu")
        nvcc = "/usr/local/cuda-12.8/bin/nvcc"
        if not os.path.exists(nvcc):
            nvcc = "nvcc"
        c_obj = os.path.join(base_dir, "phylo2vec_eigen_dist.o")
        cu_obj = os.path.join(base_dir, "phylo2vec_eigen_dist_cuda.o")
        subprocess.run(
            ["gcc", "-O3", "-march=native", "-fopenmp", "-fPIC", "-c", c_src, "-o", c_obj],
            check=True,
        )
        subprocess.run(
            [nvcc, "-O3", "-arch=sm_86", "-Xcompiler", "-fPIC -fopenmp", "-c", cu_src, "-o", cu_obj],
            check=True,
        )
        subprocess.run(
            [nvcc, "-O3", "-arch=sm_86", "-shared", "-Xcompiler", "-fopenmp", c_obj, cu_obj, "-o", out_so],
            check=True,
        )

    def _bind_functions(self) -> None:
        self.lib.load_fasta_dna_bitplanes_c.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(_DnaBitplaneAlignmentC),
        ]
        self.lib.load_fasta_dna_bitplanes_c.restype = ctypes.c_int

        self.lib.free_dna_bitplane_alignment_c.argtypes = [
            ctypes.POINTER(_DnaBitplaneAlignmentC),
        ]
        self.lib.free_dna_bitplane_alignment_c.restype = None

        self.lib.fit_eigen_gtr_map_c.argtypes = [
            ctypes.POINTER(_DnaBitplaneAlignmentC),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.POINTER(_EigenGtrParamsC),
        ]
        self.lib.fit_eigen_gtr_map_c.restype = ctypes.c_int

        self.lib.fit_eigen_gtr_map_gpu_c.argtypes = [
            ctypes.POINTER(_DnaBitplaneAlignmentC),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_uint64,
            ctypes.POINTER(_EigenGtrParamsC),
        ]
        self.lib.fit_eigen_gtr_map_gpu_c.restype = ctypes.c_int

        self.lib.compute_eigen_distance_matrix_cpu_c.argtypes = [
            ctypes.POINTER(_DnaBitplaneAlignmentC),
            ctypes.POINTER(_EigenGtrParamsC),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.compute_eigen_distance_matrix_cpu_c.restype = None

        self.lib.compute_eigen_distance_matrix_gpu_c.argtypes = [
            ctypes.POINTER(_DnaBitplaneAlignmentC),
            ctypes.POINTER(_EigenGtrParamsC),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.compute_eigen_distance_matrix_gpu_c.restype = ctypes.c_int

        self.lib.compute_eigen_subset_distance_matrix_gpu_c.argtypes = [
            ctypes.POINTER(_DnaBitplaneAlignmentC),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(_EigenGtrParamsC),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.compute_eigen_subset_distance_matrix_gpu_c.restype = ctypes.c_int

    def fit_and_compute_from_fasta(
        self,
        fasta_path: str,
        mode: str = "symmetric_wls",
        n_sample_pairs: int = 50000,
        max_taxa: int = 0,
        use_gpu: bool = True,
        num_gpus: int = 0,
        seed: int = 2025,
    ) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
        mode_id = MODE_MAP.get(mode, 2)
        aln = _DnaBitplaneAlignmentC()
        rc = self.lib.load_fasta_dna_bitplanes_c(
            fasta_path.encode("utf-8"), int(max_taxa), ctypes.byref(aln)
        )
        if rc != 0:
            raise RuntimeError(f"Failed to load FASTA '{fasta_path}' (rc={rc})")

        try:
            params = _EigenGtrParamsC()
            if use_gpu:
                self.lib.fit_eigen_gtr_map_gpu_c(
                    ctypes.byref(aln), int(n_sample_pairs), mode_id, int(seed), ctypes.byref(params)
                )
            else:
                self.lib.fit_eigen_gtr_map_c(
                    ctypes.byref(aln), int(n_sample_pairs), mode_id, int(seed), ctypes.byref(params)
                )

            N = int(aln.n_taxa)
            D = np.empty((N, N), dtype=np.float64)
            d_ptr = D.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            if use_gpu:
                self.lib.compute_eigen_distance_matrix_gpu_c(
                    ctypes.byref(aln), ctypes.byref(params), mode_id, int(num_gpus), d_ptr
                )
            else:
                self.lib.compute_eigen_distance_matrix_cpu_c(
                    ctypes.byref(aln), ctypes.byref(params), mode_id, d_ptr, None
                )

            names = [
                aln.taxon_names[i].decode("utf-8", errors="replace")
                for i in range(N)
            ]
            info = {
                "alpha": float(params.alpha),
                "s_rates": [float(params.s_rates[k]) for k in range(6)],
                "log_post": float(params.log_post),
                "n_iters": int(params.n_iters),
                "n_taxa": N,
                "n_sites": int(aln.n_sites),
                "mode": mode,
            }
            return D, names, info
        finally:
            self.lib.free_dna_bitplane_alignment_c(ctypes.byref(aln))

