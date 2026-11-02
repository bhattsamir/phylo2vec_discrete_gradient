import ctypes as C, numpy as np
lib=C.CDLL("./libeig_gpu.so")
class Aln(C.Structure): _fields_=[("n_taxa",C.c_int32),("n_sites",C.c_int32),("n_words",C.c_int32),("bp",C.c_void_p),("names",C.c_void_p)]
class Par(C.Structure): _fields_=[("s",C.c_double*6),("alpha",C.c_double),("lp",C.c_double),("it",C.c_int32),("q_deg",C.c_int32),("alpha_sd",C.c_double)]
rng=np.random.default_rng(3); N,L=256,1596
root=rng.integers(0,4,L); seqs=[]
for i in range(N):   # star-ish tree with a spread of divergences + some biased composition
    p=rng.uniform(0.0,0.6); s=root.copy(); m=rng.random(L)<p; s[m]=rng.choice(4,m.sum(),p=[.4,.1,.1,.4]); seqs.append("".join("ACGT"[c] for c in s))
arr=(C.c_char_p*N)(*[s.encode() for s in seqs]); aln=Aln()
assert lib.encode_sequences_dna_bitplanes_c(arr,None,N,L,C.byref(aln))==0
for mode,al in ((1,1.0),(2,1.0),(2,0.3),(0,1.0)):
    par=Par((C.c_double*6)(1,2,1,1,2,1),al,0,0)
    Dc=np.zeros((N,N)); Dg=np.zeros((N,N))
    lib.compute_eigen_distance_matrix_cpu_c(C.byref(aln),C.byref(par),mode,Dc.ctypes.data_as(C.c_void_p),None)
    rc=lib.compute_eigen_distance_matrix_gpu_c(C.byref(aln),C.byref(par),mode,1,Dg.ctypes.data_as(C.c_void_p))
    iu=np.triu_indices(N,1); ok=(Dc[iu]<5)&np.isfinite(Dc[iu])
    rel=np.abs(Dg[iu]-Dc[iu])[ok]/np.maximum(Dc[iu][ok],1e-12)
    asym=np.abs(Dg-Dg.T).max(); nasym=(Dg!=Dg.T).sum()//2
    print(f"mode {mode} alpha {al}: rc={rc} GPU-f32 vs CPU-f64 rel err median {np.median(rel):.1e} max {rel.max():.1e} | GPU asym max {asym:.2e} ({nasym} of {len(iu[0])} pairs differ) | CPU zeros {int((Dc[iu]==0).sum())} GPU zeros {int((Dg[iu]==0).sum())}")
