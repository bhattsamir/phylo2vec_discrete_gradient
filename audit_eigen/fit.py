import ctypes as C, numpy as np, time
lib=C.CDLL("./libeig_gpu.so")
class Aln(C.Structure): _fields_=[("n_taxa",C.c_int32),("n_sites",C.c_int32),("n_words",C.c_int32),("bp",C.c_void_p),("names",C.c_void_p)]
class Par(C.Structure): _fields_=[("s",C.c_double*6),("alpha",C.c_double),("lp",C.c_double),("it",C.c_int32),("q_deg",C.c_int32),("alpha_sd",C.c_double)]
rng=np.random.default_rng(3); N,L=120,1596; root=rng.integers(0,4,L); seqs=[]
for i in range(N):
    p=rng.uniform(0.02,0.5); s=root.copy(); m=rng.random(L)<p; s[m]=rng.choice(4,m.sum()); t="".join("ACGT"[c] for c in s)
    if i%10==0: t="-"*800+t[800:]        # some partial sequences
    seqs.append(t)
seqs.append("-"*L)                        # an all-gap taxon -> NaN pairs
arr=(C.c_char_p*len(seqs))(*[s.encode() for s in seqs]); aln=Aln(); assert lib.encode_sequences_dna_bitplanes_c(arr,None,len(seqs),L,C.byref(aln))==0
for mode in (-1,1,2):
    out={}
    for name,f in (("cpu",lib.fit_eigen_gtr_map_c),("gpu",lib.fit_eigen_gtr_map_gpu_c)):
        p=Par(); t0=time.time(); rc=f(C.byref(aln),0,mode,C.c_uint64(7),C.byref(p)); out[name]=p
        print(f"mode {mode} {name}: rc={rc} alpha={p.alpha:.4f}±{p.alpha_sd:.4f} q_deg={p.q_deg} S={[round(x,3) for x in p.s]} logpost={p.lp:.3f} iters={p.it} ({time.time()-t0:.2f}s)")
