import ctypes as C, numpy as np
lib=C.CDLL("./libeig_gpu.so")
class Aln(C.Structure): _fields_=[("n_taxa",C.c_int32),("n_sites",C.c_int32),("n_words",C.c_int32),("bp",C.c_void_p),("names",C.c_void_p)]
class Par(C.Structure): _fields_=[("s",C.c_double*6),("alpha",C.c_double),("lp",C.c_double),("it",C.c_int32),("q_deg",C.c_int32),("alpha_sd",C.c_double)]
rng=np.random.default_rng(5); L=3000
def mut(s,p,probs=None):
    s=list(s)
    for i in np.where(rng.random(len(s))<p)[0]: s[i]=rng.choice(list("ACGT"),p=probs)
    return "".join(s)
r=lambda n:"".join(rng.choice(list("ACGT"),n))
a=r(L); b=mut(a,0.1)
seqs={"a":a,"b":b,"a_noT":a.replace("T","-"),"b_noT":b.replace("T","-"),"a_onlyAC":"".join(c if c in "AC" else "-" for c in a),
      "b_onlyAC":"".join(c if c in "AC" else "-" for c in b),"polyA":"A"*L,"polyA2":mut("A"*L,0.01),"half1":a[:L//2]+"-"*(L-L//2),
      "half2":"-"*(L//2)+b[L//2:],"unrel":r(L),"ident":a}
names=list(seqs); N=len(names); arr=(C.c_char_p*N)(*[seqs[k].encode() for k in names]); aln=Aln()
assert lib.encode_sequences_dna_bitplanes_c(arr,None,N,L,C.byref(aln))==0
pairs=[("a","b"),("a_noT","b"),("a_noT","b_noT"),("a_onlyAC","b_onlyAC"),("a_onlyAC","b"),("polyA","polyA2"),("half1","half2"),("half1","b"),("a","unrel"),("a","ident")]
for mode in (0,1,2):
    par=Par((C.c_double*6)(1,2,1,1,2,1),0.5,0,0); Dc=np.zeros((N,N)); Dg=np.zeros((N,N))
    lib.compute_eigen_distance_matrix_cpu_c(C.byref(aln),C.byref(par),mode,Dc.ctypes.data_as(C.c_void_p),None)
    lib.compute_eigen_distance_matrix_gpu_c(C.byref(aln),C.byref(par),mode,1,Dg.ctypes.data_as(C.c_void_p))
    ix={k:i for i,k in enumerate(names)}
    print(f"mode {mode}: "+"  ".join(f"{p}-{q}={Dc[ix[p],ix[q]]:.4g}/{Dg[ix[p],ix[q]]:.4g}" for p,q in pairs))
