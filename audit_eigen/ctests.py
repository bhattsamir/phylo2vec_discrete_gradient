import ctypes as C, numpy as np
lib = C.CDLL("./libeig.so")
class St(C.Structure): _fields_=[("K",C.c_double*16),("Fa",C.c_double*4),("Fb",C.c_double*4),("xi",C.c_double*4),("sigma",C.c_double*4),("valid",C.c_double)]
class Par(C.Structure): _fields_=[("s",C.c_double*6),("alpha",C.c_double),("lp",C.c_double),("it",C.c_int32),("q_deg",C.c_int32),("alpha_sd",C.c_double)]
def planes(seq):
    L=len(seq); nw=(L+63)//64; p=np.zeros((4,nw),np.uint64)
    for k,ch in enumerate(seq):
        c="ACGT".find(ch)
        if c>=0: p[c,k>>6]|=np.uint64(1)<<np.uint64(k&63)
    return p, nw
def dist(sa,sb,mode,alpha=1.0):
    pa,nw=planes(sa); pb,_=planes(sb); st=St()
    lib.compute_pair_suff_stats_c(pa.ctypes.data_as(C.c_void_p),pb.ctypes.data_as(C.c_void_p),C.c_int32(nw),C.c_int32(mode),C.byref(st))
    par=Par((C.c_double*6)(*[1]*6),alpha,0,0); lib.eval_pair_distance_from_stats_c.restype=C.c_double
    return lib.eval_pair_distance_from_stats_c(C.byref(st),C.byref(par),C.c_int32(mode),None), list(st.sigma)
rng=np.random.default_rng(0); r=lambda L:"".join(rng.choice(list("ACGT"),L))
print("A) no shared sites:", [round(dist("ACGT"*25+"-"*100,"-"*100+"ACGT"*25,m)[0],4) for m in (0,1,2)])
print("   identical seqs  :", [round(dist(s:=r(2000),s,m)[0],8) for m in (0,1,2)])
for L in (2000, 20000):
    a,b=r(L),r(L)   # unrelated sequences: saturated pair
    for al in (1.0,0.5,0.2):
        print(f"B) unrelated L={L} alpha={al}: SVD-sum D={dist(a,b,1,al)[0]:.3g}  WLS D={dist(a,b,2,al)[0]:.3g}")
a=r(3000); b=list(a)
for i in rng.choice(3000,300,replace=False): b[i]=rng.choice(list("ACGT"))
b="".join(b); a_noT=a.replace("T","-"); 
print("C) 7.5% diverged pair, SVD-sum D:", round(dist(a,b,1)[0],4), "| same pair, sequence a has no T (T->gap):", round(dist(a_noT,b,1)[0],4))
