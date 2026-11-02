import numpy as np
exec(open("math_checks.py").read().split("mx = {")[0])
rng = np.random.default_rng(7)
# 1) typical Mode-0 bias at L=inf for realistic base composition (pi ~ Dir(20))
rel = {"code0": [], "paper0": []}
for _ in range(500):
    pi = rng.dirichlet([20]*4); S6 = rng.lognormal(0,0.5,6); a = rng.uniform(0.3,2); t = rng.uniform(0.05,0.8)
    Q = gtr(S6, pi); P, lam = Pgamma(Q, pi, t, a); K = np.diag(pi)@P; Fa, Fb = K.sum(1), K.sum(0)
    A = lambda F,p,q: np.diag(F**p)@K@np.diag(F**q); trL=-np.trace(Q)
    est = lambda s: sum(a*(min(x,1)**(-1/a)-1) for x in s)/trL
    rel["code0"].append(est(sv(0.5*(A(Fa,.5,-1.5)+A(Fb,-1.5,.5))))/t-1)
    rel["paper0"].append(est(sv(0.5*(A(Fa,.5,-1.5)+A(Fb,.5,-1.5))))/2/t-1)
for k,v in rel.items(): v=np.array(v); print(f"L=inf {k}: median rel bias {np.median(v):+.3f}, 90% range [{np.percentile(v,5):+.3f},{np.percentile(v,95):+.3f}]")
# 2) finite-sample: SVD-sum vs WLS(plug-in) vs WLS(model weights) vs Rayleigh quotient
def simulate(pi,Q,a,t,L):
    P,_ = Pgamma(Q,pi,t,a); x = rng.choice(4,L,p=pi)
    # per-site gamma rates
    r = rng.gamma(a,1/a,L); out=np.empty(L,int)
    Dh=np.diag(np.sqrt(pi)); R=Dh@Q@np.linalg.inv(Dh); lam,V=np.linalg.eigh((R+R.T)/2)
    for s in range(L):
        Ps = np.linalg.inv(Dh)@V@np.diag(np.exp(lam*t*r[s]))@V.T@Dh; Ps=np.clip(Ps[x[s]],0,None); out[s]=rng.choice(4,p=Ps/Ps.sum())
    K = np.zeros((4,4)); np.add.at(K,(x,out),1); return K/L
pi=np.array([.35,.15,.2,.3]); S6=np.array([1,4,1,1,4,1.]); a=0.5; Q=gtr(S6,pi)
Dh=np.diag(np.sqrt(pi)); R=Dh@Q@np.linalg.inv(Dh); lam,V=np.linalg.eigh((R+R.T)/2); lam=lam[::-1]; V=V[:,::-1]  # lam0=0
nl=-lam[1:]
for t in (0.1,0.5,1.5):
  res={k:[] for k in ("svd_sum","wls_plugin","wls_model","rayleigh")}
  for rep in range(150):
    K=simulate(pi,Q,a,t,1500)+1e-12; Fa,Fb=K.sum(1),K.sum(0); Ls=np.diag(Fa**-.5)@K@np.diag(Fb**-.5)
    s=np.clip(sv(Ls-np.outer(np.sqrt(Fa),np.sqrt(Fb)))[:3],1e-4,1-1e-12); y=a*(s**(-1/a)-1)
    res["svd_sum"].append(y.sum()/nl.sum())
    w=s**(2+2/a)/(1-s**2+1e-4); res["wls_plugin"].append((w*nl*y).sum()/(w*nl*nl).sum())
    d=y.sum()/nl.sum()
    for _ in range(5): m=(1+d*nl/a)**(-a); w=m**(2+2/a)/(1-m**2+1e-4); d=(w*nl*y).sum()/(w*nl*nl).sum()
    res["wls_model"].append(d)
    Lsym=(Ls+Ls.T)/2; rq=np.clip(np.array([V[:,k]@Lsym@V[:,k] for k in (1,2,3)]),1e-4,1); yr=a*(rq**(-1/a)-1)
    res["rayleigh"].append(yr.sum()/nl.sum())
  print(f"t={t}: "+"  ".join(f"{k} bias {np.mean(v)/t-1:+.3f} rmse {np.sqrt(np.mean((np.array(v)-t)**2))/t:.3f}" for k,v in res.items()))
