from pathlib import Path
import ctypes,json,random,itertools
import numpy as np
R=Path(__file__).resolve().parent
# Load previous independent topology/profile helpers without executing its tests.
s=(R/'profile_checks.py').read_text();exec(s[:s.index('res={};rng=')])
g=ctypes.CDLL(str(R/'acceptance_helpers.so'));lp=ctypes.POINTER(ctypes.c_int64)
g.labels.argtypes=[ctypes.c_int,ip,ip,ctypes.c_int,lp,ip]
g.distances.argtypes=[ctypes.c_int,ctypes.c_int,lp,ip,ip]
g.acceptance.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,lp,ip,lp,ip,fp,dp,ctypes.c_double,dp,ip,ip];g.acceptance.restype=ctypes.c_double

def labels(t,flip=0):
 v=np.zeros(t.n-1,np.int64);p=np.zeros(t.n,np.int32);g.labels(t.n,t.a.ctypes.data_as(ip),t.b.ctypes.data_as(ip),flip,v.ctypes.data_as(lp),p.ctypes.data_as(ip));return v,p

def gd(t,backend,flip=0):
 v,p=labels(t,flip);d=np.zeros((t.n,t.n),np.int32);g.distances(backend,t.n,v.ctypes.data_as(lp),p.ctypes.data_as(ip),d.ctypes.data_as(ip));return d

def refd(t):
 d=np.zeros((t.n,t.n),np.int32)
 for i in range(t.n):
  q=[(i,-1,0)]
  for u,p,k in q:
   if u<t.n:d[i,u]=k
   q.extend((v,u,k+1) for v in t.adj[u] if v!=p)
 return d

def check_guard(t0,t1,cap,full,lens,exlens,backend,surrogate=None,flip=0):
 v0,p0=labels(t0,flip);v1,p1=labels(t1,1-flip);off=np.r_[0,np.cumsum(lens)].astype(np.int32);exoff=np.r_[0,np.cumsum(exlens)].astype(np.int32);cap=np.ascontiguousarray(cap,np.float32);full=np.ascontiguousarray(full,np.float64)
 delta=score(t1.t,cap)-score(t0.t,cap) if surrogate is None else surrogate
 checked=ctypes.c_double();refined=ctypes.c_int32();restored=ctypes.c_int32()
 ret=g.acceptance(backend,t0.n,off.ctypes.data_as(ip),exoff.ctypes.data_as(ip),v0.ctypes.data_as(lp),p0.ctypes.data_as(ip),v1.ctypes.data_as(lp),p1.ctypes.data_as(ip),cap.ctypes.data_as(fp),full.ctypes.data_as(dp),delta,ctypes.byref(checked),ctypes.byref(refined),ctypes.byref(restored))
 return {'return':ret,'checked_delta':checked.value,'refined_pairs':refined.value,'restored_initial_vector_and_permutation':bool(restored.value),'surrogate_delta':delta,'true_delta':score(t1.t,full)-score(t0.t,full)}

res={};rng=random.Random(2309);worst=0.;worsto=0.;count=0;outside=0;top_count=0
for n in [4,5,9,17,32,64,129]:
 for repeat in range(8):
  xs=list(range(n));rng.shuffle(xs)
  while len(xs)>1:
   a=xs.pop(rng.randrange(len(xs)));b=xs.pop(rng.randrange(len(xs)));xs.append((a,b))
  t=Tree(xs[0]);truth=refd(t)
  for backend in [0,1]:
   for flip in [0,1]:assert np.array_equal(gd(t,backend,flip),truth);top_count+=1
  for u in range(2*n-1):
   ex=np.zeros(n)
   for j,w in weights(t.sub[u]).items():ex[j]=w
   for backend in [0,1]:
    p,_,_=dense_profile(t,u,-1,backend);worst=max(worst,float(abs(p-ex).max()));count+=1
   if u!=t.root:
    ex=np.zeros(n)
    for j,w in t.exact_out(u).items():ex[j]=w
    p,_,_=dense_profile(t,u,-1,outside=True);worsto=max(worsto,float(abs(p-ex).max()));outside+=1
res['new_dfs_profiles']={'clade_checks':count,'clade_max_error':worst,'outside_checks':outside,'outside_max_error':worsto};res['topological_distance_checks']=top_count
assert worst==0 and worsto==0
# Same marked sets as the earlier capped-profile failure, held fixed for exact check.
A=balanced(list(range(4096)));t=Tree(((A,4096),(4097,balanced(list(range(4098,4102))))));u=next(k for k,x in t.sub.items() if x==A)
st0=Tree(((0,1),(2,((3,4),(5,6)))));st1=Tree(((0,2),(1,((3,4),(5,6)))))
res['old_counterexample_acceptance']=[]
for backend,cap in [(0,64),(1,128)]:
 cl,cw=t.profile(u,cap,backend);el,ew=t.profile(u,-1,backend);marked=set(map(int,cl))|{4097,4098,4099}
 flag=np.array([j in marked for j in range(t.n)]);D=np.ascontiguousarray(2.+(flag[:,None]!=flag[None,:]));np.fill_diagonal(D,0)
 single=[(np.array([j],np.int32),np.ones(1,np.float32)) for j in range(4096,4102)]
 Dc=condense(D,[(cl,cw)]+single);De=condense(D,[(el,ew)]+single)
 result=check_guard(st0,st1,Dc,De,[len(cl)]+[1]*6,[len(el)]+[1]*6,backend);result.update(backend=backend,cap=cap);res['old_counterexample_acceptance'].append(result)
 assert result['return']==0 and result['restored_initial_vector_and_permutation'] and result['checked_delta']==result['true_delta']
 # A helpful reverse move must pass even if the surrogate supplied an improvement.
 reverse=check_guard(st1,st0,Dc,De,[len(cl)]+[1]*6,[len(el)]+[1]*6,backend,surrogate=-1.)
 assert reverse['return']==reverse['true_delta']<0 and not reverse['restored_initial_vector_and_permutation']
 result['reverse_control']=reverse
 del D
# Isolate changed-pair membership and permutations. Integer distances eliminate FP32 ambiguities.
num=0;error=0.;accepted=0;rejected=0
rg=np.random.default_rng(123)
for n in [4,7,12,25]:
 for rep in range(30):
  ts=[]
  for z in range(2):
   pool=list(range(n));rng.shuffle(pool)
   while len(pool)>1:pool.append((pool.pop(rng.randrange(len(pool))),pool.pop(rng.randrange(len(pool)))))
   ts.append(Tree(pool[0]))
  D=rg.integers(1,50,(n,n));D=(D+D.T).astype(float);np.fill_diagonal(D,0)
  Dc=D.copy();capped=rg.random(n)<.5;capped[0]=True
  for i in range(n):
   for j in range(i+1,n):
    if capped[i] or capped[j]:Dc[i,j]=Dc[j,i]=rg.integers(1,100)
  lens=np.ones(n,np.int32);exlens=lens+capped
  for backend in [0,1]:
   r=check_guard(*ts,Dc,D,lens,exlens,backend,surrogate=-1.,flip=rep%2);num+=1;truth=r['true_delta'];error=max(error,abs(r['checked_delta']-truth))
   assert (np.isnan(r['checked_delta']) and truth==0 and r['return']==0) or abs(r['checked_delta']-truth)<1e-10, (n,rep,backend,r)
   assert (r['return']<0)==(truth < -1e-6)
   if truth < -1e-6:accepted+=1
   else:rejected+=1;assert r['restored_initial_vector_and_permutation']
res['guard_controls']={'checks':num,'accepted':accepted,'rejected':rejected,'max_delta_error':error}
(R/'acceptance_checks.json').write_text(json.dumps(res,indent=2,default=lambda x:x.item())+'\n');print(json.dumps(res,default=lambda x:x.item()))
