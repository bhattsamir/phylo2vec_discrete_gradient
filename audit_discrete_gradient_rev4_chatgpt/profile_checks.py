from pathlib import Path
import ctypes,json,random,sys
import numpy as np
R=Path(__file__).resolve().parent
lib=ctypes.CDLL(str(R/'profile_helpers.so'));ip=ctypes.POINTER(ctypes.c_int32);fp=ctypes.POINTER(ctypes.c_float);dp=ctypes.POINTER(ctypes.c_double)
lib.profile_clade.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ctypes.c_int,ctypes.c_int,ip,fp];lib.profile_clade.restype=ctypes.c_int
lib.profile_out.argtypes=[ctypes.c_int,ip,ip,ctypes.c_int,ctypes.c_int,ip,fp];lib.profile_out.restype=ctypes.c_int
lib.condense.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,fp,dp,dp]
s=(R/'representative_check.py').read_text();scope={'np':np};exec(s[s.index('def weights'):s.index("res={'full_BME")],scope);weights,score=scope['weights'],scope['score']
def ladder(xs):
 t=xs[0]
 for x in xs[1:]:t=(t,x)
 return t
def balanced(xs):
 if len(xs)==1:return xs[0]
 k=len(xs)//2;return (balanced(xs[:k]),balanced(xs[k:]))
class Tree:
 def __init__(self,t):
  self.t=t;self.n=len(weights(t));n=self.n;self.a=np.zeros(2*n-1,dtype=np.int32);self.b=self.a.copy();self.sub={};self.next=n
  def build(x):
   if isinstance(x,int):self.sub[x]=x;return x
   l,r=(build(c) for c in x);u=self.next;self.next+=1;self.a[u]=l;self.b[u]=r;self.sub[u]=x;return u
  self.root=build(t);self.adj={i:[] for i in range(2*n-1)}
  for u in range(n,2*n-1):
   for c in (int(self.a[u]),int(self.b[u])):self.adj[u].append(c);self.adj[c].append(u)
  l,r=self.adj.pop(self.root);self.adj[l].remove(self.root);self.adj[r].remove(self.root);self.adj[l].append(r);self.adj[r].append(l)
 def profile(self,u,cap=0,backend=0,outside=False):
  l=np.empty(2*self.n,dtype=np.int32);w=np.empty(2*self.n,dtype=np.float32);args=[self.n,self.a.ctypes.data_as(ip),self.b.ctypes.data_as(ip),u,cap,l.ctypes.data_as(ip),w.ctypes.data_as(fp)]
  count=lib.profile_out(*args) if outside else lib.profile_clade(backend,*args)
  return l[:count].copy(),w[:count].copy()
 def exact_out(self,u):
  inside=set(weights(self.sub[u]));outside=set(range(self.n))-inside
  # Only one neighbor lies outside u's rooted clade.
  starts=[]
  for v in self.adj[u]:
   todo=[(v,u)];hits=[]
   while todo:
    x,p=todo.pop()
    if x<self.n:hits.append(x)
    todo.extend((y,x) for y in self.adj[x] if y!=p)
   if set(hits)==outside:starts.append(v)
  assert len(starts)==1,(u,starts)
  ans={};todo=[(starts[0],u,1.)]
  while todo:
   x,p,w=todo.pop()
   if x<self.n:ans[x]=w
   else:
    nb=[y for y in self.adj[x] if y!=p];assert len(nb)==2
    todo.extend((y,x,w/2) for y in nb)
  return ans

def dense_profile(t,u,cap,backend=0,outside=False):
 l,w=t.profile(u,cap,backend,outside);v=np.zeros(t.n);np.add.at(v,l,w);return v,l,w

def condense(D,profiles):
 off=[0];ls=[];ws=[]
 for l,w in profiles:ls.extend(l);ws.extend(w);off.append(len(ls))
 off=np.array(off,dtype=np.int32);ls=np.array(ls,dtype=np.int32);ws=np.array(ws,dtype=np.float32);out=np.zeros((len(profiles),len(profiles)))
 lib.condense(len(profiles),len(D),off.ctypes.data_as(ip),ls.ctypes.data_as(ip),ws.ctypes.data_as(fp),D.ctypes.data_as(dp),out.ctypes.data_as(dp));return out
res={};rng=random.Random(2309);count=0;outside_count=0;worst=0.;outside_worst=0.
for n in (4,5,9,17,32,64):
 for rep in range(4):
  pool=list(range(n))
  while len(pool)>1:
   x=pool.pop(rng.randrange(len(pool)));y=pool.pop(rng.randrange(len(pool)));pool.append((x,y))
  t=Tree(pool[0])
  for u in range(2*n-1):
   exact=np.zeros(n)
   for j,w in weights(t.sub[u]).items():exact[j]=w
   for backend,cap in [(0,0),(0,64),(1,128)]:
    p,_,_=dense_profile(t,u,cap,backend);worst=max(worst,float(np.max(abs(p-exact))));count+=1
   if u!=t.root:
    exact=np.zeros(n)
    for j,w in t.exact_out(u).items():exact[j]=w
    p,_,_=dense_profile(t,u,0,outside=True);outside_worst=max(outside_worst,float(np.max(abs(p-exact))));outside_count+=1
res['exact_profiles']={'checks':count,'max_error':worst,'outside_checks':outside_count,'outside_max_error':outside_worst};assert worst<1e-14 and outside_worst<1e-14
# Previous small counterexamples, now using current profile helpers.
res['small_clades']=[]
for n in (4,5,9):
 t=Tree(ladder(list(range(n))))
 for backend,cap in ((0,64),(1,128)):
  p,l,w=dense_profile(t,t.root,cap,backend);exact=np.array([weights(t.t)[i] for i in range(n)])
  res['small_clades'].append({'n':n,'backend':backend,'entries':len(l),'max_error':float(np.max(abs(p-exact)))})
res['original_examples']=[]
for n in (14,17):
 A=ladder(list(range(9)));G=(11,(12,13)) if n==14 else ladder(list(range(11,17)));before=((A,9),(10,G));after=((A,10),(9,G));t=Tree(before);u=next(k for k,v in t.sub.items() if v==A)
 D=np.ones((n,n));np.fill_diagonal(D,0)
 if n==14:
  D[0,9]=D[9,0]=2
  for j in (11,12,13):D[9,j]=D[j,9]=17/16
 else:
  flags=np.array([i in {0,10,13} for i in range(n)]);D=2.+(flags[:,None]!=flags[None,:]);np.fill_diagonal(D,0)
 g=(3,(4,5)) if n==14 else ladder(list(range(3,9)));tb=((0,1),(2,g));ta=((0,2),(1,g));truth=score(after,D)-score(before,D)
 for backend,cap in ((0,64),(1,128)):
  prof=[t.profile(u,cap,backend)]+[(np.array([j],np.int32),np.ones(1,np.float32)) for j in range(9,n)];Dc=condense(D,prof);change=score(ta,Dc)-score(tb,Dc)
  res['original_examples'].append({'n':n,'backend':backend,'true_change':truth,'current_change':change,'error':abs(truth-change)});assert abs(truth-change)<1e-12
# Large balanced profile. Distances are a tree metric: unit pendant edges plus one length-1 split.
# Independent O(N) Pauplin evaluation for this metric, verified against graph scoring on small trees.
def split_score(tree,marked):
 def go(t,root=False):
  if isinstance(t,int):return float(t in marked),0.
  (p,a),(q,b)=(go(c) for c in t);cross=p*(1-q)+(1-p)*q
  return (p+q)/2,a+b+(1. if root else .5)*cross
 return len(weights(tree))+go(tree,True)[1]
for tr in [before,after]:assert split_score(tr,{0,10,13})==score(tr,D)
A=balanced(list(range(4096)));B=4096;C=4097;G=balanced(list(range(4098,4102)));tb=((A,B),(C,G));ta=((A,C),(B,G));t=Tree(tb);u=next(k for k,v in t.sub.items() if v==A);res['capped_examples']=[]
for backend,cap in ((0,64),(1,128)):
 p,l,w=dense_profile(t,u,cap,backend);marked=set(map(int,l))|{C,4098,4099};flags=np.array([j in marked for j in range(t.n)]);D=np.ascontiguousarray(2.+(flags[:,None]!=flags[None,:]));np.fill_diagonal(D,0)
 prof=[(l,w)]+[(np.array([j],np.int32),np.ones(1,np.float32)) for j in range(4096,4102)];Dc=condense(D,prof)
 sb=((0,1),(2,((3,4),(5,6))));sa=((0,2),(1,((3,4),(5,6))));actual=split_score(ta,marked)-split_score(tb,marked);predicted=score(sa,Dc)-score(sb,Dc)
 exact=np.zeros(t.n);exact[:4096]=1/4096
 res['capped_examples'].append({'N':t.n,'clade_leaves':4096,'backend':backend,'cap':cap,'entries':len(l),'weight_sum':float(w.sum()),'total_variation':float(np.abs(p-exact).sum()/2),'true_change':actual,'current_change':predicted,'error':abs(actual-predicted)})
 assert actual>0 and predicted<0
 del D
(R/'profile_checks.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
