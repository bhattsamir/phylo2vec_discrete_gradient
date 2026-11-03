import json
from pathlib import Path
import numpy as np
A=(0,1)
for x in range(2,9):A=(A,x)
G=(11,(12,13))
before=((A,9),(10,G));after=((A,10),(9,G))
D=np.ones((14,14));np.fill_diagonal(D,0);D[0,9]=D[9,0]=2
for x in (11,12,13):D[9,x]=D[x,9]=17/16

def weights(T):
 if isinstance(T,int):return {T:1.}
 out={}
 for child in T:
  for leaf,w in weights(child).items():out[leaf]=w/2
 return out

def reps(T,k):
 if isinstance(T,int):return [T]*k
 l,r=(reps(ch,k) for ch in T)
 return l[::2]+r[::2]

def score(tree,D):
 n=len(D);adj={i:[] for i in range(n)};next_id=n
 def grow(t):
  nonlocal next_id
  if isinstance(t,int):return t
  l,r=(grow(ch) for ch in t);u=next_id;next_id+=1;adj[u]=[l,r];adj[l].append(u);adj[r].append(u);return u
 root=grow(tree);l,r=adj.pop(root);adj[l].remove(root);adj[r].remove(root);adj[l].append(r);adj[r].append(l)
 total=0
 for x in range(n):
  stack=[(x,-1,0)]
  while stack:
   u,p,d=stack.pop()
   if x<u<n:total+=D[x,u]*2.**(1-d)
   stack.extend((y,u,d+1) for y in adj[u] if y!=p)
 return total
res={'full_BME_before':score(before,D),'full_BME_after':score(after,D),'weights':weights(A)}
res['true_change']=res['full_BME_after']-res['full_BME_before']
for k in [4,8]:
 profile=reps(A,k);De=np.ones((6,6));np.fill_diagonal(De,0)
 for j,leaf in enumerate([9,10,11,12,13],1):De[0,j]=De[j,0]=np.mean(D[profile,leaf])
 De[1,3:]=De[3:,1]=17/16
 t0=((0,1),(2,(3,(4,5))));t1=((0,2),(1,(3,(4,5))))
 res[str(k)]={'representatives':profile,'surrogate_before':score(t0,De),'surrogate_after':score(t1,De),'surrogate_change':score(t1,De)-score(t0,De)}
assert all(D[i,j]<=D[i,z]+D[z,j] for i in range(14) for j in range(14) for z in range(14))
assert res['true_change']>0 and res['8']['surrogate_change']<0
Path('representative_check.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
