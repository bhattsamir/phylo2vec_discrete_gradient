from pathlib import Path
import sys,json,itertools,threading,concurrent.futures
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'snapshot'))
from discrete_gradient import DiscreteGradient,v_to_ancestry_matrix,calculate_discrete_gradient_numba
s=(R/'api_checks.py').read_text();exec(s[s.index('def adjacency'):s.index('rng=np.random')])
rng=np.random.default_rng(2309);objs={'c':DiscreteGradient('c'),'numba':DiscreteGradient('numba')};res={}
for ordered in (True,False):
 out={k:{'accepted':0,'rejected':0,'wrong_domain_behavior':0,'max_error':0.} for k in objs}
 for n in (4,5,6):
  A=rng.uniform(.1,3,(n,n));D=(A+A.T)/2;np.fill_diagonal(D,0)
  for vv in itertools.product(*(range((i if ordered else 2*i)+1) for i in range(n-1))):
   v=np.array(vv,dtype=np.int64);is_ordered=all(v[i]<=i for i in range(n-1));L=score(v,D);exact=np.zeros((n-1,n-2))
   if is_ordered:
    for i in range(1,n-1):
     for j in range(i):
      w=v.copy();w[i]=j;exact[i,j]=score(w,D)-L
   for k,obj in objs.items():
    try:p=obj.calculate_discrete_gradient(v,D)
    except ValueError:
     out[k]['rejected']+=1;out[k]['wrong_domain_behavior']+=int(is_ordered);continue
    out[k]['accepted']+=1;out[k]['wrong_domain_behavior']+=int(not is_ordered)
    if is_ordered:out[k]['max_error']=max(out[k]['max_error'],float(np.max(np.abs(p-exact))))
 res['ordered' if ordered else 'general_domain']=out
v=np.array([0,0,3]);D=np.ones((4,4));np.fill_diagonal(D,0);D[0,2]=D[2,0]=10
res['minimal_rejections']={}
for k,f in [('c',objs['c'].calculate_discrete_gradient),('numba',objs['numba'].calculate_discrete_gradient),('direct_numba',calculate_discrete_gradient_numba)]:
 try:f(v,D);res['minimal_rejections'][k]=False
 except ValueError:res['minimal_rejections'][k]=True
v=np.array([0,0,0]);w=v.copy();w[2]=1
res['normalization']={'before':score(v,D),'after':score(w,D),'delta':float(objs['c'].calculate_discrete_gradient_full(v,D)[2,1])}
res['fractional_rejected']=False
try:DiscreteGradient._validate_v(np.array([0,.9,1.9]))
except ValueError:res['fractional_rejected']=True
n=64;v=np.array([rng.integers(i+1) for i in range(n-1)],dtype=np.int64);Ds=[]
for k in range(2):
 A=rng.uniform(1,10,(n,n));D=(A+A.T)/2;np.fill_diagonal(D,0);Ds.append(D)
dg=objs['c'];refs=[dg.calculate_discrete_gradient_full(v,D).copy() for D in Ds];bar=threading.Barrier(2)
def work(k):
 errors=[]
 for t in range(100):
  bar.wait();p=dg.calculate_discrete_gradient_full(v,Ds[k]);errors.append(float(np.max(np.abs(p-refs[k]))))
 return {'bad_calls':sum(x>1e-10 for x in errors),'max_error':max(errors)}
with concurrent.futures.ThreadPoolExecutor(2) as pool:res['concurrent_calls']=list(pool.map(work,[0,1]))
(R/'followup_api.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
assert all(x['wrong_domain_behavior']==0 and x['max_error']<1e-12 for group in ('ordered','general_domain') for x in res[group].values())
assert all(res['minimal_rejections'].values()) and res['fractional_rejected']
assert all(x['bad_calls']==0 for x in res['concurrent_calls'])
