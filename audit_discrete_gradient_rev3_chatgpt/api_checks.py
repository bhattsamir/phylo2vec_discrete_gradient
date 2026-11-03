"""Independent small-tree checks. Imports only the captured source snapshot."""
import sys, json, itertools, ctypes, concurrent.futures, threading
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot'))
from discrete_gradient import DiscreteGradient, v_to_ancestry_matrix

def adjacency(v):
    n=len(v)+1
    # Decode topology, then independently score via graph distances.
    M=v_to_ancestry_matrix(np.asarray(v,dtype=np.int64))
    adj=[set() for _ in range(2*n-1)]
    for k,(a,b) in enumerate(M):
        p=n+k
        for c in (int(a),int(b)):
            adj[p].add(c); adj[c].add(p)
    a,b=adj[-1]
    adj[a].remove(2*n-2); adj[b].remove(2*n-2)
    adj[a].add(b); adj[b].add(a); adj[-1].clear()
    return adj

def score(v,D):
    n=len(v)+1; adj=adjacency(v); out=0.
    for a in range(n):
        stack=[(a,-1,0)]
        while stack:
            u,p,d=stack.pop()
            if a<u<n: out+=D[a,u]*2.**(1-d)
            stack.extend((b,u,d+1) for b in adj[u] if b!=p)
    return out

rng=np.random.default_rng(2309)
dg=DiscreteGradient('c'); nb=DiscreteGradient('numba')
results={}
for ordered in (True,False):
    worst={backend:{'error':0.,'cases':0} for backend in ('c','numba')}
    for n in (4,5,6):
        A=rng.uniform(.1,3,(n,n)); D=(A+A.T)/2; np.fill_diagonal(D,0)
        for vv in itertools.product(*(range((i if ordered else 2*i)+1) for i in range(n-1))):
            v=np.array(vv,dtype=np.int64); L=score(v,D)
            exact=np.zeros((n-1,n-2))
            for i in range(1,n-1):
                for j in range(i):
                    w=v.copy();w[i]=j;exact[i,j]=score(w,D)-L
            for backend,obj in [('c',dg),('numba',nb)]:
                pred=obj.calculate_discrete_gradient(v,D)
                err=float(np.max(np.abs(pred-exact))); worst[backend]['cases']+=1
                if err>worst[backend]['error']:
                    i,j=np.unravel_index(np.argmax(np.abs(pred-exact)),pred.shape)
                    worst[backend].update(error=err,n=n,v=v.tolist(),D=D.tolist(),i=int(i),j=int(j),predicted=float(pred[i,j]),expected=float(exact[i,j]))
    results['ordered_lower' if ordered else 'general_lower']=worst
    print(json.dumps({('ordered_lower' if ordered else 'general_lower'):worst}),flush=True)
# Small normalization example.
v=np.array([0,0,0],dtype=np.int64);D=np.ones((4,4));np.fill_diagonal(D,0);D[0,2]=D[2,0]=10
pred=dg.calculate_discrete_gradient_full(v,D);w=v.copy();w[2]=1
results['normalization']={'v':v.tolist(),'w':w.tolist(),'L_before':score(v,D),'L_after':score(w,D),'returned_delta':float(pred[2,1])}
# Nonintegral vector is silently converted before validation.
results['fractional_vector_accepted']=DiscreteGradient._validate_v(np.array([0,0.9,1.9])).tolist()
# Same topology, different distances concurrently, so races cannot be blamed on input mutation.
n=64;v=np.array([rng.integers(i+1) for i in range(n-1)],dtype=np.int64)
Ds=[]
for k in range(2):
    A=rng.uniform(1,10,(n,n));D=(A+A.T)/2;np.fill_diagonal(D,0);Ds.append(D)
refs=[dg.calculate_discrete_gradient_full(v,D).copy() for D in Ds]
bar=threading.Barrier(2)
def worker(k):
    errors=[]
    for t in range(100):
        bar.wait();p=dg.calculate_discrete_gradient_full(v,Ds[k]);errors.append(float(np.max(np.abs(p-refs[k]))))
    return {'bad_calls':sum(x>1e-10 for x in errors),'max_error':max(errors)}
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    results['concurrent_calls']=list(pool.map(worker,[0,1]))
(ROOT/'api_checks.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:v for k,v in results.items() if k not in ('ordered_lower','general_lower')}),flush=True)
