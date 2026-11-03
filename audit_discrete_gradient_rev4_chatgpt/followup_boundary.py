from pathlib import Path
import subprocess,ctypes,json,sys
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'snapshot'))
from discrete_gradient import v_to_ancestry_matrix
s=(R/'api_checks.py').read_text();exec(s[s.index('def adjacency'):s.index('rng=np.random')])
s=(R/'snapshot/phylo2vec_fastme.c').read_text();prefix,body=s.split('double optimize_phylo2vec_bme_c(',1)
assert body.count('for (int32_t iter = 0; iter < 500; ++iter)')==1
body=body.replace('for (int32_t iter = 0; iter < 500; ++iter)','for (int32_t iter = 0; iter < review_iteration_cap; ++iter)',1).replace('iter + 1 < 500','iter + 1 < review_iteration_cap')
(R/'core_variable_cap.c').write_text(prefix+'int32_t review_iteration_cap = 1;\ndouble optimize_phylo2vec_bme_c('+body)
subprocess.run(['gcc','-O2','-pthread','-shared','-fPIC',str(R/'core_variable_cap.c'),'-lm','-o',str(R/'core_variable_cap.so')],check=True)
lib=ctypes.CDLL(str(R/'core_variable_cap.so'));f=lib.optimize_phylo2vec_bme_c;ip=ctypes.POINTER(ctypes.c_int64);dp=ctypes.POINTER(ctypes.c_double)
f.argtypes=[ip,dp,ctypes.c_int32,ip,ctypes.POINTER(ctypes.c_int32),dp];f.restype=ctypes.c_double;cap=ctypes.c_int32.in_dll(lib,'review_iteration_cap')
rng=np.random.default_rng(2309);results=[]
for limit in [1,2,3,5,10,20]:
 cap.value=limit
 for t in range(20):
  n=12;v=np.zeros(n-1,dtype=np.int64) if t==0 else np.array([rng.integers(2*i+1) for i in range(n-1)],dtype=np.int64)
  A=rng.uniform(.1,3,(n,n));D=(A+A.T)/2;np.fill_diagonal(D,0);out=np.empty_like(v);ne=ctypes.c_int32();ms=ctypes.c_double()
  actual=f(v.ctypes.data_as(ip),D.ctypes.data_as(dp),n,out.ctypes.data_as(ip),ctypes.byref(ne),ctypes.byref(ms));truth=score(out,D)
  results.append({'cap':limit,'case':t,'reported':actual,'truth':truth,'error':abs(actual-truth),'gradient_evals':ne.value})
res={'original_case':results[0],'cases':len(results),'max_error':max(x['error'] for x in results),'bad':sum(int(x['error']>1e-10) for x in results),'all':results}
(R/'followup_boundary.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='all'}));assert res['bad']==0
