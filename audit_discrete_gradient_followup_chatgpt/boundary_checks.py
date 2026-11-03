from pathlib import Path
import subprocess,ctypes,json,sys
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'snapshot'))
from discrete_gradient import v_to_ancestry_matrix
# Reuse only this audit's graph-distance scoring helpers, not its test runner.
scope=globals();src=(R/'api_checks.py').read_text();exec(src[src.index('def adjacency'):src.index('rng=np.random')],scope)
s=(R/'snapshot/phylo2vec_fastme.c').read_text();prefix,body=s.split('double optimize_phylo2vec_bme_c(',1)
assert body.count('for (int32_t iter = 0; iter < 500; ++iter)')==1
body=body.replace('for (int32_t iter = 0; iter < 500; ++iter)','for (int32_t iter = 0; iter < 1; ++iter)',1)
(R/'core_iteration_cap_1.c').write_text(prefix+'double optimize_phylo2vec_bme_c('+body)
subprocess.run(['gcc','-O2','-shared','-fPIC',str(R/'core_iteration_cap_1.c'),'-lm','-o',str(R/'core_iteration_cap_1.so')],check=True)
lib=ctypes.CDLL(str(R/'core_iteration_cap_1.so'));f=lib.optimize_phylo2vec_bme_c
ip=ctypes.POINTER(ctypes.c_int64);dp=ctypes.POINTER(ctypes.c_double)
f.argtypes=[ip,dp,ctypes.c_int32,ip,ctypes.POINTER(ctypes.c_int32),dp];f.restype=ctypes.c_double
rng=np.random.default_rng(2309);n=12;v=np.zeros(n-1,dtype=np.int64);A=rng.uniform(.1,3,(n,n));D=(A+A.T)/2;np.fill_diagonal(D,0)
out=np.empty_like(v);ne=ctypes.c_int32();ms=ctypes.c_double()
reported=f(v.ctypes.data_as(ip),D.ctypes.data_as(dp),n,out.ctypes.data_as(ip),ctypes.byref(ne),ctypes.byref(ms))
r={'diagnostic_loop_cap':1,'production_loop_cap':500,'n':n,'initial_vector':v.tolist(),'returned_vector':out.tolist(),'initial_BME':score(v,D),'reported_BME':reported,'returned_tree_BME':score(out,D),'gradient_evals':ne.value,'D':D.tolist()}
assert abs(r['returned_tree_BME']-reported)>1e-6
(R/'boundary_checks.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:x for k,x in r.items() if k!='D'}))
