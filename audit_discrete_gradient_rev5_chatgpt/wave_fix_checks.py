from pathlib import Path
import numpy as np,ctypes,json
R=Path(__file__).resolve().parent
exec((R/'wave_checks.py').read_text())
g=ctypes.CDLL(str(R/'wave_driver.so'));g.controlled_wave.argtypes=[ctypes.c_int,ip,ip,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,dp,dp];g.trace.argtypes=[ip,dp];g.trace.restype=ctypes.c_int
common=next(u for u in full.sub if full.a[u] in (uA,uB) and full.b[u] in (uA,uB))
mg=mapgroup.astype(np.int32);dg=D.astype(float);results=[]
for reverse in [False,True]:
 a=full.a.copy();b=full.b.copy()
 if reverse:a[common],b[common]=b[common],a[common]
 for ngpus in [1,2,8]:
  result=np.zeros(4);g.controlled_wave(full.n,a.ctypes.data_as(ip),b.ctypes.data_as(ip),uA,uB,ngpus,mg.ctypes.data_as(ip),dg.ctypes.data_as(dp),result.ctypes.data_as(dp))
  targets=np.zeros(100,np.int32);deltas=np.zeros(100);n=g.trace(targets.ctypes.data_as(ip),deltas.ctypes.data_as(dp))
  attempts=[{'target':'A' if targets[i]==0 else 'B','delta':float(deltas[i])} for i in range(n)]
  out={'reverse_window_order':reverse,'configured_gpus':ngpus,'initial_BME':result[0],'reported_delta':result[1],'final_BME':result[2],'actual_delta':result[2]-result[0],'applied_proposals':int(result[3]),'attempts':attempts}
  assert result[3]==1 and result[1]<0 and result[2]<result[0]
  assert abs(result[1]-(result[2]-result[0]))<1e-12
  assert attempts[0]['delta']<0 and attempts[1]['delta']>0
  results.append(out)
(R/'wave_fix_checks.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
