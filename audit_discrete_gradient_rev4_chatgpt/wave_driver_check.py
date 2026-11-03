from pathlib import Path
import numpy as np,ctypes,json
R=Path(__file__).resolve().parent
# Executes and revalidates construction of the prescribed sequence example.
exec((R/'wave_checks.py').read_text())
g=ctypes.CDLL(str(R/'wave_driver.so'));g.controlled_wave.argtypes=[ctypes.c_int,ip,ip,ctypes.c_int,ctypes.c_int,ip,dp,dp]
result=np.zeros(4);dg=D.astype(float);mg=mapgroup.astype(np.int32)
g.controlled_wave(full.n,full.a.ctypes.data_as(ip),full.b.ctypes.data_as(ip),uA,uB,mg.ctypes.data_as(ip),dg.ctypes.data_as(dp),result.ctypes.data_as(dp))
res={'N':full.n,'threads':1,'initial_full_graph_BME':result[0],'controller_reported_delta':result[1],'final_full_graph_BME':result[2],'actual_delta':result[2]-result[0],'applied_proposals':int(result[3]),'scope':'Unmodified source wave controller with controlled exact-local-acceptance proposal callback replacing GPU optimizer. Independent full expanded-tree graph score.'}
assert result[3]==2 and result[1]<0 and result[2]-result[0]>.05
assert abs(result[0]-L[0])<1e-10 and abs(result[2]-L[3])<1e-10
(R/'wave_driver_check.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
