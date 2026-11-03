"""Source scheduler + exact outside profile counterexample.
Uses prescribed valid subtree proposals, not an end-to-end CUDA optimizer run.
"""
from pathlib import Path
import ctypes,json,numpy as np
R=Path(__file__).resolve().parent;s=(R/'profile_checks.py').read_text();exec(s[:s.index('res={};rng=')])
libw=ctypes.CDLL(str(R/'wave_helpers.so'));libw.schedule.argtypes=[ctypes.c_int,ip,ip,ip,ip,ip];libw.schedule.restype=ctypes.c_int
up=ctypes.POINTER(ctypes.c_uint64);libw.jc.argtypes=[ctypes.c_int,up,fp]
x=json.loads((R/'wave_sequence_case.json').read_text());seq=np.repeat(np.array(x['patterns'],np.uint8),x['counts'],axis=0).T
packed=np.zeros((10,125),np.uint64)
for i in range(10):
 for j in range(1596):
  packed[i,int(seq[i,j])*25+j//64]|=np.uint64(1)<<np.uint64(j%64);packed[i,100+j//64]|=np.uint64(1)<<np.uint64(j%64)
D=np.zeros((10,10),np.float32);libw.jc(10,packed.ctypes.data_as(up),D.ctypes.data_as(fp));np.save(R/'wave_packed_sequences.npy',packed)
small=[ladder([(((0,1),2),((3,4),5)),6,7,8,9]),ladder([(((0,2),1),((3,4),5)),6,7,8,9]),ladder([(((0,1),2),((3,5),4)),6,7,8,9]),ladder([(((0,2),1),((3,5),4)),6,7,8,9])]
# Expand six labels to 256 identical copies. BME equals the 10-label score:
# within each zero-distance group contributes zero, its normalized profile has mass one.
groups=[balanced(list(range(i*256,(i+1)*256))) for i in range(6)]+list(range(1536,1540))
def expand(x):return groups[x] if isinstance(x,int) else tuple(expand(c) for c in x)
full=Tree(expand(small[0]));nodes=np.zeros(full.n,np.int32);levels=nodes.copy();counts=nodes.copy();nw=libw.schedule(full.n,full.a.ctypes.data_as(ip),full.b.ctypes.data_as(ip),nodes.ctypes.data_as(ip),levels.ctypes.data_as(ip),counts.ctypes.data_as(ip));waves=[{'node':int(nodes[i]),'level':int(levels[i]),'leaves':int(counts[i])} for i in range(nw)]
A=expand(((0,1),2));B=expand(((3,4),5));uA=next(u for u,v in full.sub.items() if v==A);uB=next(u for u,v in full.sub.items() if v==B)
assert int(levels[np.where(nodes[:nw]==uA)[0][0]])==int(levels[np.where(nodes[:nw]==uB)[0][0]])==0
mapgroup=np.r_[np.repeat(np.arange(6),256),np.arange(6,10)]
# Exact profiles obtained using the actual newly added DFS outside builder.
profiles=[];deltas=[];Ds=[]
for u,ids in [(uA,[0,1,2]),(uB,[3,4,5])]:
 l,w=full.profile(u,-1,outside=True);p=np.bincount(mapgroup[l],weights=w,minlength=10);profiles.append(p.tolist())
 Dc=np.zeros((4,4),float)
 for i in range(3):
  for j in range(3):Dc[i,j]=D[ids[i],ids[j]]
  Dc[i,3]=Dc[3,i]=np.dot(D[ids[i]],p)
 change=score(((0,2),(1,3)),Dc)-score(((0,1),(2,3)),Dc);deltas.append(float(change));Ds.append(Dc.tolist())
L=[float(score(t,D.astype(float))) for t in small]
assert max(abs(deltas[i]-(L[i+1]-L[0])) for i in range(2))<1e-12
assert max(deltas)<-1e-6 and L[3]>L[0]+.05
# Cross-window interaction from changed attachment profiles.
p0=np.array([.25,.25,.5]);p1=np.array([.25,.5,.25]);interaction=float((p1-p0)@D[:3,3:6].astype(float)@(p1-p0)*.5)
assert abs(interaction-(L[3]-L[1]-L[2]+L[0]))<1e-12
out={'N':full.n,'group_copies':256,'production_schedule':waves,'A_root':uA,'B_root':uB,'sequence_length':1596,'scores':L,'individual_deltas':deltas,'sum_reported_delta':sum(deltas),'joint_delta':L[3]-L[0],'omitted_interaction':interaction,'full_D':D.tolist(),'outside_group_profiles':profiles,'condensed_quartets':Ds,'scope':'Actual source scheduler, outside profiles and distance function evaluated on CPU. Prescribed valid updates, not an end-to-end optimizer trajectory.'}
(R/'wave_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['full_D','outside_group_profiles','condensed_quartets']}))
