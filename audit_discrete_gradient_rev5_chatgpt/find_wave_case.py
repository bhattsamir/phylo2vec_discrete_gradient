from pathlib import Path
import numpy as np,json
R=Path(__file__).resolve().parent
s=(R/'representative_check.py').read_text();ns={'np':np};exec(s[s.index('def weights'):s.index("res={'full_BME")],ns);score=ns['score']
def ladder(x):
 t=x[0]
 for a in x[1:]:t=(t,a)
 return t
A=[((0,1),2),((0,2),1),((1,2),0)];B=[((3,4),5),((3,5),4),((4,5),3)]
trees=[ladder([(a,b),6,7,8,9]) for a in A for b in B];pairs=[(i,j) for i in range(10) for j in range(i+1,10)]
rows=[]
for t in trees:
 row=[]
 for i,j in pairs:
  D=np.zeros((10,10));D[i,j]=D[j,i]=1;row.append(score(t,D))
 rows.append(row)
W=np.array(rows);rng=np.random.default_rng(9019)
# Binary sequence patterns, including four outside leaves fixed to the first symbol.
patterns=np.zeros((32,10),np.int64)
for i in range(32):patterns[i,1:6]=[(i>>k)&1 for k in range(5)]
H=np.array([patterns[:,i]!=patterns[:,j] for i,j in pairs],float).T
for rep in range(100000):
 probs=rng.dirichlet(np.full(32,.2));counts=rng.multinomial(1596,probs)
 p=(counts@H)/1596.;dist=-.75*np.log(1-(4/3)*np.minimum(.74,p));scores=W@dist
 ai=int(np.argmin(scores[[0,3,6]]));bi=int(np.argmin(scores[[0,1,2]]))
 if ai and bi and scores[3*ai]<scores[0]-1e-4 and scores[bi]<scores[0]-1e-4 and scores[3*ai+bi]>scores[0]+1e-3:
  D=np.zeros((10,10))
  for (i,j),d in zip(pairs,dist):D[i,j]=D[j,i]=d
  out={'seed':9019,'trial':rep,'counts':counts.tolist(),'patterns':patterns.tolist(),'A_choice':ai,'B_choice':bi,'length':1596,'L_before':float(scores[0]),'delta_A':float(scores[3*ai]-scores[0]),'delta_B':float(scores[bi]-scores[0]),'delta_both':float(scores[3*ai+bi]-scores[0]),'all_nine_scores':scores.tolist(),'D':D.tolist()}
  (R/'wave_sequence_case.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['D','patterns','counts']}));break
else:raise RuntimeError('not found')
