from pathlib import Path
import runpy,json
import numpy as np
ns=runpy.run_path(str(Path(__file__).with_name('representative_check.py')))
score,reps,weights=ns['score'],ns['reps'],ns['weights']
A=(0,1)
for x in range(2,9):A=(A,x)
G=(11,12)
for x in range(13,17):G=(G,x)
before=((A,9),(10,G));after=((A,10),(9,G))
# Exact tree metric: unit pendant edges plus one length-1 split {0,10,13}|others.
# Unspecified internal edges have length zero. Small positive resolutions preserve signs.
S={0,10,13};D=np.zeros((17,17))
for i in range(17):
 for j in range(i+1,17):D[i,j]=D[j,i]=2.+float((i in S)!=(j in S))
# Only A is condensed. All other leaves retain their original distance entries.
leaves=list(range(9,17));g=(3,4)
for x in range(5,9):g=(g,x)
t0=((0,1),(2,g));t1=((0,2),(1,g))
res={'metric':'unit pendant edges, length-one split {0,10,13}|complement, other internal edges zero','full_BME_before':score(before,D),'full_BME_after':score(after,D),'pA_true':weights(A)[0],'pG_true':weights(G)[13]}
res['true_change']=res['full_BME_after']-res['full_BME_before']
for k in [4,8]:
 profile=reps(A,k);De=np.zeros((9,9));De[1:,1:]=D[np.ix_(leaves,leaves)]
 for j,leaf in enumerate(leaves,1):De[0,j]=De[j,0]=np.mean(D[profile,leaf])
 res[str(k)]={'representatives':profile,'surrogate_before':score(t0,De),'surrogate_after':score(t1,De),'surrogate_change':score(t1,De)-score(t0,De)}
 assert res[str(k)]['surrogate_change']<0
assert res['true_change']>0
Path(__file__).with_suffix('.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
