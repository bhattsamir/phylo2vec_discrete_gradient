from pathlib import Path
import subprocess,json,collections
R=Path(__file__).resolve().parent;s=(R/'snapshot/bench_minibatch_recovery.cu').read_text();start=s.index('static void compute_tree_summaries_blind(');body=s.index('{',start);depth=1;i=body+1
while depth:
 depth+=(s[i]=='{')-(s[i]=='}');i+=1
fn=s[start:i].replace('compute_tree_summaries_blind','compute_tree_summaries_rep4')
(R/'small_clade_check.cpp').write_text('#include "host_helpers.hpp"\n'+fn+'''
int main(){for(int N:{4,5}){std::vector<int32_t>a(2*N-1),b(2*N-1),ord,par,cnt,r4,r8,r0;a[N]=0;b[N]=1;for(int k=1;k<N-1;k++){a[N+k]=N+k-1;b[N+k]=k+1;}compute_tree_summaries_blind(a.data(),b.data(),N,ord,par,cnt,r8);compute_tree_summaries_rep4(a.data(),b.data(),N,ord,par,cnt,r0,r4);int K=N==4?4:8;auto &v=N==4?r4:r8;printf("N=%d K=%d reps",N,K);for(int k=0;k<K;k++)printf(" %d",v[K*(2*N-2)+k]);printf("\\n");}return 0;}
''')
subprocess.run(['g++','-O2','-std=c++17',str(R/'small_clade_check.cpp'),'-o',str(R/'small_clade_check')],check=True)
out=subprocess.check_output([str(R/'small_clade_check')],text=True);print(out,end='');res=[]
for line in out.splitlines():
 fields=line.split();n=int(fields[0].split('=')[1]);k=int(fields[1].split('=')[1]);reps=list(map(int,fields[3:]));counts=collections.Counter(reps)
 truth=[2.**(-(n-1)),2.**(-(n-1))]+[2.**(-(n-j)) for j in range(2,n)]
 approx=[counts[j]/k for j in range(n)];res.append({'n':n,'representatives':k,'truth_weights':truth,'representative_weights':approx,'max_error':max(abs(x-y) for x,y in zip(truth,approx))})
(R/'small_clade_check.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res));assert all(x['max_error']>0 for x in res)
