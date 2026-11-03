from pathlib import Path
import re,subprocess
R=Path(__file__).resolve().parent
sources={k:(R/'snapshot'/f).read_text() for k,f in [('rna','bench_rnasim_gpu.cu'),('rec','bench_minibatch_recovery.cu')]}
def extract(s,name):
 m=re.search(r'^static[^\n]*\b'+name+r'\(',s,re.M);assert m,name
 b=s.index('{',m.start());j=b+1;level=1
 while level:level+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[m.start():j]
header='#include <cstdint>\n#include <vector>\n#include <queue>\n#include <algorithm>\n#include <cstring>\n#include <cmath>\n'
for ns,s in sources.items():
 funcs=['compute_tree_summaries_blind','compute_clade_dyadic_profile']+(['compute_outgroup_dyadic_profile','build_condensed_D_from_profiles_host'] if ns=='rna' else [])
 header+='namespace '+ns+' {\n'+'\n'.join(extract(s,f) for f in funcs)+'\n}\n'
header+='''
extern "C" int profile_clade(int backend,int N,const int32_t*a,const int32_t*b,int u,int cap,int32_t*l,float*w){std::vector<int32_t>ord,par,cnt,rep,leaves;std::vector<float>weights;if(backend==0){rna::compute_tree_summaries_blind(a,b,N,ord,par,cnt,rep);rna::compute_clade_dyadic_profile(u,N,a,b,rep.data(),cap,leaves,weights);}else{rec::compute_tree_summaries_blind(a,b,N,ord,par,cnt,rep);rec::compute_clade_dyadic_profile(u,N,a,b,rep.data(),cap,leaves,weights);}std::copy(leaves.begin(),leaves.end(),l);std::copy(weights.begin(),weights.end(),w);return leaves.size();}
extern "C" int profile_out(int N,const int32_t*a,const int32_t*b,int u,int cap,int32_t*l,float*w){std::vector<int32_t>ord,par,cnt,rep,leaves;std::vector<float>weights;rna::compute_tree_summaries_blind(a,b,N,ord,par,cnt,rep);rna::compute_outgroup_dyadic_profile(u,2*N-2,N,a,b,par.data(),rep.data(),cap,leaves,weights);std::copy(leaves.begin(),leaves.end(),l);std::copy(weights.begin(),weights.end(),w);return leaves.size();}
extern "C" void condense(int K,int N,const int32_t*off,const int32_t*l,const float*w,const double*D,double*out){rna::build_condensed_D_from_profiles_host(K,N,off,l,w,D,out);}
'''
(R/'profile_helpers.cpp').write_text(header)
subprocess.run(['g++','-O2','-shared','-fPIC','-std=c++17',str(R/'profile_helpers.cpp'),'-o',str(R/'profile_helpers.so')],check=True)
