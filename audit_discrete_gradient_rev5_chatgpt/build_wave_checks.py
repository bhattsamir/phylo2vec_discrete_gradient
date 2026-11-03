from pathlib import Path
import re,subprocess
R=Path(__file__).resolve().parent
s=(R/'snapshot/bench_rnasim_gpu.cu').read_text()
def extract(name):
 m=re.search(r'^(?:static|__device__)[^\n]*\b'+name+r'\(',s,re.M);assert m,name
 b=s.index('{',m.start());j=b+1;level=1
 while level:level+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[m.start():j]
src='''#include <cstdint>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#define __device__
#define __forceinline__ inline
#define __popcll __builtin_popcountll
'''
for name in ['compute_tree_summaries_blind','build_component_waves','dev_jc69_dist_popc']:src+=extract(name)+'\n'
src+='''
extern "C" int schedule(int N,const int32_t*a,const int32_t*b,int32_t*nodes,int32_t*levels,int32_t*counts){std::vector<int32_t>A(a,a+2*N-1),B(b,b+2*N-1),ord,par,cnt,rep;compute_tree_summaries_blind(a,b,N,ord,par,cnt,rep);std::vector<uint8_t>roots;std::vector<std::vector<int32_t>>waves;build_component_waves(N,ord,A,B,1050,1050,roots,waves);int i=0;for(size_t k=0;k<waves.size();k++)for(int u:waves[k]){nodes[i]=u;levels[i]=k;counts[i]=cnt[u];i++;}return i;}
extern "C" void jc(int N,const uint64_t*seq,float*D){for(int i=0;i<N;i++)for(int j=0;j<N;j++)D[i*N+j]=dev_jc69_dist_popc(i,j,seq,25);}
'''
(R/'wave_helpers.cpp').write_text(src)
subprocess.run(['g++','-O2','-shared','-fPIC','-std=c++17','wave_helpers.cpp','-o','wave_helpers.so'],cwd=R,check=True)
