"""Actual wave controller with a controlled proposal callback replacing GPU search.
The callback evaluates a prescribed valid group swap against the outside profile
supplied by the controller. It commits only if that exact local delta improves.
"""
from pathlib import Path
import re,subprocess
R=Path(__file__).resolve().parent;s=(R/'snapshot/bench_rnasim_gpu.cu').read_text()
def extract(name):
 m=re.search(r'^static[^\n]*\b'+name+r'\(',s,re.M);assert m,name
 b=s.index('{',m.start());j=b+1;level=1
 while level:level+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[m.start():j]
src='''#include <cstdint>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <omp.h>
struct RnaSimGpuWorkspace {};
static int target[2], applied[2];static const double *distances;static const int32_t *group_id;
'''
for name in ['compute_tree_summaries_blind','compute_outgroup_dyadic_profile','build_component_waves','update_one_node_summary']:src+=extract(name)+'\n'
f=extract('run_subtree_minibatch_window');src+=f[:f.index('{')]+'{\n'+'''
 int t=sub_root==target[0]?0:(sub_root==target[1]?1:-1);
 if(t<0||applied[t])return 0;
 int g=3*t;double out[3]={0,0,0};
 for(size_t p=0;p<outgroup_leaves.size();p++)for(int i=0;i<3;i++)out[i]+=outgroup_weights[p]*distances[(g+i)*10+group_id[outgroup_leaves[p]]];
 double delta=.25*(distances[g*10+g+2]+out[1]-distances[g*10+g+1]-out[2]);
 if(delta>=-1e-6)return 0;
 int inner=curr_c1[sub_root];std::swap(curr_c2[inner],curr_c2[sub_root]);
 update_one_node_summary(inner,curr_c1[inner],curr_c2[inner],parent_r.data(),leaf_cnt.data(),node_rep0.data());
 update_one_node_summary(sub_root,curr_c1[sub_root],curr_c2[sub_root],parent_r.data(),leaf_cnt.data(),node_rep0.data());
 applied[t]=1;return delta;
}
'''
src+=extract('run_multi_gpu_minibatch_bme')+'\n'
src+='''
static double graph_score(int N,const std::vector<int32_t>&a,const std::vector<int32_t>&b){
 int root=2*N-2;std::vector<std::vector<int>>adj(root);for(int u=N;u<root;u++)for(int c:{a[u],b[u]}){adj[u].push_back(c);adj[c].push_back(u);}adj[a[root]].push_back(b[root]);adj[b[root]].push_back(a[root]);
 double L=0;struct Entry{int u,p,d;};std::vector<Entry>todo;todo.reserve(root);
 for(int i=0;i<N;i++){todo.clear();todo.push_back({i,-1,0});for(size_t k=0;k<todo.size();k++){auto x=todo[k];if(x.u<N&&x.u>i)L+=ldexp(distances[group_id[i]*10+group_id[x.u]],1-x.d);for(int v:adj[x.u])if(v!=x.p)todo.push_back({v,x.u,x.d+1});}}
 return L;
}
extern "C" void controlled_wave(int N,const int32_t*a,const int32_t*b,int ua,int ub,const int32_t*groups,const double*D,double*out){target[0]=ua;target[1]=ub;applied[0]=applied[1]=0;distances=D;group_id=groups;std::vector<int32_t>A(a,a+2*N-1),B(b,b+2*N-1);std::vector<uint64_t*>seqs(1,nullptr);std::vector<RnaSimGpuWorkspace>work(1);out[0]=graph_score(N,A,B);out[1]=run_multi_gpu_minibatch_bme(N,25,1,seqs,work,A,B);out[2]=graph_score(N,A,B);out[3]=applied[0]+applied[1];}
'''
(R/'wave_driver.cpp').write_text(src)
subprocess.run(['g++','-O2','-shared','-fPIC','-std=c++17','-fopenmp','wave_driver.cpp','-o','wave_driver.so'],cwd=R,check=True)
