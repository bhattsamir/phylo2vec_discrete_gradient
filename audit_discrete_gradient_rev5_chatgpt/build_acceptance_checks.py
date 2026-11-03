"""Extract current host acceptance code. Substitute only GPU transport/refinement.
Refinement returns an independently supplied full-profile matrix. This is not a
native CUDA test. The source selection, distances, delta and rejection remain.
"""
from pathlib import Path
import re, subprocess
R=Path(__file__).resolve().parent

def extract(s,name):
 m=re.search(r'^static[^\n]*\b'+name+r'\(',s,re.M); assert m,name
 b=s.index('{',m.start());j=b+1;level=1
 while level: level+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[m.start():j]

src='''#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cassert>
#define CUDA_CHECK(x) x
static const int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1;
static int cudaMemcpy(void*d,const void*s,size_t n,int){memcpy(d,s,n);return 0;}
static int cudaMemcpyAsync(void*d,const void*s,size_t n,int,int){memcpy(d,s,n);return 0;}
static int cudaStreamSynchronize(int){return 0;}
struct Workspace {
 int stream=0; std::vector<int32_t> offsets,leaves,pa,pb; std::vector<float> weights,D; std::vector<double> pairs;
 int32_t *d_exact_offsets,*d_exact_leaves,*d_pair_a,*d_pair_b;
 float *d_exact_weights,*d_D,*d_D_cond; double*d_pair_exact_D;
 Workspace(int K,int count,const float*in):offsets(K+1),leaves(count),pa(K*K),pb(K*K),weights(count),D(in,in+K*K),pairs(K*K){d_exact_offsets=offsets.data();d_exact_leaves=leaves.data();d_pair_a=pa.data();d_pair_b=pb.data();d_exact_weights=weights.data();d_D=d_D_cond=D.data();d_pair_exact_D=pairs.data();}
 void ensure_exact_prof_cap(size_t){}
};
static double review_delta; static int review_refined;
static void review_refine(int K,int n,Workspace&g,const double*full){review_refined=n;for(int p=0;p<n;p++){int a=g.d_pair_a[p],b=g.d_pair_b[p];double d=full[a*K+b];g.d_pair_exact_D[p]=d;g.d_D[a*K+b]=g.d_D[b*K+a]=(float)d;}}
'''
for ns,f in [('rna','bench_rnasim_gpu.cu'),('rec','bench_minibatch_recovery.cu')]:
 s=(R/'snapshot'/f).read_text();fn=extract(s,'optimize_skeleton_on_gpu');tail=fn[fn.index('    if (best_L >= init_L - 1e-6)'):]
 tail=re.sub(r'k_refine_(real_seq|true)_condensed_D_pairs<<<.*?>>>\(.*?\);','review_refine(K, n_pairs, gpu_ws, full);',tail,flags=re.S)
 assert '<<<' not in tail
 tail=tail.replace('            if (exact_delta_L >= -1e-6)', '            review_delta = exact_delta_L;\n            if (exact_delta_L >= -1e-6)')
 src+='namespace '+ns+' {\n'+extract(s,'compute_skeleton_leaf_dists')+'\n'
 src+='''static double guard(int32_t K,const int32_t*prof_offsets,const int32_t*exact_prof_offsets,const int32_t*exact_prof_leaves,const float*exact_prof_weights,const std::vector<int64_t>&init_v_cf,const std::vector<int32_t>&init_perm,const std::vector<int64_t>&best_v_cf,const std::vector<int32_t>&best_perm,double init_L,double best_L,Workspace&gpu_ws,const double*full,std::vector<int64_t>&v_cf,std::vector<int32_t>&perm){
'''+tail+'\n}\n'
s=(R/'snapshot/bench_rnasim_gpu.cu').read_text();src+=extract(s,'relabel_from_rooted_tree')
src+='''
extern "C" void labels(int K,const int32_t*a,const int32_t*b,int flip,int64_t*v,int32_t*p){relabel_from_rooted_tree(a,b,K,flip,v,p);}
extern "C" void distances(int backend,int K,const int64_t*v,const int32_t*p,int32_t*out){std::vector<int32_t>d;if(backend==0)rna::compute_skeleton_leaf_dists(v,p,K,d);else rec::compute_skeleton_leaf_dists(v,p,K,d);std::copy(d.begin(),d.end(),out);}
extern "C" double acceptance(int backend,int K,const int32_t*off,const int32_t*exoff,const int64_t*v0,const int32_t*p0,const int64_t*v1,const int32_t*p1,const float*cap,const double*full,double surrogate_delta,double*checked,int32_t*refined,int32_t*restored){
 std::vector<int64_t>iv(v0,v0+K-1),bv(v1,v1+K-1),v=bv;std::vector<int32_t>ip(p0,p0+K),bp(p1,p1+K),p=bp;
 std::vector<int32_t>ls(exoff[K]);std::vector<float>ws(exoff[K]);Workspace g(K,exoff[K],cap);review_delta=NAN;review_refined=0;
 double ret=backend==0?rna::guard(K,off,exoff,ls.data(),ws.data(),iv,ip,bv,bp,0,surrogate_delta,g,full,v,p):rec::guard(K,off,exoff,ls.data(),ws.data(),iv,ip,bv,bp,0,surrogate_delta,g,full,v,p);
 *checked=review_delta;*refined=review_refined;*restored=(v==iv&&p==ip);return ret;
}
'''
(R/'acceptance_helpers.cpp').write_text(src)
subprocess.run(['g++','-O2','-shared','-fPIC','-std=c++17','acceptance_helpers.cpp','-o','acceptance_helpers.so'],cwd=R,check=True)
