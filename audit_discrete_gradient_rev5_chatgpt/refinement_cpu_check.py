"""CPU phase emulation of current CUDA refinement body, not native GPU execution.
Threads accumulate independently. Shared singleton-sequence loading is performed
serially before use, and the reduction runs in its original barrier phases.
"""
from pathlib import Path
import re,subprocess,ctypes,json
import numpy as np
R=Path(__file__).resolve().parent;s=(R/'snapshot/bench_rnasim_gpu.cu').read_text()
def extract(name):
 m=re.search(r'^(?:__global__|__device__)[^\n]*\b'+name+r'\(',s,re.M);assert m
 b=s.index('{',m.start());j=b+1;level=1
 while level:level+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[m.start():j]
f=extract('k_refine_real_seq_condensed_D_pairs').replace('__global__','static')
f=f.replace('    double my_sum = 0.0;', '    double s_sum[256];\n    for (threadIdx.x=0;threadIdx.x<256;threadIdx.x++) {\n    double my_sum = 0.0;')
old='''        if (threadIdx.x < (uint32_t)(5 * W)) {
            s_seq_a[threadIdx.x] = sx[threadIdx.x];
        }''';assert old in f
f=f.replace(old,'        for (int k=0;k<5*W;k++) s_seq_a[k]=sx[k];')
f=f.replace('    __shared__ double s_sum[256];\n    s_sum[threadIdx.x] = my_sum;', '    s_sum[threadIdx.x] = my_sum;\n    }')
f=f.replace('if (threadIdx.x < s) s_sum[threadIdx.x] += s_sum[threadIdx.x + s];','for(int t=0;t<s;t++) s_sum[t] += s_sum[t+s];').replace('if (threadIdx.x == 0)', 'if (true)')
src='''#include <cstdint>
#include <cmath>
#define __device__
#define __forceinline__ inline
#define __shared__
#define __syncthreads() ((void)0)
#define __popcll __builtin_popcountll
struct Idx {int x;};static Idx blockIdx={0},blockDim={256},threadIdx={0};
'''+extract('dev_jc69_dist_popc')+'\n'+f+'''
extern "C" double refine(int na,int nb,const float*w,const uint64_t*seq){int K=2,pa=0,pb=1;int off[3]={0,na,na+nb};int*leaves=new int[na+nb];for(int i=0;i<na+nb;i++)leaves[i]=i;float D[4]={0};double result=0;blockIdx.x=0;k_refine_real_seq_condensed_D_pairs(K,25,1,&pa,&pb,off,leaves,w,seq,D,&result);delete[]leaves;return result;}
extern "C" double independent_pair_sum(int na,int nb,const float*w,const uint64_t*seq){double total=0;for(int a=0;a<na;a++)for(int b=na;b<na+nb;b++)total+=(double)w[a]*(double)w[b]*(double)dev_jc69_dist_popc(a,b,seq,25);return total;}
'''
(R/'refinement_cpu.cpp').write_text(src);subprocess.run(['g++','-O2','-shared','-fPIC','-std=c++17','refinement_cpu.cpp','-o','refinement_cpu.so'],cwd=R,check=True)
g=ctypes.CDLL(str(R/'refinement_cpu.so'));fp=ctypes.POINTER(ctypes.c_float);up=ctypes.POINTER(ctypes.c_uint64)
for name in ['refine','independent_pair_sum']:
 fn=getattr(g,name);fn.argtypes=[ctypes.c_int,ctypes.c_int,fp,up];fn.restype=ctypes.c_double
rng=np.random.default_rng(9911);cases=[]
for na,nb in [(1,1),(1,3),(3,1),(1,255),(256,1),(1,257),(1,2048),(5,19),(19,5),(32,64)]:
 for mode in range(6):
  n=na+nb;states=np.zeros((n,1600),np.uint8);mut=rng.random((n,1600))<.3;states[mut]=rng.integers(0,4,np.count_nonzero(mut),dtype=np.uint8)
  valid=rng.random((n,1600))>=.05
  if mode==1:valid[:na]=False
  if mode==2:valid[na:]=False
  packed=np.zeros((n,125),np.uint64)
  for c in range(5):
   bits=valid if c==4 else (states==c)&valid
   packed[:,c*25:(c+1)*25]=np.packbits(bits,axis=1,bitorder='little').copy().view(np.uint64)
  w=np.r_[np.full(na,1/na),np.full(nb,1/nb)].astype(np.float32)
  if mode==3:w[::2]=0
  if mode==4:w[:na]=0
  if mode==5:w[na:]=0
  a=g.refine(na,nb,w.ctypes.data_as(fp),packed.ctypes.data_as(up));b=g.independent_pair_sum(na,nb,w.ctypes.data_as(fp),packed.ctypes.data_as(up));err=abs(a-b)
  assert err<1e-12,(na,nb,mode,a,b)
  cases.append({'na':na,'nb':nb,'mode':mode,'emulated':a,'reference':b,'error':err})
res={'checks':len(cases),'max_error':max(c['error'] for c in cases),'cases':cases,'scope':'Serial CPU barrier-phase emulation using current kernel body and source scalar distance function. Not native CUDA.'}
(R/'refinement_cpu_check.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='cases'}))
