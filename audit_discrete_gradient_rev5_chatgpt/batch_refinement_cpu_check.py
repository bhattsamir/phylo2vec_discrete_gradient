"""Serial barrier-phase emulation of the late 16-pair shared-input branch.
The already tested scalar fallback is omitted here, leaving untouched outputs NaN.
"""
from pathlib import Path
import re,subprocess,ctypes,json
import numpy as np
R=Path(__file__).resolve().parent;s=(R/'late_snapshot/bench_rnasim_gpu.cu').read_text()
def extract(name):
 m=re.search(r'^(?:__global__|__device__)[^\n]*\b'+name+r'\(',s,re.M);assert m
 b=s.index('{',m.start());j=b+1;level=1
 while level:level+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[m.start():j]
f=extract('k_refine_real_seq_condensed_D_pairs');f=f[:f.index('    int32_t a = d_pair_a[pid];')]+'}\n';f=f.replace('__global__','static')
f=f.replace('for (int32_t idx = threadIdx.x; idx < batch_sz * 125; idx += blockDim.x)', 'for (int32_t idx = 0; idx < batch_sz * 125; ++idx)')
f=f.replace('if (threadIdx.x < (uint32_t)batch_sz)', 'for (threadIdx.x=0;threadIdx.x<batch_sz;threadIdx.x++)')
f=f.replace('        double my_sums[16];','        double all_sums[256][16];\n        for(threadIdx.x=0;threadIdx.x<256;threadIdx.x++){\n        double*my_sums=all_sums[threadIdx.x];')
f=f.replace('        __shared__ double s_red[256];','        }\n        __shared__ double s_red[256];')
f=f.replace('s_red[threadIdx.x] = my_sums[k];','for(int t=0;t<256;t++)s_red[t] = all_sums[t][k];')
f=f.replace('if (threadIdx.x < s) s_red[threadIdx.x] += s_red[threadIdx.x + s];','for(int t=0;t<s;t++)s_red[t] += s_red[t+s];').replace('if (threadIdx.x == 0)', 'if (true)')
src='''#include <cstdint>
#include <cmath>
#include <vector>
#define __device__
#define __forceinline__ inline
#define __shared__
#define __syncthreads() ((void)0)
#define __popcll __builtin_popcountll
struct Idx {int x;};static Idx blockIdx={0},blockDim={256},threadIdx={0};
'''+extract('dev_jc69_dist_popc')+'\n'+f+'''
extern "C" void run(int K,int n,const int*pa,const int*pb,const int*off,const float*w,const uint64_t*seq,double*out,double*ref){std::vector<int>ls(off[K]);for(int i=0;i<off[K];i++)ls[i]=i;std::vector<float>D(K*K);for(int p=0;p<n;p++){blockIdx.x=p;k_refine_real_seq_condensed_D_pairs(K,25,n,pa,pb,off,ls.data(),w,seq,D.data(),out);double sum=0;for(int a=off[pa[p]];a<off[pa[p]+1];a++)for(int b=off[pb[p]];b<off[pb[p]+1];b++)sum+=(double)w[a]*(double)w[b]*(double)dev_jc69_dist_popc(a,b,seq,25);ref[p]=sum;}}
'''
(R/'batch_refinement_cpu.cpp').write_text(src);subprocess.run(['g++','-O2','-shared','-fPIC','-std=c++17','batch_refinement_cpu.cpp','-o','batch_refinement_cpu.so'],cwd=R,check=True)
g=ctypes.CDLL(str(R/'batch_refinement_cpu.so'));ip=ctypes.POINTER(ctypes.c_int32);fp=ctypes.POINTER(ctypes.c_float);up=ctypes.POINTER(ctypes.c_uint64);dp=ctypes.POINTER(ctypes.c_double)
g.run.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ip,fp,up,dp,dp]
rng=np.random.default_rng(763);lens=np.array([1]*64+[257,3,2048],np.int32);off=np.r_[0,np.cumsum(lens)].astype(np.int32);n=int(off[-1]);K=len(lens)
states=rng.integers(0,4,(n,1600),dtype=np.uint8);valid=rng.random((n,1600))>.1;valid[off[65]:off[66]]=False
packed=np.zeros((n,125),np.uint64)
for c in range(5):
 bits=valid if c==4 else (states==c)&valid
 packed[:,c*25:(c+1)*25]=np.packbits(bits,axis=1,bitorder='little').copy().view(np.uint64)
w=np.concatenate([np.full(int(l),1/int(l),np.float32) for l in lens]);w[::31]=0
results=[]
for size in [1,2,3,15,16,17,31,32,33,48]:
 for mode in range(5):
  a=np.arange(size,dtype=np.int32);b=np.full(size,64 if mode<3 else 66,np.int32)
  if mode==1:b[::7]=65
  if mode==2:a[::9]=65
  if mode==4:b[15::16]=65
  expected=np.zeros(size,bool)
  for start in range(0,size,16):
   common=int(b[start]);end=start
   if lens[common]>1:
    while end<min(size,start+16) and b[end]==common and lens[a[end]]==1:end+=1
   if end-start>=2:expected[start:end]=True
  out=np.full(size,np.nan);ref=np.zeros(size)
  g.run(K,size,a.ctypes.data_as(ip),b.ctypes.data_as(ip),off.ctypes.data_as(ip),w.ctypes.data_as(fp),packed.ctypes.data_as(up),out.ctypes.data_as(dp),ref.ctypes.data_as(dp))
  assert np.array_equal(np.isfinite(out),expected),(size,mode,out,expected)
  err=float(np.max(abs(out[expected]-ref[expected]),initial=0))
  assert err<1e-12,(size,mode,err)
  results.append({'pairs':size,'mode':mode,'batched_outputs':int(expected.sum()),'error':err})
res={'cases':len(results),'batched_outputs':sum(x['batched_outputs'] for x in results),'max_error':max(x['error'] for x in results),'all':results,'scope':'CPU phase emulation of new batching branch, scalar fallback omitted, native CUDA not executed.'}
(R/'batch_refinement_cpu_check.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({k:v for k,v in res.items() if k!='all'}))
