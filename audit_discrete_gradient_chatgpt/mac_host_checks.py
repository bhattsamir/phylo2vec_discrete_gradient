from pathlib import Path
import subprocess
R=Path(__file__).resolve().parent
s=(R/'snapshot/bench_parallel_gpu.m').read_text().split('static NSString *const kMetalShaderSource')[0]
s='\n'.join(x for x in s.splitlines() if not x.startswith('#import'))
s=s.replace('extern void calculate_discrete_gradient_full_c','extern "C" void calculate_discrete_gradient_full_c').replace('^(size_t tid)','[&](size_t tid)')
adapter='''#define restrict __restrict__
#include <cstddef>
using dispatch_queue_t=int;
const int QOS_CLASS_USER_INTERACTIVE=0;
static int dispatch_get_global_queue(int,int){return 0;}
template<class F> void dispatch_apply(size_t n,int,F f){for(size_t i=0;i<n;i++)f(i);}
'''
(R/'mac_host_port.cpp').write_text(adapter+s)
(R/'mac_host_check.cpp').write_text('''#include "mac_host_port.cpp"
#include <vector>
#include <random>
int main(){std::mt19937 gen(2309);double worst_full=0,worst_tiled=0;int bad=0;
for(int N:{6,16,33,96,256})for(int t=0;t<4;t++){
 int M=2*N-2,C=2*N-3;std::vector<int64_t> v(N-1);for(int i=0;i<N-1;i++)v[i]=t==0?0:t==1?i:gen()%(i+1);
 std::vector<double>D(N*N,0),ref((N-1)*C),full(ref.size()),tiled(ref.size());for(int i=0;i<N;i++)for(int j=i+1;j<N;j++)D[i*N+j]=D[j*N+i]=.1+(gen()%10000)/1000.;
 std::vector<int32_t> iw(50*N),ti(N_CPU_THREADS*N);std::vector<double>dw(M*M+10*M),td(N_CPU_THREADS*M),sl((N-1)*N+3*M),tt((size_t)N_CPU_THREADS*M*B_TILE);
 calculate_discrete_gradient_full_c(v.data(),D.data(),N,ref.data());
 calculate_discrete_gradient_multicore_f64(v.data(),D.data(),N,full.data(),iw.data(),dw.data(),ti.data(),td.data());
 calculate_discrete_gradient_tiled_f64(v.data(),D.data(),N,tiled.data(),iw.data(),sl.data(),ti.data(),tt.data());
 for(int i=1;i<N-1;i++)for(int j=0;j<=2*i;j++){size_t k=i*C+j;double a=fabs(ref[k]-full[k]),b=fabs(ref[k]-tiled[k]);worst_full=fmax(worst_full,a);worst_tiled=fmax(worst_tiled,b);if(!std::isfinite(a)||!std::isfinite(b)||a>1e-9||b>1e-9)bad++;}
}
printf("Serial dispatch adapter, 20 trees: bad_entries=%d, max_full_error=%.17g, max_tiled_error=%.17g\\n",bad,worst_full,worst_tiled);return bad!=0;}
''')
subprocess.run(['g++','-O2','-std=c++17',str(R/'mac_host_check.cpp'),str(R/'core.o'),'-o',str(R/'mac_host_check')],check=True)
subprocess.run([str(R/'mac_host_check')],check=True)
