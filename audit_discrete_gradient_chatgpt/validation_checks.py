from pathlib import Path
import subprocess
R=Path(__file__).resolve().parent
s=(R/'snapshot/phylo2vec_cuda_8xh100.cu').read_text()
needle='    double max_err_smem = 0.0, max_err_gmem = 0.0, max_err_tiled = 0.0;'
assert s.count(needle)==1
s=s.replace(needle,'    for (int32_t i=1; i<n_pairs; ++i) { h_best_delta[i] = NAN; h_best_j[i] = -999; }\n'+needle)
(R/'cuda_validator_injected.cu').write_text(s)
(R/'cuda_validator_check.cpp').write_text('''#define main source_main
#include "cuda_validator_injected.cu"
#undef main
int main(){const int N=6; int64_t v[N-1]={0,0,0,0,0};double D[N*N];for(int i=0;i<N;i++)for(int j=0;j<N;j++)D[i*N+j]=i==j?0:1;D[2]=D[2*N]=10;double a,b,c;int ok=run_and_validate_cuda_pipeline(v,D,N,NULL,NULL,NULL,&a,&b,&c,0);printf("injected top1 gains=NaN and indices=-999: validator_ok=%d, full_errors=%.17g %.17g %.17g\\n",ok,a,b,c);return ok==1?0:1;}
''')
subprocess.run(['g++','-O2','-std=c++17',str(R/'cuda_validator_check.cpp'),str(R/'core.o'),'-o',str(R/'cuda_validator_check')],check=True)
subprocess.run([str(R/'cuda_validator_check')],check=True)
(R/'index_expression_check.c').write_text('''#include <stdint.h>
#include <stdio.h>
int main(){volatile int32_t n=23172;int32_t m=2*n-2,i=m-1;int32_t index=i*m+i;printf("N=%d signed diagonal index=%d expected=%lld\\n",n,index,(long long)i*m+i);n=32770;int32_t pairs=n-1,cols=2*pairs-1;size_t entries=(size_t)(pairs*cols);printf("N=%d full gradient entries=%zu expected=%lld\\n",n,entries,(long long)pairs*cols);}
''')
subprocess.run(['gcc','-O1','-fsanitize=undefined',str(R/'index_expression_check.c'),'-o',str(R/'index_expression_check')],check=True)
subprocess.run([str(R/'index_expression_check')],check=True)
