from pathlib import Path
import subprocess,re,json
R=Path(__file__).resolve().parent;s=(R/'snapshot/phylo2vec_cuda_8xh100.cu').read_text();needle='    double max_err_smem = 0.0, max_err_gmem = 0.0, max_err_tiled = 0.0;';assert s.count(needle)==1
# Three fault modes: NaN only, illegal index only, and a legal but wrong chosen column.
inject='''
    const char *mode=getenv("REVIEW_FAULT");
    if(mode)for(int32_t i=1;i<n_pairs;++i){
      if(mode[0]=='n')h_best_delta[i]=NAN;
      if(mode[0]=='i')h_best_j[i]=-999;
      if(mode[0]=='w')for(int32_t j=0;j<=2*i;++j)if(h_delta_gmem[(size_t)i*max_cols+j]>h_best_delta[i]+0.01f){h_best_j[i]=j;break;}
    }
'''
(R/'cuda_validator_injected.cu').write_text(s.replace(needle,inject+needle))
(R/'cuda_validator_check.cpp').write_text('''#define main source_main
#include "cuda_validator_injected.cu"
#undef main
int main(){const int N=6; int64_t v[N-1]={0,0,0,0,0};double D[N*N];for(int i=0;i<N;i++)for(int j=0;j<N;j++)D[i*N+j]=i==j?0:1;D[2]=D[2*N]=10;for(const char *mode:{"none","index","wrong"}){setenv("REVIEW_FAULT",mode,1);double a,b,c;int ok=run_and_validate_cuda_pipeline(v,D,N,NULL,NULL,NULL,&a,&b,&c,0);printf("fault=%s validator_ok=%d errors=%.17g %.17g %.17g\\n",mode,ok,a,b,c);if(ok)return 1;}unsetenv("REVIEW_FAULT");int ok=run_and_validate_cuda_pipeline(v,D,N,NULL,NULL,NULL,NULL,NULL,NULL,0);printf("unmodified validator_ok=%d\\n",ok);return ok?0:1;}
'''.replace('for(const char *mode:{"none","index","wrong"})','const char *modes[]={"nan","index","wrong"};for(const char *mode:modes)'))
subprocess.run(['g++','-O2','-std=c++17',str(R/'cuda_validator_check.cpp'),str(R/'core.o'),'-lpthread','-o',str(R/'cuda_validator_check')],check=True)
subprocess.run([str(R/'cuda_validator_check')],check=True)
c=(R/'snapshot/phylo2vec_fastme.c').read_text()
assert 'subdist[(size_t)i * n_nodes_u + i]' in c
assert 'memset(delta_full_out, 0, (size_t)n_pairs * (size_t)max_cols * sizeof(double))' in c
unsafe=[]
for lineno,line in enumerate(c.splitlines(),1):
 if re.search(r'subdist\[[^\]]*\*\s*n_nodes_u',line) and '(size_t)' not in line:unsafe.append([lineno,line])
(R/'index_expression_check.c').write_text('''#include <stdint.h>
#include <stdio.h>
int main(){int32_t n=23172,n_nodes_u=2*n-2,i=n_nodes_u-1;size_t index=(size_t)i*n_nodes_u+i;printf("N=%d diagonal index=%zu\\n",n,index);n=32770;int32_t n_pairs=n-1,max_cols=2*n_pairs-1;size_t entries=(size_t)n_pairs*(size_t)max_cols;printf("N=%d gradient entries=%zu\\n",n,entries);return index!=2147580963ULL||entries!=2147581953ULL;}
''')
subprocess.run(['gcc','-O1','-fsanitize=undefined','-fno-sanitize-recover=all',str(R/'index_expression_check.c'),'-o',str(R/'index_expression_check')],check=True)
subprocess.run([str(R/'index_expression_check')],check=True)
print('Remaining unpromoted subdist*n_nodes_u lines:',unsafe);assert not unsafe
