#define main source_main
#include "cuda_validator_injected.cu"
#undef main
int main(){const int N=6; int64_t v[N-1]={0,0,0,0,0};double D[N*N];for(int i=0;i<N;i++)for(int j=0;j<N;j++)D[i*N+j]=i==j?0:1;D[2]=D[2*N]=10;const char *modes[]={"nan","index","wrong"};for(const char *mode:modes){setenv("REVIEW_FAULT",mode,1);double a,b,c;int ok=run_and_validate_cuda_pipeline(v,D,N,NULL,NULL,NULL,&a,&b,&c,0);printf("fault=%s validator_ok=%d errors=%.17g %.17g %.17g\n",mode,ok,a,b,c);if(ok)return 1;}unsetenv("REVIEW_FAULT");int ok=run_and_validate_cuda_pipeline(v,D,N,NULL,NULL,NULL,NULL,NULL,NULL,0);printf("unmodified validator_ok=%d\n",ok);return ok?0:1;}
