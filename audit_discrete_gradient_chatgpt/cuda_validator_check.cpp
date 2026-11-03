#define main source_main
#include "cuda_validator_injected.cu"
#undef main
int main(){const int N=6; int64_t v[N-1]={0,0,0,0,0};double D[N*N];for(int i=0;i<N;i++)for(int j=0;j<N;j++)D[i*N+j]=i==j?0:1;D[2]=D[2*N]=10;double a,b,c;int ok=run_and_validate_cuda_pipeline(v,D,N,NULL,NULL,NULL,&a,&b,&c,0);printf("injected top1 gains=NaN and indices=-999: validator_ok=%d, full_errors=%.17g %.17g %.17g\n",ok,a,b,c);return ok==1?0:1;}
