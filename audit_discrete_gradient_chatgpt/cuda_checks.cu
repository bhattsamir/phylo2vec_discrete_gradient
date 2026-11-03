#define main original_benchmark_main
#include "snapshot/phylo2vec_cuda_8xh100.cu"
#undef main
#include <vector>
#include <random>
int main() {
    std::mt19937 gen(2309); int failures=0; double worst=0;
    for(int N: {6,16,33,96,256}) for(int t=0;t<4;t++) {
        std::vector<int64_t> v(N-1);
        for(int i=0;i<N-1;i++) v[i]=(t==0?0:t==1?i:gen()%(i+1));
        std::vector<double> D(N*N,0);
        for(int i=0;i<N;i++) for(int j=i+1;j<N;j++) D[i*N+j]=D[j*N+i]=.1+(gen()%10000)/1000.;
        double a=0,b=0,c=0;
        int ok=run_and_validate_cuda_pipeline(v.data(),D.data(),N,NULL,NULL,NULL,&a,&b,&c,0);
        worst=fmax(worst,fmax(a,fmax(b,c)));
        if(!ok||a>3e-5||b>3e-5||c>3e-5) failures++;
        printf("N=%d case=%d ok=%d full_error=%.10g fallback_error=%.10g tiled_error=%.10g\n",N,t,ok,a,b,c);fflush(stdout);
    }
    printf("SUMMARY cases=20 failures=%d worst=%.17g\n",failures,worst);
    return failures!=0;
}
