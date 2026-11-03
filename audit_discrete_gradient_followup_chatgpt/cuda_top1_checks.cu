#define main original_main
#include "snapshot/phylo2vec_cuda_8xh100.cu"
#undef main
#include <vector>
#include <random>
static float input_D(int i,int j,uint32_t seed){if(i==j)return 0;uint32_t a=i<j?i:j,b=i<j?j:i;uint32_t h=seed^(a*0x9E3779B9U)^(b*0x85EBCA6BU);h^=h>>16;h*=0x7FEB352DU;h^=h>>15;h*=0x846CA68BU;h^=h>>16;return .1f+2.f*((float)(h&0xFFFFFFU)/16777216.f);}
int main(){std::mt19937 gen(2309);int bad=0;double worst=0;
for(int N:{6,33,257})for(int t=0;t<2;t++){
 std::vector<int64_t>v(N-1);for(int i=0;i<N-1;i++)v[i]=t==0?0:gen()%(i+1);
 std::vector<double>D(N*N),ref((N-1)*(2*N-3));for(int i=0;i<N;i++)for(int j=0;j<N;j++)D[i*N+j]=input_D(i,j,1337U+N);
 calculate_discrete_gradient_full_c(v.data(),D.data(),N,ref.data());double exactmin=0;for(int i=1;i<N-1;i++)for(int j=0;j<=2*i;j++)exactmin=fmin(exactmin,ref[i*(2*N-3)+j]);
 HostTreeTopology T;build_host_tree_topology(v.data(),N,&T);
 for(int mode=0;mode<3;mode++){float d;int32_t i,j;int ok=mode==0?run_full_subdist_top1_multi_gpu(N,&T,v.data(),1,NULL,NULL,NULL,&d,&i,&j):mode==1?run_tiled_top1_multi_gpu(N,&T,v.data(),1,NULL,NULL,&d,&i,&j):run_sharded_multi_gpu_top1(N,&T,v.data(),1,0,NULL,NULL,NULL,NULL,&d,&i,&j);
 bool valid=ok&&i>=1&&i<N-1&&j>=0&&j<=2*i;double value=valid?ref[i*(2*N-3)+j]:NAN;double err=fabs(d-value),gap=value-exactmin;bool pass=valid&&std::isfinite(err)&&err<1e-5&&gap<1e-5;bad+=!pass;worst=fmax(worst,err);
 printf("N=%d case=%d mode=%d pass=%d row=%d col=%d delta=%.9g actual=%.12g min_gap=%.9g\n",N,t,mode,pass,i,j,d,value,gap);fflush(stdout);
 }free_host_tree_topology(&T);
}printf("SUMMARY one GPU, 18 selections, failures=%d maxDeltaError=%.17g\n",bad,worst);return bad!=0;}
