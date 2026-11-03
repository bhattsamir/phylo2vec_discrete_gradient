#define main original_rnasim_main
#include "snapshot/bench_rnasim_gpu.cu"
#undef main
#include <random>
#include <set>
#include <cmath>

static double exact_score(const std::vector<int32_t>& adj,int N,const std::vector<float>& D) {
    int M=2*N-2; double L=0;
    for(int x=0;x<N;x++) {
        std::vector<int> todo={x}, par(M,-2),dist(M,0);par[x]=-1;
        for(size_t k=0;k<todo.size();k++) {
            int u=todo[k];
            for(int s=0;s<3;s++){int w=adj[3*u+s];if(w<0||w==par[u])continue;if(par[w]!=-2)return NAN;par[w]=u;dist[w]=dist[u]+1;todo.push_back(w);}
        }
        if((int)todo.size()!=M)return NAN;
        for(int y=x+1;y<N;y++)L+=(double)D[x*N+y]*ldexp(1.,1-dist[y]);
    }
    return L;
}

static void ladder(int N,std::vector<int32_t>&c1,std::vector<int32_t>&c2) {
    c1.assign(2*N-1,0);c2=c1;c1[N]=0;c2[N]=1;
    for(int k=1;k<N-1;k++){c1[N+k]=N+k-1;c2[N+k]=k+1;}
}
static int balanced_rec(int lo,int hi,int N,int &next,std::vector<int32_t>&a,std::vector<int32_t>&b){
    if(hi-lo==1)return lo;int mid=(lo+hi)/2;int l=balanced_rec(lo,mid,N,next,a,b),r=balanced_rec(mid,hi,N,next,a,b);int u=next++;a[u]=l;b[u]=r;return u;
}
static void host_checks() {
    for(int kind=0;kind<2;kind++)for(int N:{5000,8192}){
        std::vector<int32_t>a,b;
        ladder(N,a,b);if(kind){int next=N;balanced_rec(0,N,N,next,a,b);}
        std::vector<int32_t>ord,parent,count,reps;compute_tree_summaries_blind(a.data(),b.data(),N,ord,parent,count,reps);
        std::vector<uint8_t>active(2*N-1,0),comp;std::vector<std::vector<int32_t>> waves;
        build_component_waves(N,ord,a,b,1050,1050,comp,waves);
        std::set<int>leaves,internals;int kmin=1000000,kmax=0,skipped=0;
        for(auto &wave:waves)for(int sr:wave){std::vector<int32_t>roots,ins,l,r;int rid;extract_active_skeleton(sr,a.data(),b.data(),N,1,active,count,roots,ins,l,r,&rid,1200,comp.data());int K=roots.size()+(sr!=2*N-2);kmin=std::min(kmin,K);kmax=std::max(kmax,K);if(K<6||K>2048){skipped++;continue;}for(int u:roots)if(u<N)leaves.insert(u);for(int u:ins)internals.insert(u);}
        std::vector<int32_t>roots,ins,l,r;int rid;extract_active_skeleton(2*N-2,a.data(),b.data(),N,std::max(8,N/700),active,count,roots,ins,l,r,&rid,1200,NULL);
        printf("scheduler kind=%s N=%d globalK=%zu waves=%zu Kmin=%d Kmax=%d skipped=%d leaves=%zu internals=%zu totalInternals=%d\n",kind?"balanced":"ladder",N,roots.size(),waves.size(),kmin,kmax,skipped,leaves.size(),internals.size(),N-1);
    }
}

__global__ void own_column_internals(float* sd,const int32_t* a,const int32_t* b,int N,int M,int root) {
    int v=N+blockIdx.x*blockDim.x+threadIdx.x;if(v>=root)return;
    for(int u=N;u<root;u++)sd[(size_t)u*M+v]=.5f*(sd[(size_t)a[u]*M+v]+sd[(size_t)b[u]*M+v]);
}
static int gpu_checks(bool own_column=false) {
    std::mt19937 gen(10931);int total=0,bad=0,empty=0;double maxerr=0,maxL=0,maxRunDiff=0;
    for(int N:{6,17,64,129}){
        int M=2*N-2,R=2*N-1;RnaSimGpuWorkspace ws;ws.init(0,N);
        for(int t=0;t<8;t++){
            std::vector<int64_t>v(N-1);std::vector<int32_t>perm(N);
            for(int i=0;i<N-1;i++)v[i]=t==0?0:t==1?i:gen()%(i+1);for(int i=0;i<N;i++)perm[i]=i;
            std::vector<int32_t>a,b,vmap,lmin,next,par,sz,tin,adj;int other,rc0,rc1,rc2;
            build_skeleton_metadata(v.data(),N,a,b,vmap,lmin,next,par,sz,tin,adj,&other,&rc0,&rc1,&rc2);
            std::vector<float>D(N*N,0),hbest(N),prior;std::vector<int32_t>hj(N);
            for(int i=0;i<N;i++)for(int j=i+1;j<N;j++)D[i*N+j]=D[j*N+i]=.1f+(gen()%10000)/1000.f;
            CUDA_CHECK(cudaMemcpy(ws.d_D,D.data(),D.size()*sizeof(float),cudaMemcpyHostToDevice));
#define CP(dst,src) CUDA_CHECK(cudaMemcpy(ws.dst,src.data(),src.size()*sizeof(src[0]),cudaMemcpyHostToDevice))
            CP(d_perm,perm);CP(d_v_curr,v);CP(d_c1_r,a);CP(d_c2_r,b);CP(d_v_map,vmap);CP(d_m_lmin,lmin);CP(d_next_k,next);CP(d_parent_u,par);CP(d_sub_size,sz);CP(d_tin,tin);CP(d_adj_u,adj);
#undef CP
            double base=exact_score(adj,N,D);
            for(int rep=0;rep<3;rep++){
                CUDA_CHECK(cudaMemset(ws.d_subdist,0,M*M*sizeof(float)));
                dim3 block(16,16),grid((N+15)/16,(N+15)/16),tr((N+15)/16,(N-2+15)/16);
                k_copy_leaf_rows_perm<<<grid,block>>>(ws.d_D,ws.d_perm,ws.d_subdist,N,M);
                k_bottom_up_leaves<<<(N+255)/256,256>>>(ws.d_subdist,ws.d_c1_r,ws.d_c2_r,N,M,M-1);
                k_transpose_leaf_block<<<tr,block>>>(ws.d_subdist,N,M,M-1);
                if(own_column)own_column_internals<<<(N-2+255)/256,256>>>(ws.d_subdist,ws.d_c1_r,ws.d_c2_r,N,M,M-1);else k_bottom_up_internals<<<(N-2+255)/256,256>>>(ws.d_subdist,ws.d_c1_r,ws.d_c2_r,N,M,M-1);
                k_top_down_and_root<<<(M+255)/256,256>>>(ws.d_subdist,ws.d_c1_r,ws.d_c2_r,ws.d_tin,ws.d_sub_size,N,M,M-1,rc0,rc1,rc2);
                k_populate_nd_u<<<(N-1+255)/256,256>>>(ws.d_subdist,ws.d_adj_u,ws.d_nd_u,N,M,M-1);
                k_compute_bme_length<<<1,256>>>(ws.d_subdist,ws.d_parent_u,N,M,ws.d_curr_L);
                k_spr_rows_top1<<<N-2,128,M*sizeof(float)>>>(2,N,M,other,ws.d_v_curr,ws.d_v_map,ws.d_adj_u,ws.d_parent_u,ws.d_sub_size,ws.d_m_lmin,ws.d_next_k,ws.d_subdist,ws.d_nd_u,ws.d_row_scratch,ws.d_best_delta,ws.d_best_j);
                double L;CUDA_CHECK(cudaMemcpy(&L,ws.d_curr_L,sizeof(double),cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hbest.data(),ws.d_best_delta,N*sizeof(float),cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(hj.data(),ws.d_best_j,N*sizeof(int),cudaMemcpyDeviceToHost));
                maxL=std::max(maxL,std::abs(L-base));
                for(int i=1;i<N-1;i++){
                    int s=vmap[i+1],anchor=par[s],dest=hj[i];
                    if(dest<0){empty++;continue;}if(dest>=M-1){bad++;continue;}
                    auto changed=adj;apply_unrooted_spr_host(changed.data(),s,anchor,dest,par[dest]);
                    double exact=exact_score(changed,N,D)-base;
                    double err=std::abs(exact-hbest[i]);maxerr=std::max(maxerr,err);total++;
                    if(!std::isfinite(exact)||err>1e-4){bad++;printf("BAD N=%d t=%d row=%d dest=%d pred=%.9g truth=%.17g\n",N,t,i,dest,hbest[i],exact);}
                    if(rep>0)maxRunDiff=std::max(maxRunDiff,(double)std::abs(hbest[i]-prior[i]));
                }
                prior=hbest;
            }
        }
        ws.destroy();
    }
    printf("GPU own_column=%d empty=%d summary checked=%d bad=%d maxDeltaError=%.17g maxLengthError=%.17g maxRepeatDeltaDifference=%.17g\n",own_column,empty,total,bad,maxerr,maxL,maxRunDiff);
    return bad!=0;
}
int main(int argc,char**argv){ if(argc>1&&std::string(argv[1])=="gpu")return gpu_checks();if(argc>1&&std::string(argv[1])=="gpu-safe")return gpu_checks(true);host_checks();return 0; }
