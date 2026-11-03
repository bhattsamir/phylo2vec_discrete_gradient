#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cassert>
#define CUDA_CHECK(x) x
static const int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1;
static int cudaMemcpy(void*d,const void*s,size_t n,int){memcpy(d,s,n);return 0;}
static int cudaMemcpyAsync(void*d,const void*s,size_t n,int,int){memcpy(d,s,n);return 0;}
static int cudaStreamSynchronize(int){return 0;}
struct Workspace {
 int stream=0; std::vector<int32_t> offsets,leaves,pa,pb; std::vector<float> weights,D; std::vector<double> pairs;
 int32_t *d_exact_offsets,*d_exact_leaves,*d_pair_a,*d_pair_b;
 float *d_exact_weights,*d_D,*d_D_cond; double*d_pair_exact_D;
 Workspace(int K,int count,const float*in):offsets(K+1),leaves(count),pa(K*K),pb(K*K),weights(count),D(in,in+K*K),pairs(K*K){d_exact_offsets=offsets.data();d_exact_leaves=leaves.data();d_pair_a=pa.data();d_pair_b=pb.data();d_exact_weights=weights.data();d_D=d_D_cond=D.data();d_pair_exact_D=pairs.data();}
 void ensure_exact_prof_cap(size_t){}
};
static double review_delta; static int review_refined;
static void review_refine(int K,int n,Workspace&g,const double*full){review_refined=n;for(int p=0;p<n;p++){int a=g.d_pair_a[p],b=g.d_pair_b[p];double d=full[a*K+b];g.d_pair_exact_D[p]=d;g.d_D[a*K+b]=g.d_D[b*K+a]=(float)d;}}
namespace rna {
static void compute_skeleton_leaf_dists(
    const int64_t *v_cf, const int32_t *perm, int32_t K,
    std::vector<int32_t> &dist_out
) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t root_r = R_sk - 1;
    std::vector<int32_t> c1(R_sk), c2(R_sk), uf(R_sk);
    for (int32_t x = 0; x < K; ++x) uf[x] = x;
    for (int32_t k = 0; k < K - 1; ++k) {
        int32_t i = K - 2 - k;
        int32_t m = K + k;
        int32_t x = (int32_t)v_cf[i];
        int32_t y = i + 1;
        c1[m] = uf[y];
        c2[m] = uf[x];
        uf[x] = m;
    }
    std::vector<int32_t> depth(R_sk, 0);
    for (int32_t u = root_r; u >= K; --u) {
        depth[c1[u]] = depth[u] + 1;
        depth[c2[u]] = depth[u] + 1;
    }
    dist_out.assign((size_t)K * K, 0);
    std::vector<std::vector<int32_t>> sub_leaves(R_sk);
    for (int32_t x = 0; x < K; ++x) sub_leaves[x].push_back(x);
    for (int32_t u = K; u <= root_r; ++u) {
        int32_t b0 = c1[u], b1 = c2[u];
        const auto &L0 = sub_leaves[b0];
        const auto &L1 = sub_leaves[b1];
        int32_t sub = (u == root_r) ? 1 : (2 * depth[u]);
        for (int32_t x : L0) {
            int32_t ox = perm[x];
            int32_t dx = depth[x] - sub;
            for (int32_t y : L1) {
                int32_t oy = perm[y];
                int32_t d = dx + depth[y];
                dist_out[(size_t)ox * K + oy] = d;
                dist_out[(size_t)oy * K + ox] = d;
            }
        }
        if (u < root_r) {
            auto &Lu = sub_leaves[u];
            Lu.reserve(L0.size() + L1.size());
            Lu.insert(Lu.end(), L0.begin(), L0.end());
            Lu.insert(Lu.end(), L1.begin(), L1.end());
            std::vector<int32_t>().swap(sub_leaves[b0]);
            std::vector<int32_t>().swap(sub_leaves[b1]);
        }
    }
}
static double guard(int32_t K,const int32_t*prof_offsets,const int32_t*exact_prof_offsets,const int32_t*exact_prof_leaves,const float*exact_prof_weights,const std::vector<int64_t>&init_v_cf,const std::vector<int32_t>&init_perm,const std::vector<int64_t>&best_v_cf,const std::vector<int32_t>&best_perm,double init_L,double best_L,Workspace&gpu_ws,const double*full,std::vector<int64_t>&v_cf,std::vector<int32_t>&perm){
    if (best_L >= init_L - 1e-6) {
        v_cf = init_v_cf;
        perm = init_perm;
        return 0.0;
    }

    if (exact_prof_offsets != NULL && exact_prof_leaves != NULL && exact_prof_weights != NULL) {
        std::vector<uint8_t> is_capped(K, 0);
        bool any_capped = false;
        for (int32_t a = 0; a < K; ++a) {
            int32_t cap_len = prof_offsets[a + 1] - prof_offsets[a];
            int32_t ex_len  = exact_prof_offsets[a + 1] - exact_prof_offsets[a];
            if (ex_len != cap_len) {
                is_capped[a] = 1;
                any_capped = true;
            }
        }
        if (any_capped) {
            std::vector<int32_t> dist_init, dist_best;
            compute_skeleton_leaf_dists(init_v_cf.data(), init_perm.data(), K, dist_init);
            compute_skeleton_leaf_dists(best_v_cf.data(), best_perm.data(), K, dist_best);

            std::vector<int32_t> pair_a, pair_b;
            std::vector<int32_t> uncap_a, uncap_b;
            for (int32_t a = 0; a < K; ++a) {
                for (int32_t b = a + 1; b < K; ++b) {
                    int32_t db = dist_best[(size_t)a * K + b];
                    int32_t di = dist_init[(size_t)a * K + b];
                    if (db != di) {
                        if (is_capped[a] || is_capped[b]) {
                            pair_a.push_back(a);
                            pair_b.push_back(b);
                        } else {
                            uncap_a.push_back(a);
                            uncap_b.push_back(b);
                        }
                    }
                }
            }
            if (pair_a.empty() && uncap_a.empty()) {
                v_cf = init_v_cf;
                perm = init_perm;
                return 0.0;
            }

            std::vector<double> h_pair_exact_D(pair_a.size(), 0.0);
            if (!pair_a.empty()) {
                int32_t n_pairs = (int32_t)pair_a.size();
                int32_t total_exact_prof = exact_prof_offsets[K];
                gpu_ws.ensure_exact_prof_cap((size_t)total_exact_prof);
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_offsets, exact_prof_offsets, (size_t)(K + 1) * sizeof(int32_t),          cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_leaves,  exact_prof_leaves,  (size_t)total_exact_prof * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_weights, exact_prof_weights, (size_t)total_exact_prof * sizeof(float),   cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_pair_a, pair_a.data(), (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_pair_b, pair_b.data(), (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));

                review_refine(K, n_pairs, gpu_ws, full);
                CUDA_CHECK(cudaMemcpyAsync(h_pair_exact_D.data(), gpu_ws.d_pair_exact_D, (size_t)n_pairs * sizeof(double), cudaMemcpyDeviceToHost, gpu_ws.stream));
                CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
            }

            std::vector<float> h_D_cond;
            if (!uncap_a.empty()) {
                h_D_cond.resize((size_t)K * K);
                CUDA_CHECK(cudaMemcpyAsync(h_D_cond.data(), gpu_ws.d_D, (size_t)K * K * sizeof(float), cudaMemcpyDeviceToHost, gpu_ws.stream));
                CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
            }

            double exact_delta_L = 0.0;
            for (size_t pid = 0; pid < pair_a.size(); ++pid) {
                int32_t a = pair_a[pid], b = pair_b[pid];
                double w_diff = ldexp(1.0, 1 - dist_best[(size_t)a * K + b]) - ldexp(1.0, 1 - dist_init[(size_t)a * K + b]);
                exact_delta_L += w_diff * h_pair_exact_D[pid];
            }
            for (size_t uid = 0; uid < uncap_a.size(); ++uid) {
                int32_t a = uncap_a[uid], b = uncap_b[uid];
                double w_diff = ldexp(1.0, 1 - dist_best[(size_t)a * K + b]) - ldexp(1.0, 1 - dist_init[(size_t)a * K + b]);
                exact_delta_L += w_diff * (double)h_D_cond[(size_t)a * K + b];
            }

            review_delta = exact_delta_L;
            if (exact_delta_L >= -1e-6) {
                v_cf = init_v_cf;
                perm = init_perm;
                return 0.0;
            }
            v_cf = best_v_cf;
            perm = best_perm;
            return exact_delta_L;
        }
    }

    v_cf = best_v_cf;
    perm = best_perm;
    return best_L - init_L;
}
}
namespace rec {
static void compute_skeleton_leaf_dists(
    const int64_t *v_cf, const int32_t *perm, int32_t K,
    std::vector<int32_t> &dist_out
) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t root_r = R_sk - 1;
    std::vector<int32_t> c1(R_sk), c2(R_sk), uf(R_sk);
    for (int32_t x = 0; x < K; ++x) uf[x] = x;
    for (int32_t k = 0; k < K - 1; ++k) {
        int32_t i = K - 2 - k;
        int32_t m = K + k;
        int32_t x = (int32_t)v_cf[i];
        int32_t y = i + 1;
        c1[m] = uf[y];
        c2[m] = uf[x];
        uf[x] = m;
    }
    std::vector<int32_t> depth(R_sk, 0);
    for (int32_t u = root_r; u >= K; --u) {
        depth[c1[u]] = depth[u] + 1;
        depth[c2[u]] = depth[u] + 1;
    }
    dist_out.assign((size_t)K * K, 0);
    std::vector<std::vector<int32_t>> sub_leaves(R_sk);
    for (int32_t x = 0; x < K; ++x) sub_leaves[x].push_back(x);
    for (int32_t u = K; u <= root_r; ++u) {
        int32_t b0 = c1[u], b1 = c2[u];
        const auto &L0 = sub_leaves[b0];
        const auto &L1 = sub_leaves[b1];
        int32_t sub = (u == root_r) ? 1 : (2 * depth[u]);
        for (int32_t x : L0) {
            int32_t ox = perm[x];
            int32_t dx = depth[x] - sub;
            for (int32_t y : L1) {
                int32_t oy = perm[y];
                int32_t d = dx + depth[y];
                dist_out[(size_t)ox * K + oy] = d;
                dist_out[(size_t)oy * K + ox] = d;
            }
        }
        if (u < root_r) {
            auto &Lu = sub_leaves[u];
            Lu.reserve(L0.size() + L1.size());
            Lu.insert(Lu.end(), L0.begin(), L0.end());
            Lu.insert(Lu.end(), L1.begin(), L1.end());
            std::vector<int32_t>().swap(sub_leaves[b0]);
            std::vector<int32_t>().swap(sub_leaves[b1]);
        }
    }
}
static double guard(int32_t K,const int32_t*prof_offsets,const int32_t*exact_prof_offsets,const int32_t*exact_prof_leaves,const float*exact_prof_weights,const std::vector<int64_t>&init_v_cf,const std::vector<int32_t>&init_perm,const std::vector<int64_t>&best_v_cf,const std::vector<int32_t>&best_perm,double init_L,double best_L,Workspace&gpu_ws,const double*full,std::vector<int64_t>&v_cf,std::vector<int32_t>&perm){
    if (best_L >= init_L - 1e-6) {
        v_cf = init_v_cf;
        perm = init_perm;
        return 0.0;
    }

    if (exact_prof_offsets != NULL && exact_prof_leaves != NULL && exact_prof_weights != NULL) {
        std::vector<uint8_t> is_capped(K, 0);
        bool any_capped = false;
        for (int32_t a = 0; a < K; ++a) {
            int32_t cap_len = prof_offsets[a + 1] - prof_offsets[a];
            int32_t ex_len  = exact_prof_offsets[a + 1] - exact_prof_offsets[a];
            if (ex_len != cap_len) {
                is_capped[a] = 1;
                any_capped = true;
            }
        }
        if (any_capped) {
            std::vector<int32_t> dist_init, dist_best;
            compute_skeleton_leaf_dists(init_v_cf.data(), init_perm.data(), K, dist_init);
            compute_skeleton_leaf_dists(best_v_cf.data(), best_perm.data(), K, dist_best);

            std::vector<int32_t> pair_a, pair_b;
            std::vector<int32_t> uncap_a, uncap_b;
            for (int32_t a = 0; a < K; ++a) {
                for (int32_t b = a + 1; b < K; ++b) {
                    int32_t db = dist_best[(size_t)a * K + b];
                    int32_t di = dist_init[(size_t)a * K + b];
                    if (db != di) {
                        if (is_capped[a] || is_capped[b]) {
                            pair_a.push_back(a);
                            pair_b.push_back(b);
                        } else {
                            uncap_a.push_back(a);
                            uncap_b.push_back(b);
                        }
                    }
                }
            }
            if (pair_a.empty() && uncap_a.empty()) {
                v_cf = init_v_cf;
                perm = init_perm;
                return 0.0;
            }

            std::vector<double> h_pair_exact_D(pair_a.size(), 0.0);
            if (!pair_a.empty()) {
                int32_t n_pairs = (int32_t)pair_a.size();
                int32_t total_exact_prof = exact_prof_offsets[K];
                gpu_ws.ensure_exact_prof_cap((size_t)total_exact_prof);
                CUDA_CHECK(cudaMemcpy(gpu_ws.d_exact_offsets, exact_prof_offsets, (size_t)(K + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(gpu_ws.d_exact_leaves,  exact_prof_leaves,  (size_t)total_exact_prof * sizeof(int32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(gpu_ws.d_exact_weights, exact_prof_weights, (size_t)total_exact_prof * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(gpu_ws.d_pair_a, pair_a.data(), (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(gpu_ws.d_pair_b, pair_b.data(), (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));

                review_refine(K, n_pairs, gpu_ws, full);
                CUDA_CHECK(cudaMemcpy(h_pair_exact_D.data(), gpu_ws.d_pair_exact_D, (size_t)n_pairs * sizeof(double), cudaMemcpyDeviceToHost));
            }

            std::vector<float> h_D_cond;
            if (!uncap_a.empty()) {
                h_D_cond.resize((size_t)K * K);
                CUDA_CHECK(cudaMemcpy(h_D_cond.data(), gpu_ws.d_D_cond, (size_t)K * K * sizeof(float), cudaMemcpyDeviceToHost));
            }

            double exact_delta_L = 0.0;
            for (size_t pid = 0; pid < pair_a.size(); ++pid) {
                int32_t a = pair_a[pid], b = pair_b[pid];
                double w_diff = ldexp(1.0, 1 - dist_best[(size_t)a * K + b]) - ldexp(1.0, 1 - dist_init[(size_t)a * K + b]);
                exact_delta_L += w_diff * h_pair_exact_D[pid];
            }
            for (size_t uid = 0; uid < uncap_a.size(); ++uid) {
                int32_t a = uncap_a[uid], b = uncap_b[uid];
                double w_diff = ldexp(1.0, 1 - dist_best[(size_t)a * K + b]) - ldexp(1.0, 1 - dist_init[(size_t)a * K + b]);
                exact_delta_L += w_diff * (double)h_D_cond[(size_t)a * K + b];
            }

            review_delta = exact_delta_L;
            if (exact_delta_L >= -1e-6) {
                v_cf = init_v_cf;
                perm = init_perm;
                return 0.0;
            }
            v_cf = best_v_cf;
            perm = best_perm;
            return exact_delta_L;
        }
    }

    v_cf = best_v_cf;
    perm = best_perm;
    return best_L - init_L;
}
}
static void relabel_from_rooted_tree(
    const int32_t *c1_r,
    const int32_t *c2_r,
    int32_t n_leaves,
    int32_t flip_mode,
    int64_t *v_cf_out,
    int32_t *perm_out
) {
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;
    std::vector<int32_t> ord(n_leaves - 1);
    int32_t ord_len = 0;
    ord[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        if (b0 >= n_leaves) ord[ord_len++] = b0;
        if (b1 >= n_leaves) ord[ord_len++] = b1;
    }
    std::vector<int32_t> min_leaf(n_nodes_r);
    for (int32_t x = 0; x < n_leaves; ++x) min_leaf[x] = x;
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        min_leaf[u] = (min_leaf[b0] < min_leaf[b1]) ? min_leaf[b0] : min_leaf[b1];
    }
    std::vector<int32_t> rep_leaf(n_nodes_r, 0);
    int32_t rc1 = c1_r[root_r], rc2 = c2_r[root_r];
    int32_t r_stay = (min_leaf[rc1] < min_leaf[rc2]) ? rc1 : rc2;
    int32_t r_move = (min_leaf[rc1] < min_leaf[rc2]) ? rc2 : rc1;
    if (flip_mode == 1) { int32_t tmp = r_stay; r_stay = r_move; r_move = tmp; }
    rep_leaf[r_stay] = 0;
    rep_leaf[r_move] = 1;
    v_cf_out[0] = 0;
    int32_t next_label = 2;
    for (int32_t idx = 1; idx < ord_len; ++idx) {
        int32_t m = ord[idx];
        int32_t c1 = c1_r[m], c2 = c2_r[m];
        int32_t c_stay = (min_leaf[c1] < min_leaf[c2]) ? c1 : c2;
        int32_t c_move = (min_leaf[c1] < min_leaf[c2]) ? c2 : c1;
        if (flip_mode == 1) { int32_t tmp = c_stay; c_stay = c_move; c_move = tmp; }
        int32_t x = rep_leaf[m];
        int32_t y = next_label++;
        rep_leaf[c_stay] = x;
        rep_leaf[c_move] = y;
        v_cf_out[y - 1] = x;
    }
    for (int32_t old_l = 0; old_l < n_leaves; ++old_l) {
        perm_out[rep_leaf[old_l]] = old_l;
    }
}
extern "C" void labels(int K,const int32_t*a,const int32_t*b,int flip,int64_t*v,int32_t*p){relabel_from_rooted_tree(a,b,K,flip,v,p);}
extern "C" void distances(int backend,int K,const int64_t*v,const int32_t*p,int32_t*out){std::vector<int32_t>d;if(backend==0)rna::compute_skeleton_leaf_dists(v,p,K,d);else rec::compute_skeleton_leaf_dists(v,p,K,d);std::copy(d.begin(),d.end(),out);}
extern "C" double acceptance(int backend,int K,const int32_t*off,const int32_t*exoff,const int64_t*v0,const int32_t*p0,const int64_t*v1,const int32_t*p1,const float*cap,const double*full,double surrogate_delta,double*checked,int32_t*refined,int32_t*restored){
 std::vector<int64_t>iv(v0,v0+K-1),bv(v1,v1+K-1),v=bv;std::vector<int32_t>ip(p0,p0+K),bp(p1,p1+K),p=bp;
 std::vector<int32_t>ls(exoff[K]);std::vector<float>ws(exoff[K]);Workspace g(K,exoff[K],cap);review_delta=NAN;review_refined=0;
 double ret=backend==0?rna::guard(K,off,exoff,ls.data(),ws.data(),iv,ip,bv,bp,0,surrogate_delta,g,full,v,p):rec::guard(K,off,exoff,ls.data(),ws.data(),iv,ip,bv,bp,0,surrogate_delta,g,full,v,p);
 *checked=review_delta;*refined=review_refined;*restored=(v==iv&&p==ip);return ret;
}
