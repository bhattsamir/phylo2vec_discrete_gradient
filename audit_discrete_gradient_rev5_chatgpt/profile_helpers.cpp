#include <cstdint>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstring>
#include <cmath>
namespace rna {
static void compute_tree_summaries_blind(
    const int32_t *c1_r, const int32_t *c2_r, int32_t N,
    std::vector<int32_t> &ord_out,
    std::vector<int32_t> &parent_r_out,
    std::vector<int32_t> &leaf_cnt_out,
    std::vector<int32_t> &node_rep0_out
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;

    ord_out.resize(N - 1);
    parent_r_out.assign(n_nodes_r, -1);
    int32_t ord_len = 0;
    ord_out[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord_out[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        parent_r_out[b0] = u;
        parent_r_out[b1] = u;
        if (b0 >= N) ord_out[ord_len++] = b0;
        if (b1 >= N) ord_out[ord_len++] = b1;
    }

    leaf_cnt_out.assign(n_nodes_r, 1);
    node_rep0_out.resize(n_nodes_r);
    std::vector<int32_t> min_depth(n_nodes_r, 0);
    for (int32_t x = 0; x < N; ++x) {
        node_rep0_out[x] = x;
    }
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord_out[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        leaf_cnt_out[u] = leaf_cnt_out[b0] + leaf_cnt_out[b1];
        if (min_depth[b0] <= min_depth[b1]) {
            min_depth[u] = 1 + min_depth[b0];
            node_rep0_out[u] = node_rep0_out[b0];
        } else {
            min_depth[u] = 1 + min_depth[b1];
            node_rep0_out[u] = node_rep0_out[b1];
        }
    }
}
static void compute_clade_dyadic_profile(
    int32_t u, int32_t N,
    const int32_t *c1_r, const int32_t *c2_r,
    const int32_t *node_rep0,
    int32_t max_prof_leaves,
    std::vector<int32_t> &prof_leaves_out,
    std::vector<float> &prof_weights_out
) {
    if (u < N) {
        prof_leaves_out.push_back(u);
        prof_weights_out.push_back(1.0f);
        return;
    }
    if (max_prof_leaves < 0) {
        struct StackEntry { int32_t u; double weight; };
        std::vector<StackEntry> st;
        st.reserve(64);
        st.push_back({u, 1.0});
        while (!st.empty()) {
            StackEntry top = st.back();
            st.pop_back();
            int32_t b0 = c1_r[top.u], b1 = c2_r[top.u];
            double hw = 0.5 * top.weight;
            if (b1 < N) {
                prof_leaves_out.push_back(b1);
                prof_weights_out.push_back((float)hw);
            } else {
                st.push_back({b1, hw});
            }
            if (b0 < N) {
                prof_leaves_out.push_back(b0);
                prof_weights_out.push_back((float)hw);
            } else {
                st.push_back({b0, hw});
            }
        }
        return;
    }
    struct WeightedNode {
        double weight;
        int32_t u;
        bool operator<(const WeightedNode &o) const {
            if (weight != o.weight) return weight < o.weight;
            return u > o.u;
        }
    };
    std::priority_queue<WeightedNode> pq;
    pq.push({1.0, u});
    int32_t start_leaves = (int32_t)prof_leaves_out.size();
    while (!pq.empty()) {
        int32_t curr_total = ((int32_t)prof_leaves_out.size() - start_leaves) + (int32_t)pq.size();
        if (max_prof_leaves > 0 && curr_total >= max_prof_leaves) break;
        WeightedNode top = pq.top();
        pq.pop();
        int32_t b0 = c1_r[top.u], b1 = c2_r[top.u];
        double hw = 0.5 * top.weight;
        if (b0 < N) {
            prof_leaves_out.push_back(b0);
            prof_weights_out.push_back((float)hw);
        } else {
            pq.push({hw, b0});
        }
        if (b1 < N) {
            prof_leaves_out.push_back(b1);
            prof_weights_out.push_back((float)hw);
        } else {
            pq.push({hw, b1});
        }
    }
    while (!pq.empty()) {
        WeightedNode top = pq.top();
        pq.pop();
        prof_leaves_out.push_back(node_rep0[top.u]);
        prof_weights_out.push_back((float)top.weight);
    }
}
static void compute_outgroup_dyadic_profile(
    int32_t sub_root, int32_t root_r, int32_t N,
    const int32_t *c1_r, const int32_t *c2_r,
    const int32_t *parent_r, const int32_t *node_rep0,
    int32_t max_prof_leaves,
    std::vector<int32_t> &prof_leaves_out,
    std::vector<float> &prof_weights_out
) {
    if (sub_root == root_r || parent_r[sub_root] == -1) return;
    int32_t p0 = parent_r[sub_root];
    int32_t start_u = (p0 == root_r)
        ? ((c1_r[root_r] == sub_root) ? c2_r[root_r] : c1_r[root_r])
        : p0;
    if (start_u < N) {
        prof_leaves_out.push_back(start_u);
        prof_weights_out.push_back(1.0f);
        return;
    }
    if (max_prof_leaves < 0) {
        struct DirectedStackEntry { int32_t u; int32_t from_u; double weight; };
        std::vector<DirectedStackEntry> st;
        st.reserve(64);
        st.push_back({start_u, sub_root, 1.0});
        while (!st.empty()) {
            DirectedStackEntry top = st.back();
            st.pop_back();
            int32_t v = top.u;
            int32_t n0 = c1_r[v];
            int32_t n1 = c2_r[v];
            int32_t pv = parent_r[v];
            int32_t n2 = (pv == root_r)
                ? ((c1_r[root_r] == v) ? c2_r[root_r] : c1_r[root_r])
                : pv;
            int32_t nbs[3] = {n0, n1, n2};
            double hw = 0.5 * top.weight;
            for (int k = 2; k >= 0; --k) {
                int32_t nb = nbs[k];
                if (nb == top.from_u || nb < 0) continue;
                if (nb < N) {
                    prof_leaves_out.push_back(nb);
                    prof_weights_out.push_back((float)hw);
                } else {
                    st.push_back({nb, v, hw});
                }
            }
        }
        return;
    }
    struct DirectedWeightedNode {
        double weight;
        int32_t u;
        int32_t from_u;
        bool operator<(const DirectedWeightedNode &o) const {
            if (weight != o.weight) return weight < o.weight;
            return u > o.u;
        }
    };
    std::priority_queue<DirectedWeightedNode> pq;
    pq.push({1.0, start_u, sub_root});
    int32_t start_leaves = (int32_t)prof_leaves_out.size();
    while (!pq.empty()) {
        int32_t curr_total = ((int32_t)prof_leaves_out.size() - start_leaves) + (int32_t)pq.size();
        if (max_prof_leaves > 0 && curr_total >= max_prof_leaves) break;
        DirectedWeightedNode top = pq.top();
        pq.pop();
        int32_t v = top.u;
        int32_t n0 = c1_r[v];
        int32_t n1 = c2_r[v];
        int32_t pv = parent_r[v];
        int32_t n2 = (pv == root_r)
            ? ((c1_r[root_r] == v) ? c2_r[root_r] : c1_r[root_r])
            : pv;
        int32_t nbs[3] = {n0, n1, n2};
        double hw = 0.5 * top.weight;
        for (int k = 0; k < 3; ++k) {
            int32_t nb = nbs[k];
            if (nb == top.from_u || nb < 0) continue;
            if (nb < N) {
                prof_leaves_out.push_back(nb);
                prof_weights_out.push_back((float)hw);
            } else {
                pq.push({hw, nb, v});
            }
        }
    }
    while (!pq.empty()) {
        DirectedWeightedNode top = pq.top();
        pq.pop();
        int32_t v = top.u;
        int32_t rep_leaf;
        if (top.from_u != c1_r[v] && top.from_u != c2_r[v]) {
            rep_leaf = node_rep0[v];
        } else {
            int32_t sib = (top.from_u == c1_r[v]) ? c2_r[v] : c1_r[v];
            rep_leaf = node_rep0[sib];
        }
        prof_leaves_out.push_back(rep_leaf);
        prof_weights_out.push_back((float)top.weight);
    }
}
static __attribute__((unused)) void build_condensed_D_from_profiles_host(
    int32_t K, int32_t N,
    const int32_t *prof_offsets,
    const int32_t *prof_leaves,
    const float *prof_weights,
    const double *D_full,
    double *D_cond_out
) {
    for (int32_t a = 0; a < K; ++a) {
        D_cond_out[(size_t)a * K + a] = 0.0;
        int32_t off_a = prof_offsets[a], end_a = prof_offsets[a + 1];
        for (int32_t b = a + 1; b < K; ++b) {
            int32_t off_b = prof_offsets[b], end_b = prof_offsets[b + 1];
            double d_val = 0.0;
            for (int32_t p = off_a; p < end_a; ++p) {
                int32_t la = prof_leaves[p];
                double wa = (double)prof_weights[p];
                double row_sum = 0.0;
                for (int32_t q = off_b; q < end_b; ++q) {
                    int32_t lb = prof_leaves[q];
                    double wb = (double)prof_weights[q];
                    row_sum += wb * D_full[(size_t)la * N + lb];
                }
                d_val += wa * row_sum;
            }
            D_cond_out[(size_t)a * K + b] = d_val;
            D_cond_out[(size_t)b * K + a] = d_val;
        }
    }
}
}
namespace rec {
static void compute_tree_summaries_blind(
    const int32_t *c1_r, const int32_t *c2_r, int32_t N,
    std::vector<int32_t> &ord,
    std::vector<int32_t> &curr_parent,
    std::vector<int32_t> &leaf_cnt,
    std::vector<int32_t> &node_rep0
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    ord.resize(N - 1);
    curr_parent.assign(n_nodes_r, -1);
    leaf_cnt.assign(n_nodes_r, 1);
    node_rep0.resize(n_nodes_r);
    std::vector<int32_t> min_depth(n_nodes_r, 0);

    int32_t ord_len = 0;
    ord[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        curr_parent[b0] = u;
        curr_parent[b1] = u;
        if (b0 >= N) ord[ord_len++] = b0;
        if (b1 >= N) ord[ord_len++] = b1;
    }

    for (int32_t x = 0; x < N; ++x) {
        node_rep0[x] = x;
    }
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        leaf_cnt[u] = leaf_cnt[b0] + leaf_cnt[b1];
        if (min_depth[b0] <= min_depth[b1]) {
            min_depth[u] = 1 + min_depth[b0];
            node_rep0[u] = node_rep0[b0];
        } else {
            min_depth[u] = 1 + min_depth[b1];
            node_rep0[u] = node_rep0[b1];
        }
    }
}
static void compute_clade_dyadic_profile(
    int32_t u, int32_t N,
    const int32_t *c1_r, const int32_t *c2_r,
    const int32_t *node_rep0,
    int32_t max_prof_leaves,
    std::vector<int32_t> &prof_leaves_out,
    std::vector<float> &prof_weights_out
) {
    if (u < N) {
        prof_leaves_out.push_back(u);
        prof_weights_out.push_back(1.0f);
        return;
    }
    if (max_prof_leaves < 0) {
        struct StackEntry { int32_t u; double weight; };
        std::vector<StackEntry> st;
        st.reserve(64);
        st.push_back({u, 1.0});
        while (!st.empty()) {
            StackEntry top = st.back();
            st.pop_back();
            int32_t b0 = c1_r[top.u], b1 = c2_r[top.u];
            double hw = 0.5 * top.weight;
            if (b1 < N) {
                prof_leaves_out.push_back(b1);
                prof_weights_out.push_back((float)hw);
            } else {
                st.push_back({b1, hw});
            }
            if (b0 < N) {
                prof_leaves_out.push_back(b0);
                prof_weights_out.push_back((float)hw);
            } else {
                st.push_back({b0, hw});
            }
        }
        return;
    }
    struct WeightedNode {
        double weight;
        int32_t u;
        bool operator<(const WeightedNode &o) const {
            if (weight != o.weight) return weight < o.weight;
            return u > o.u;
        }
    };
    std::priority_queue<WeightedNode> pq;
    pq.push({1.0, u});
    int32_t start_leaves = (int32_t)prof_leaves_out.size();
    while (!pq.empty()) {
        int32_t curr_total = ((int32_t)prof_leaves_out.size() - start_leaves) + (int32_t)pq.size();
        if (max_prof_leaves > 0 && curr_total >= max_prof_leaves) break;
        WeightedNode top = pq.top();
        pq.pop();
        int32_t b0 = c1_r[top.u], b1 = c2_r[top.u];
        double hw = 0.5 * top.weight;
        if (b0 < N) {
            prof_leaves_out.push_back(b0);
            prof_weights_out.push_back((float)hw);
        } else {
            pq.push({hw, b0});
        }
        if (b1 < N) {
            prof_leaves_out.push_back(b1);
            prof_weights_out.push_back((float)hw);
        } else {
            pq.push({hw, b1});
        }
    }
    while (!pq.empty()) {
        WeightedNode top = pq.top();
        pq.pop();
        prof_leaves_out.push_back(node_rep0[top.u]);
        prof_weights_out.push_back((float)top.weight);
    }
}
}

extern "C" int profile_clade(int backend,int N,const int32_t*a,const int32_t*b,int u,int cap,int32_t*l,float*w){std::vector<int32_t>ord,par,cnt,rep,leaves;std::vector<float>weights;if(backend==0){rna::compute_tree_summaries_blind(a,b,N,ord,par,cnt,rep);rna::compute_clade_dyadic_profile(u,N,a,b,rep.data(),cap,leaves,weights);}else{rec::compute_tree_summaries_blind(a,b,N,ord,par,cnt,rep);rec::compute_clade_dyadic_profile(u,N,a,b,rep.data(),cap,leaves,weights);}std::copy(leaves.begin(),leaves.end(),l);std::copy(weights.begin(),weights.end(),w);return leaves.size();}
extern "C" int profile_out(int N,const int32_t*a,const int32_t*b,int u,int cap,int32_t*l,float*w){std::vector<int32_t>ord,par,cnt,rep,leaves;std::vector<float>weights;rna::compute_tree_summaries_blind(a,b,N,ord,par,cnt,rep);rna::compute_outgroup_dyadic_profile(u,2*N-2,N,a,b,par.data(),rep.data(),cap,leaves,weights);std::copy(leaves.begin(),leaves.end(),l);std::copy(weights.begin(),weights.end(),w);return leaves.size();}
extern "C" void condense(int K,int N,const int32_t*off,const int32_t*l,const float*w,const double*D,double*out){rna::build_condensed_D_from_profiles_host(K,N,off,l,w,D,out);}
