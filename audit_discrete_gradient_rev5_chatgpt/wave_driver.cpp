#include <cstdint>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <omp.h>
struct RnaSimGpuWorkspace {};
static int target[2], applied[2];static std::vector<int> attempt_targets;static std::vector<double> attempt_deltas;static const double *distances;static const int32_t *group_id;
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
static void build_component_waves(
    int32_t N, const std::vector<int32_t> &ord,
    const std::vector<int32_t> &curr_c1, const std::vector<int32_t> &curr_c2,
    int32_t first_cap, int32_t max_cap,
    std::vector<uint8_t> &is_comp_root,
    std::vector<std::vector<int32_t>> &level_waves
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    is_comp_root.assign(n_nodes_r, 0);
    level_waves.clear();
    if (N < 6) return;

    std::vector<int32_t> rem_int(n_nodes_r, 0);
    bool any_closed = false;

    for (int32_t idx = (int32_t)ord.size() - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t c1 = curr_c1[u], c2 = curr_c2[u];
        int32_t r1 = rem_int[c1], r2 = rem_int[c2];
        int32_t cap = (!any_closed && first_cap > 0) ? first_cap : max_cap;

        if (r1 >= 15 && r2 >= 15 && (1 + r1 + r2 > cap)) {
            is_comp_root[c1] = 1;
            is_comp_root[c2] = 1;
            any_closed = true;
            r1 = 0;
            r2 = 0;
        } else if (1 + r1 + r2 > cap) {
            if (r1 >= r2) {
                if (r1 >= 5) {
                    is_comp_root[c1] = 1;
                    any_closed = true;
                    r1 = 0;
                }
            } else {
                if (r2 >= 5) {
                    is_comp_root[c2] = 1;
                    any_closed = true;
                    r2 = 0;
                }
            }
            if (1 + r1 + r2 > max_cap) {
                if (r1 >= 5) {
                    is_comp_root[c1] = 1;
                    any_closed = true;
                    r1 = 0;
                }
                if (r2 >= 5) {
                    is_comp_root[c2] = 1;
                    any_closed = true;
                    r2 = 0;
                }
            }
        }
        rem_int[u] = 1 + r1 + r2;
    }

    if (rem_int[root_r] < 5) {
        std::vector<int32_t> st = {root_r};
        int32_t best_c = -1;
        while (!st.empty()) {
            int32_t u = st.back();
            st.pop_back();
            int32_t ch[2] = {curr_c1[u], curr_c2[u]};
            for (int k = 0; k < 2; ++k) {
                int32_t c = ch[k];
                if (c >= N) {
                    if (is_comp_root[c]) {
                        if (best_c == -1 || rem_int[c] < rem_int[best_c]) best_c = c;
                    } else {
                        st.push_back(c);
                    }
                }
            }
        }
        if (best_c != -1) {
            is_comp_root[best_c] = 0;
            rem_int[root_r] += rem_int[best_c];
        }
    }
    is_comp_root[root_r] = 1;

    std::vector<int32_t> comp_lvl(n_nodes_r, 0);
    std::vector<int32_t> max_ch_lvl(n_nodes_r, -1);
    std::vector<int32_t> direct_leaves(n_nodes_r, 0);
    for (int32_t x = 0; x < N; ++x) direct_leaves[x] = 1;
    int32_t max_lvl = 0;
    for (int32_t idx = (int32_t)ord.size() - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t c1 = curr_c1[u], c2 = curr_c2[u];
        int32_t l1 = (c1 >= N && is_comp_root[c1]) ? (comp_lvl[c1] + 1) : max_ch_lvl[c1];
        int32_t l2 = (c2 >= N && is_comp_root[c2]) ? (comp_lvl[c2] + 1) : max_ch_lvl[c2];
        int32_t dl1 = (c1 >= N && is_comp_root[c1]) ? 0 : direct_leaves[c1];
        int32_t dl2 = (c2 >= N && is_comp_root[c2]) ? 0 : direct_leaves[c2];
        direct_leaves[u] = dl1 + dl2;
        max_ch_lvl[u] = std::max(l1, l2);
        if (is_comp_root[u]) {
            comp_lvl[u] = std::max(0, max_ch_lvl[u]);
            if (comp_lvl[u] > max_lvl) max_lvl = comp_lvl[u];
        }
    }

    level_waves.resize(max_lvl + 1);
    for (int32_t idx = (int32_t)ord.size() - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        if (is_comp_root[u] && direct_leaves[u] > 0) {
            level_waves[comp_lvl[u]].push_back(u);
        }
    }
    while (!level_waves.empty() && level_waves.back().empty()) {
        level_waves.pop_back();
    }
}
static inline void update_one_node_summary(
    int32_t u, int32_t b0, int32_t b1,
    int32_t *parent_r, int32_t *leaf_cnt, int32_t *node_rep0
) {
    parent_r[b0] = u;
    parent_r[b1] = u;
    leaf_cnt[u] = leaf_cnt[b0] + leaf_cnt[b1];
    node_rep0[u] = (leaf_cnt[b0] <= leaf_cnt[b1]) ? node_rep0[b0] : node_rep0[b1];
}
static double run_subtree_minibatch_window(
    int32_t sub_root, int32_t N, int32_t W, int32_t condense_thresh, int32_t max_iters,
    const std::vector<uint8_t> &node_has_active,
    const std::vector<int32_t> &outgroup_leaves,
    const std::vector<float> &outgroup_weights,
    const uint64_t *d_packed_seqs,
    RnaSimGpuWorkspace &gpu_ws,
    std::vector<int32_t> &curr_c1, std::vector<int32_t> &curr_c2,
    std::vector<int32_t> &parent_r, std::vector<int32_t> &leaf_cnt, std::vector<int32_t> &node_rep0,
    int32_t max_internals = 1200,
    const uint8_t *stop_at_comp_root = NULL,
    const int32_t *outgroup_c1 = NULL,
    const int32_t *outgroup_c2 = NULL,
    const int32_t *outgroup_par = NULL,
    const int32_t *outgroup_rep0 = NULL
) {

 int t=sub_root==target[0]?0:(sub_root==target[1]?1:-1);
 if(t<0||applied[t])return 0;
 int g=3*t;double out[3]={0,0,0};
 for(size_t p=0;p<outgroup_leaves.size();p++)for(int i=0;i<3;i++)out[i]+=outgroup_weights[p]*distances[(g+i)*10+group_id[outgroup_leaves[p]]];
 double delta=.25*(distances[g*10+g+2]+out[1]-distances[g*10+g+1]-out[2]);
 attempt_targets.push_back(t);attempt_deltas.push_back(delta);
 if(delta>=-1e-6)return 0;
 int inner=curr_c1[sub_root];std::swap(curr_c2[inner],curr_c2[sub_root]);
 update_one_node_summary(inner,curr_c1[inner],curr_c2[inner],parent_r.data(),leaf_cnt.data(),node_rep0.data());
 update_one_node_summary(sub_root,curr_c1[sub_root],curr_c2[sub_root],parent_r.data(),leaf_cnt.data(),node_rep0.data());
 applied[t]=1;return delta;
}
static double run_multi_gpu_minibatch_bme(
    int32_t N, int32_t W, int32_t n_gpus,
    std::vector<uint64_t *> &d_packed_seqs,
    std::vector<RnaSimGpuWorkspace> &mb_ws,
    std::vector<int32_t> &curr_c1,
    std::vector<int32_t> &curr_c2
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    const int32_t max_prof_leaves = (N <= 2048) ? 0 : 64;
    std::vector<int32_t> ord, parent_r, leaf_cnt, node_rep0;
    compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);

    double total_mb_delta_L = 0.0;
    std::vector<uint8_t> node_has_active(n_nodes_r, 0);
    std::vector<int32_t> empty_out_leaves;
    std::vector<float> empty_out_weights;

    /* Pass 1: Global Backbone (priority-queue bounded to max_internals = 1200 so K <= 1201 <= 2048) */
    if (N >= 5000) {
        int32_t bb_thresh = N / 700;
        if (bb_thresh < 8) bb_thresh = 8;
        total_mb_delta_L += run_subtree_minibatch_window(
            root_r, N, W, bb_thresh, 25, node_has_active, empty_out_leaves, empty_out_weights,
            d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
            1200, NULL
        );
        compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
    }

    /* Pass 2A: Leaf-Complete Connected Component Waves (100% of all N original leaves across all leaf-bearing internal components, K <= 1056 <= 2048) */
    std::vector<uint8_t> is_comp_root;
    std::vector<std::vector<int32_t>> level_waves;
    build_component_waves(N, ord, curr_c1, curr_c2, 1050, 1050, is_comp_root, level_waves);

    int32_t leaves_in_lvl0 = 0;
    if (!level_waves.empty()) {
        for (int32_t sr : level_waves[0]) leaves_in_lvl0 += leaf_cnt[sr];
    }

    (void)n_gpus;
    for (size_t lvl = 0; lvl < level_waves.size(); ++lvl) {
        const std::vector<int32_t> &wave = level_waves[lvl];
        std::vector<int32_t> out_leaves;
        std::vector<float> out_weights;
        for (int32_t idx = 0; idx < (int32_t)wave.size(); ++idx) {
            int32_t sr = wave[idx];
            out_leaves.clear();
            out_weights.clear();
            if (sr != root_r) {
                compute_outgroup_dyadic_profile(
                    sr, root_r, N, curr_c1.data(), curr_c2.data(), parent_r.data(), node_rep0.data(),
                    max_prof_leaves, out_leaves, out_weights
                );
            }
            double dL = run_subtree_minibatch_window(
                sr, N, W, 1, 25, node_has_active, out_leaves, out_weights,
                d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                1200, is_comp_root.data()
            );
            if (dL != 0.0) {
                total_mb_delta_L += dL;
                for (int32_t p = parent_r[sr]; p != -1; p = parent_r[p]) {
                    update_one_node_summary(p, curr_c1[p], curr_c2[p], parent_r.data(), leaf_cnt.data(), node_rep0.data());
                }
                parent_r[root_r] = -1;
            }
        }
        if (lvl + 1 < level_waves.size()) {
            compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
        }
    }

    /* Pass 2B: Staggered Boundary-Crossing Component Waves (when spine-heavy) + Bounded Medium Cross-Clade Skeletons */
    if (N > 1000) {
        if (leaves_in_lvl0 < (9 * N) / 10) {
            compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            std::vector<uint8_t> is_comp_root_stag;
            std::vector<std::vector<int32_t>> stag_waves;
            build_component_waves(N, ord, curr_c1, curr_c2, 525, 1050, is_comp_root_stag, stag_waves);
            for (int32_t lvl = (int32_t)stag_waves.size() - 1; lvl >= 0; --lvl) {
                const std::vector<int32_t> &wave = stag_waves[lvl];
                std::vector<int32_t> out_leaves;
                std::vector<float> out_weights;
                for (int32_t idx = 0; idx < (int32_t)wave.size(); ++idx) {
                    int32_t sr = wave[idx];
                    out_leaves.clear();
                    out_weights.clear();
                    if (sr != root_r) {
                        compute_outgroup_dyadic_profile(
                            sr, root_r, N, curr_c1.data(), curr_c2.data(), parent_r.data(), node_rep0.data(),
                            max_prof_leaves, out_leaves, out_weights
                        );
                    }
                    double dL = run_subtree_minibatch_window(
                        sr, N, W, 1, 25, node_has_active, out_leaves, out_weights,
                        d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                        1200, is_comp_root_stag.data()
                    );
                    if (dL != 0.0) {
                        total_mb_delta_L += dL;
                        for (int32_t p = parent_r[sr]; p != -1; p = parent_r[p]) {
                            update_one_node_summary(p, curr_c1[p], curr_c2[p], parent_r.data(), leaf_cnt.data(), node_rep0.data());
                        }
                        parent_r[root_r] = -1;
                    }
                }
                compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            }
            if (N >= 5000) {
                int32_t bb_thresh = std::max(8, N / 700);
                total_mb_delta_L += run_subtree_minibatch_window(
                    root_r, N, W, bb_thresh, 25, node_has_active, empty_out_leaves, empty_out_weights,
                    d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                    1200, NULL
                );
                compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            }
        }

        std::vector<int32_t> mid_roots;
        std::vector<int32_t> spine_above_p2b;
        {
            std::vector<int32_t> st = {root_r};
            while (!st.empty()) {
                int32_t u = st.back(); st.pop_back();
                if (u < N) continue;
                if (leaf_cnt[u] <= 3200) {
                    if (leaf_cnt[u] > 1050) mid_roots.push_back(u);
                } else {
                    spine_above_p2b.push_back(u);
                    st.push_back(curr_c2[u]);
                    st.push_back(curr_c1[u]);
                }
            }
        }
        std::vector<int32_t> out_leaves_p2b;
        std::vector<float> out_weights_p2b;
        for (int32_t idx = 0; idx < (int32_t)mid_roots.size(); ++idx) {
            int32_t sr = mid_roots[idx];
            out_leaves_p2b.clear();
            out_weights_p2b.clear();
            if (sr != root_r) {
                compute_outgroup_dyadic_profile(
                    sr, root_r, N, curr_c1.data(), curr_c2.data(), parent_r.data(), node_rep0.data(),
                    max_prof_leaves, out_leaves_p2b, out_weights_p2b
                );
            }
            int32_t c_thresh = std::max(4, leaf_cnt[sr] / 800);
            double dL = run_subtree_minibatch_window(
                sr, N, W, c_thresh, 25, node_has_active, out_leaves_p2b, out_weights_p2b,
                d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                1200, NULL
            );
            if (dL != 0.0) {
                total_mb_delta_L += dL;
                for (int32_t p = parent_r[sr]; p != -1; p = parent_r[p]) {
                    update_one_node_summary(p, curr_c1[p], curr_c2[p], parent_r.data(), leaf_cnt.data(), node_rep0.data());
                }
                parent_r[root_r] = -1;
            }
        }
    }

    return total_mb_delta_L;
}

static double graph_score(int N,const std::vector<int32_t>&a,const std::vector<int32_t>&b){
 int root=2*N-2;std::vector<std::vector<int>>adj(root);for(int u=N;u<root;u++)for(int c:{a[u],b[u]}){adj[u].push_back(c);adj[c].push_back(u);}adj[a[root]].push_back(b[root]);adj[b[root]].push_back(a[root]);
 double L=0;struct Entry{int u,p,d;};std::vector<Entry>todo;todo.reserve(root);
 for(int i=0;i<N;i++){todo.clear();todo.push_back({i,-1,0});for(size_t k=0;k<todo.size();k++){auto x=todo[k];if(x.u<N&&x.u>i)L+=ldexp(distances[group_id[i]*10+group_id[x.u]],1-x.d);for(int v:adj[x.u])if(v!=x.p)todo.push_back({v,x.u,x.d+1});}}
 return L;
}
extern "C" void controlled_wave(int N,const int32_t*a,const int32_t*b,int ua,int ub,int ngpus,const int32_t*groups,const double*D,double*out){attempt_targets.clear();attempt_deltas.clear();target[0]=ua;target[1]=ub;applied[0]=applied[1]=0;distances=D;group_id=groups;std::vector<int32_t>A(a,a+2*N-1),B(b,b+2*N-1);std::vector<uint64_t*>seqs(ngpus,nullptr);std::vector<RnaSimGpuWorkspace>work(ngpus);out[0]=graph_score(N,A,B);out[1]=run_multi_gpu_minibatch_bme(N,25,ngpus,seqs,work,A,B);out[2]=graph_score(N,A,B);out[3]=applied[0]+applied[1];}
extern "C" int trace(int32_t*targets,double*deltas){std::copy(attempt_targets.begin(),attempt_targets.end(),targets);std::copy(attempt_deltas.begin(),attempt_deltas.end(),deltas);return attempt_targets.size();}
