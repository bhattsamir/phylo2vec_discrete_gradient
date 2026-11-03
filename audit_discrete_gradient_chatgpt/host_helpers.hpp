#include <vector>
#include <queue>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>

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

static void decode_cf_to_rooted(
    const int64_t *v, int32_t N,
    int32_t *c1_r, int32_t *c2_r, int32_t *scratch_uf
) {
    const int32_t n_pairs = N - 1;
    for (int32_t x = 0; x < N; ++x) scratch_uf[x] = x;
    for (int32_t k = 0; k < n_pairs; ++k) {
        int32_t i = n_pairs - 1 - k;
        int32_t m = N + k;
        int32_t x = (int32_t)v[i];
        int32_t y = i + 1;
        c1_r[m] = scratch_uf[y];
        c2_r[m] = scratch_uf[x];
        scratch_uf[x] = m;
    }
}

static void build_skeleton_metadata(
    const int64_t *v_cf, int32_t K,
    std::vector<int32_t> &c1_cf, std::vector<int32_t> &c2_cf,
    std::vector<int32_t> &v_map_cf, std::vector<int32_t> &m_lmin_cf,
    std::vector<int32_t> &next_k_cf, std::vector<int32_t> &parent_u_cf,
    std::vector<int32_t> &sub_sz_cf, std::vector<int32_t> &tin_cf,
    std::vector<int32_t> &adj_u_cf,
    int32_t *out_other_rc, int32_t *out_rc0, int32_t *out_rc1, int32_t *out_rc2
) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t M_sk = 2 * K - 2;
    const int32_t root_u_sk = M_sk - 1;

    c1_cf.assign(R_sk, 0); c2_cf.assign(R_sk, 0);
    v_map_cf.assign(K, 0);
    m_lmin_cf.assign(K, 0);
    next_k_cf.assign(K, 0);
    parent_u_cf.assign(M_sk, -1);
    sub_sz_cf.assign(M_sk, 1);
    tin_cf.assign(M_sk, 0);
    adj_u_cf.assign(3 * M_sk, -1);

    std::vector<int32_t> scratch_uf(R_sk), last_seen(K, K);
    for (int32_t x = 0; x < K; ++x) scratch_uf[x] = x;

    for (int32_t k = 0; k < K - 1; ++k) {
        int32_t i = K - 2 - k;
        int32_t m = K + k;
        int32_t x = (int32_t)v_cf[i];
        int32_t y = i + 1;
        int32_t b0 = scratch_uf[y], b1 = scratch_uf[x];
        c1_cf[m] = b0; c2_cf[m] = b1;
        m_lmin_cf[k] = x; v_map_cf[y] = b0;
        scratch_uf[x] = m;
    }
    for (int32_t k = K - 3; k >= 0; --k) {
        int32_t lmin = m_lmin_cf[k];
        next_k_cf[k] = last_seen[lmin];
        last_seen[lmin] = k;
    }
    int32_t rc1 = c1_cf[R_sk - 1], rc2 = c2_cf[R_sk - 1];
    int32_t other_rc = (rc1 < rc2) ? rc1 : rc2;
    int32_t r_c0 = c1_cf[root_u_sk], r_c1 = c2_cf[root_u_sk], r_c2 = other_rc;
    *out_other_rc = other_rc;
    *out_rc0 = r_c0; *out_rc1 = r_c1; *out_rc2 = r_c2;

    for (int32_t u = 0; u < K; ++u) sub_sz_cf[u] = 1;
    for (int32_t u = K; u < root_u_sk; ++u) {
        int32_t b0 = c1_cf[u], b1 = c2_cf[u];
        parent_u_cf[b0] = u; parent_u_cf[b1] = u;
        sub_sz_cf[u] = 1 + sub_sz_cf[b0] + sub_sz_cf[b1];
        adj_u_cf[3 * u + 0] = b0; adj_u_cf[3 * u + 1] = b1;
    }
    parent_u_cf[r_c0] = root_u_sk; parent_u_cf[r_c1] = root_u_sk; parent_u_cf[r_c2] = root_u_sk;
    parent_u_cf[root_u_sk] = -1;
    for (int32_t u = 0; u < K; ++u) {
        adj_u_cf[3 * u + 0] = parent_u_cf[u]; adj_u_cf[3 * u + 1] = -1; adj_u_cf[3 * u + 2] = -1;
    }
    for (int32_t u = K; u < root_u_sk; ++u) adj_u_cf[3 * u + 2] = parent_u_cf[u];
    adj_u_cf[3 * root_u_sk + 0] = r_c0; adj_u_cf[3 * root_u_sk + 1] = r_c1; adj_u_cf[3 * root_u_sk + 2] = r_c2;

    tin_cf[root_u_sk] = 0;
    tin_cf[r_c0] = 1;
    tin_cf[r_c1] = 1 + sub_sz_cf[r_c0];
    tin_cf[r_c2] = 1 + sub_sz_cf[r_c0] + sub_sz_cf[r_c1];
    for (int32_t u = root_u_sk - 1; u >= K; --u) {
        int32_t b0 = c1_cf[u], b1 = c2_cf[u];
        tin_cf[b0] = tin_cf[u] + 1;
        tin_cf[b1] = tin_cf[u] + 1 + sub_sz_cf[b0];
    }
}

static void compute_tree_summaries_blind(
    const int32_t *c1_r, const int32_t *c2_r, int32_t N,
    std::vector<int32_t> &ord_out,
    std::vector<int32_t> &parent_r_out,
    std::vector<int32_t> &leaf_cnt_out,
    std::vector<int32_t> &node_rep4_out
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
    node_rep4_out.resize(8 * n_nodes_r);
    for (int32_t x = 0; x < N; ++x) {
        for (int32_t r = 0; r < 8; ++r) node_rep4_out[8 * x + r] = x;
    }
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord_out[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        leaf_cnt_out[u] = leaf_cnt_out[b0] + leaf_cnt_out[b1];
        node_rep4_out[8 * u + 0] = node_rep4_out[8 * b0 + 0];
        node_rep4_out[8 * u + 1] = node_rep4_out[8 * b0 + 2];
        node_rep4_out[8 * u + 2] = node_rep4_out[8 * b0 + 4];
        node_rep4_out[8 * u + 3] = node_rep4_out[8 * b0 + 6];
        node_rep4_out[8 * u + 4] = node_rep4_out[8 * b1 + 0];
        node_rep4_out[8 * u + 5] = node_rep4_out[8 * b1 + 2];
        node_rep4_out[8 * u + 6] = node_rep4_out[8 * b1 + 4];
        node_rep4_out[8 * u + 7] = node_rep4_out[8 * b1 + 6];
    }
}

static void extract_active_skeleton(
    int32_t root_r, const int32_t *c1_r, const int32_t *c2_r, int32_t N,
    int32_t condense_thresh, const std::vector<uint8_t> &node_has_active,
    const std::vector<int32_t> &leaf_cnt,
    std::vector<int32_t> &super_roots,
    std::vector<int32_t> &split_internals,
    std::vector<int32_t> &skel_left,
    std::vector<int32_t> &skel_right,
    int32_t *out_root_skel_id,
    int32_t max_internals = 1200,
    const uint8_t *stop_at_comp_root = NULL
) {
    if (root_r < N) {
        super_roots.push_back(root_r);
        *out_root_skel_id = -1;
        return;
    }

    struct ExpNode { int32_t u; int32_t left_exp; int32_t right_exp; };
    struct PqCand {
        int32_t prio;
        int32_t u;
        int32_t parent_exp;
        int32_t side;
        bool operator<(const PqCand &o) const {
            if (prio != o.prio) return prio < o.prio;
            return u < o.u;
        }
    };

    std::vector<ExpNode> exp_nodes;
    exp_nodes.reserve(std::min(max_internals, std::max(16, leaf_cnt[root_r])));
    exp_nodes.push_back({root_r, -1, -1});

    auto is_eligible = [&](int32_t w) -> bool {
        if (w < N) return false;
        if (stop_at_comp_root && stop_at_comp_root[w]) return false;
        if (!node_has_active[w] && leaf_cnt[w] <= condense_thresh) return false;
        return true;
    };

    std::priority_queue<PqCand> pq;
    int32_t b0_root = c1_r[root_r], b1_root = c2_r[root_r];
    if (is_eligible(b0_root)) pq.push({leaf_cnt[b0_root], b0_root, 0, 0});
    if (is_eligible(b1_root)) pq.push({leaf_cnt[b1_root], b1_root, 0, 1});

    while (!pq.empty() && (int32_t)exp_nodes.size() < max_internals) {
        PqCand cand = pq.top();
        pq.pop();
        int32_t my_exp = (int32_t)exp_nodes.size();
        exp_nodes.push_back({cand.u, -1, -1});
        if (cand.side == 0) exp_nodes[cand.parent_exp].left_exp = my_exp;
        else                exp_nodes[cand.parent_exp].right_exp = my_exp;

        int32_t b0 = c1_r[cand.u], b1 = c2_r[cand.u];
        if (is_eligible(b0)) pq.push({leaf_cnt[b0], b0, my_exp, 0});
        if (is_eligible(b1)) pq.push({leaf_cnt[b1], b1, my_exp, 1});
    }

    struct Frame { int32_t exp_idx; int32_t state; int32_t left_id; };
    std::vector<Frame> st;
    st.reserve(256);
    st.push_back({0, 0, -1});
    int32_t last_ret = -1;

    while (!st.empty()) {
        Frame &f = st.back();
        const ExpNode &en = exp_nodes[f.exp_idx];
        if (f.state == 0) {
            f.state = 1;
            if (en.left_exp == -1) {
                int32_t sid = (int32_t)super_roots.size();
                super_roots.push_back(c1_r[en.u]);
                last_ret = sid;
            } else {
                st.push_back({en.left_exp, 0, -1});
            }
        } else if (f.state == 1) {
            f.left_id = last_ret;
            f.state = 2;
            if (en.right_exp == -1) {
                int32_t sid = (int32_t)super_roots.size();
                super_roots.push_back(c2_r[en.u]);
                last_ret = sid;
            } else {
                st.push_back({en.right_exp, 0, -1});
            }
        } else {
            int32_t right_id = last_ret;
            int32_t my_idx = (int32_t)split_internals.size();
            split_internals.push_back(en.u);
            skel_left.push_back(f.left_id);
            skel_right.push_back(right_id);
            last_ret = -1 - my_idx;
            st.pop_back();
        }
    }
    *out_root_skel_id = -1 - last_ret;
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

static void compute_outgroup_rep8(
    int32_t sub_root, int32_t root_r,
    const std::vector<int32_t> &curr_c1, const std::vector<int32_t> &curr_c2,
    const std::vector<int32_t> &parent_r, const std::vector<int32_t> &node_rep4,
    int32_t out_rep8[8]
) {
    int32_t p1 = parent_r[sub_root];
    int32_t sib1 = (curr_c1[p1] == sub_root) ? curr_c2[p1] : curr_c1[p1];
    int32_t sib2 = sib1, sib3 = sib1;
    if (p1 != root_r && parent_r[p1] != -1) {
        int32_t p2 = parent_r[p1];
        sib2 = (curr_c1[p2] == p1) ? curr_c2[p2] : curr_c1[p2];
        sib3 = sib2;
        if (p2 != root_r && parent_r[p2] != -1) {
            int32_t p3 = parent_r[p2];
            sib3 = (curr_c1[p3] == p2) ? curr_c2[p3] : curr_c1[p3];
        }
    }
    out_rep8[0] = node_rep4[8 * sib1 + 0];
    out_rep8[1] = node_rep4[8 * sib1 + 2];
    out_rep8[2] = node_rep4[8 * sib1 + 4];
    out_rep8[3] = node_rep4[8 * sib1 + 6];
    out_rep8[4] = node_rep4[8 * sib2 + 0];
    out_rep8[5] = node_rep4[8 * sib2 + 4];
    out_rep8[6] = node_rep4[8 * sib3 + 0];
    out_rep8[7] = node_rep4[8 * sib3 + 4];
}
