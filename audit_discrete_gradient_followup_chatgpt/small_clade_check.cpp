#include "host_helpers.hpp"
static void compute_tree_summaries_rep4(
    const int32_t *c1_r, const int32_t *c2_r, int32_t N,
    std::vector<int32_t> &ord,
    std::vector<int32_t> &curr_parent,
    std::vector<int32_t> &leaf_cnt,
    std::vector<int32_t> &node_rep0,
    std::vector<int32_t> &node_rep4
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    ord.resize(N - 1);
    curr_parent.assign(n_nodes_r, -1);
    leaf_cnt.assign(n_nodes_r, 1);
    node_rep0.resize(n_nodes_r);
    node_rep4.resize((size_t)n_nodes_r * 4);

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
        node_rep4[4 * x + 0] = x;
        node_rep4[4 * x + 1] = x;
        node_rep4[4 * x + 2] = x;
        node_rep4[4 * x + 3] = x;
    }
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        leaf_cnt[u] = leaf_cnt[b0] + leaf_cnt[b1];
        node_rep0[u] = node_rep0[b0];
        node_rep4[4 * u + 0] = node_rep4[4 * b0 + 0];
        node_rep4[4 * u + 1] = node_rep4[4 * b0 + 2];
        node_rep4[4 * u + 2] = node_rep4[4 * b1 + 0];
        node_rep4[4 * u + 3] = node_rep4[4 * b1 + 2];
    }
}
int main(){for(int N:{4,5}){std::vector<int32_t>a(2*N-1),b(2*N-1),ord,par,cnt,r4,r8,r0;a[N]=0;b[N]=1;for(int k=1;k<N-1;k++){a[N+k]=N+k-1;b[N+k]=k+1;}compute_tree_summaries_blind(a.data(),b.data(),N,ord,par,cnt,r8);compute_tree_summaries_rep4(a.data(),b.data(),N,ord,par,cnt,r0,r4);int K=N==4?4:8;auto &v=N==4?r4:r8;printf("N=%d K=%d reps",N,K);for(int k=0;k<K;k++)printf(" %d",v[K*(2*N-2)+k]);printf("\n");}return 0;}
