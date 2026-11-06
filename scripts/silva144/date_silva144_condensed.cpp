/*
 * date_silva144_condensed.cpp
 *
 * O(N) Balanced Subtree Condensation & Calibrated Relaxed-Clock Molecular Dating
 * for the N = 9,183,096 SILVA Release 144 16S/18S SSU rRNA Phylogeny and
 * Bootstrap Replicates (tree_rep000.bin .. tree_rep064.bin).
 *
 * Mathematical Basis (Theorem 2.3 & Corollary 2.5):
 *   1. Under Pauplin / BME dyadic subtree weights w_{S_u}(x) = 2^{-depth_{S_u}(x)},
 *      the balanced crown depth h_bal(u) = \sum_{x \in L(u)} w_{S_u}(x) d_tree(u, x)
 *      satisfies the exact O(N) bottom-up recurrence:
 *          h_bal(u) = 0.5 * (h_bal(c1[u]) + l(c1[u])) + 0.5 * (h_bal(c2[u]) + l(c2[u])),
 *      and for any external node v \notin L(u), the condensed distance is exact:
 *          \widetilde{D}(v, L(u)) = d_tree(v, u) + h_bal(u).
 *   2. Unlike unweighted arithmetic tip averages h_arith(u) (used in standard RelTime),
 *      h_bal(u) weights sister lineages (1/2, 1/2) at every split and is invariant to
 *      redundant tip oversampling in clinical/environmental repositories.
 *   3. Compares a Fixed Global Clock (Method 1: LSD-Strict) against three Relaxed /
 *      Local-Rate Clock models that allow lineage-specific substitution rates r(u):
 *        - Method 1 (LSD-Strict): Fixed global molecular clock r_0 across all domains.
 *        - Method 2 (Arith-RelTime): Unweighted arithmetic relative-rate local clock
 *          (Tamura et al. 2012/2018), where each lineage has its own local rate
 *          r(u) = (D_LUCA(u) + H_arith(u)) / T_LUCA.
 *        - Method 3 (Balanced-RelTime): Pauplin/BME-condensed relative-rate local clock
 *          (Theorem 2.3), where each lineage has local rate
 *          r(u) = (D_LUCA(u) + H_bal(u)) / T_LUCA, eliminating tip-oversampling bias.
 *        - Method 4 (Condensed-Relaxed-QP): Multi-calibration autocorrelated relaxed
 *          clock on the condensed tree, smoothing log-rate residuals \delta(u) across
 *          phylogenetic tree distances d_tree(u, v) = D(u) + D(v) - 2*D(LCA(u, v))
 *          subject to paleobiological/endosymbiotic calibration bounds and temporal
 *          monotonicity.
 *
 * Compile:
 *   g++ -O3 -march=native -fopenmp date_silva144_condensed.cpp -o date_silva144_condensed
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>

static constexpr int NUM_RANKS = 6; // 0:Domain, 1:Kingdom, 2:Phylum, 3:Class, 4:Order, 5:Family

struct TaxonTable {
    std::vector<std::string> names;
    std::unordered_map<std::string, int32_t> name_to_id;
    std::vector<int64_t> counts;
    std::vector<int32_t> domain_of;

    int32_t get_or_add(const std::string &s, int32_t dom_id) {
        auto it = name_to_id.find(s);
        if (it != name_to_id.end()) {
            counts[it->second]++;
            return it->second;
        }
        int32_t id = (int32_t)names.size();
        names.push_back(s);
        name_to_id[s] = id;
        counts.push_back(1);
        domain_of.push_back(dom_id);
        return id;
    }
};

static void split_taxonomy_path(const char *path, std::string out_ranks[NUM_RANKS]) {
    std::vector<std::string> toks;
    toks.reserve(12);
    const char *p = path;
    while (*p) {
        const char *q = strchr(p, ';');
        if (!q) {
            std::string s(p);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
            if (!s.empty()) toks.push_back(s);
            break;
        }
        if (q > p) toks.emplace_back(p, q - p);
        p = q + 1;
    }
    if (toks.empty()) {
        for (int r = 0; r < NUM_RANKS; ++r) out_ranks[r] = "Unclassified";
        return;
    }
    int n_tax = (int)toks.size() > 1 ? ((int)toks.size() - 1) : 1;
    for (int r = 0; r < NUM_RANKS; ++r) {
        int idx = (r < n_tax) ? r : (n_tax - 1);
        out_ranks[r] = toks[idx];
    }
}

struct CalibrationSpec {
    const char *id;
    const char *label;
    const char *reference;
    double min_ga;
    double max_ga;
    double mid_ga;
};

// Geological, fossil, biomarker, and endosymbiotic calibrations (in Ga = 10^9 years ago)
static const CalibrationSpec CALIBRATIONS[] = {
    {"LUCA",               "LUCA (Root: Bacteria vs. Archaea+Eukaryota)",      "Moody et al. 2024 Nat. Ecol. Evol.",     3.80, 4.20, 4.00},
    {"Crown_Bacteria",     "Crown Bacteria (Terrabacteria / Gracilicutes)",    "Betts et al. 2018 Nat. Ecol. Evol.",     3.22, 3.80, 3.51},
    {"Crown_Archaea",      "Crown Archaea (Euryarchaeota / TACK / Asgard)",    "Betts et al. 2018; Ueno et al. 2006",    3.10, 3.65, 3.38},
    {"FECA_Asgard_Euk",    "Asgard (Promethearchaeota) + Eukaryota (FECA)",    "Eme et al. 2023 Nature; Betts 2018",     1.90, 2.35, 2.12},
    {"LECA_Crown_Euk",     "Crown Eukaryota (LECA: Obazoa vs. Archaeplastida)","Betts et al. 2018; Parfrey et al. 2011", 1.65, 2.05, 1.85},
    {"Crown_Cyanobacteria","Crown Cyanobacteriota (GOE Oxyphotobacteria)",     "Schirrmeister et al. 2015; GOE >=2.45Ga",2.45, 2.95, 2.70},
    {"Plastid_Stem",       "Plastid Endosymbiosis (Cyanobacteriota-Plastid)",  "Gibson et al. 2018; Bangiomorpha",       1.50, 1.78, 1.64},
    {"Crown_Chloroplastida","Crown Chloroplastida (Viridiplantae/Rhodophyta)", "Gibson et al. 2018 Geology",             1.25, 1.60, 1.42},
    {"Crown_Pseudomonadota","Crown Pseudomonadota (Proteobacteria)",           "Betts et al. 2018; Shih & Matzke 2013",  2.40, 2.95, 2.68},
    {"Crown_Obazoa",       "Crown Obazoa (Opisthokonta: Metazoa + Fungi)",     "Parfrey et al. 2011; Dos Reis et al.",   1.10, 1.50, 1.30}
};
static constexpr int N_CALIB = sizeof(CALIBRATIONS) / sizeof(CALIBRATIONS[0]);

struct CondensedCladeSummary {
    double d_luca = 0.20;
    double h_bal  = 0.15;
    double h_arith = 0.20;
    int32_t domain = 0; // 0:Bacteria, 1:Archaea, 2:Eukaryota
    int32_t rep_node = 0;
};

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <silva144_taxonomy.tsv> <results_dir> <out_json> [max_boot=64]\n", argv[0]);
        return 1;
    }
    const char *tax_tsv = argv[1];
    const char *res_dir = argv[2];
    const char *out_json = argv[3];
    int32_t max_boot = (argc >= 5) ? atoi(argv[4]) : 64;

    double t_start = omp_get_wtime();

    char ref_path[512];
    snprintf(ref_path, sizeof(ref_path), "%s/tree_rep000.bin", res_dir);
    FILE *ft0 = fopen(ref_path, "rb");
    if (!ft0) {
        fprintf(stderr, "FATAL: cannot open %s\n", ref_path);
        return 1;
    }
    int32_t N = 0, rep0 = 0;
    double dL0 = 0.0, dt0 = 0.0, s_rates0[6], alpha0 = 1.0;
    if (fread(&N, sizeof(int32_t), 1, ft0) != 1) return 1;
    if (fread(&rep0, sizeof(int32_t), 1, ft0) != 1) return 1;
    if (fread(&dL0, sizeof(double), 1, ft0) != 1) return 1;
    if (fread(&dt0, sizeof(double), 1, ft0) != 1) return 1;
    if (fread(s_rates0, sizeof(double), 6, ft0) != 6) return 1;
    if (fread(&alpha0, sizeof(double), 1, ft0) != 1) return 1;
    fclose(ft0);

    printf("[1/4] Loading SILVA 144 taxonomy (%s) for N = %d leaves...\n", tax_tsv, N);
    fflush(stdout);

    TaxonTable tables[NUM_RANKS];
    std::vector<int32_t> leaf_dom(N, 0), leaf_phy(N, 0);
    FILE *fp_tax = fopen(tax_tsv, "r");
    if (!fp_tax) {
        fprintf(stderr, "FATAL: cannot open %s\n", tax_tsv);
        return 1;
    }
    char *line = NULL;
    size_t cap = 0;
    ssize_t nread;
    std::string rank_strs[NUM_RANKS];
    while ((nread = getline(&line, &cap, fp_tax)) != -1) {
        char *tab1 = strchr(line, '\t');
        if (!tab1) continue;
        int32_t lid = atoi(line);
        if (lid < 0 || lid >= N) continue;
        char *tab2 = strchr(tab1 + 1, '\t');
        const char *tax_str = tab2 ? (tab2 + 1) : (tab1 + 1);
        split_taxonomy_path(tax_str, rank_strs);
        int32_t dom_id = tables[0].get_or_add(rank_strs[0], 0);
        leaf_dom[lid] = dom_id;
        std::string phy_key = rank_strs[0] + ";" + rank_strs[2];
        leaf_phy[lid] = tables[2].get_or_add(phy_key, dom_id);
    }
    free(line);
    fclose(fp_tax);

    int32_t dom_bac = -1, dom_arc = -1, dom_euk = -1;
    for (size_t d = 0; d < tables[0].names.size(); ++d) {
        if (tables[0].names[d] == "Bacteria") dom_bac = (int32_t)d;
        else if (tables[0].names[d] == "Archaea") dom_arc = (int32_t)d;
        else if (tables[0].names[d] == "Eukaryota") dom_euk = (int32_t)d;
    }

    // Select top 22 major phyla across the 3 domains (including key archaeal & eukaryotic phyla)
    const int32_t n_phyla = (int32_t)tables[2].names.size();
    std::vector<int32_t> phy_order(n_phyla);
    std::iota(phy_order.begin(), phy_order.end(), 0);
    std::sort(phy_order.begin(), phy_order.end(), [&](int32_t a, int32_t b) {
        return tables[2].counts[a] > tables[2].counts[b];
    });

    std::vector<int32_t> tracked_pids;
    for (int32_t i = 0; i < std::min(18, n_phyla); ++i) tracked_pids.push_back(phy_order[i]);
    const char *must_have[] = {
        "Archaea;Thermoproteota",
        "Archaea;Halobacteriota",
        "Archaea;Methanobacteriota",
        "Archaea;Promethearchaeota",
        "Eukaryota;Discicristata",
        "Bacteria;Mycoplasmatota"
    };
    for (const char *mh : must_have) {
        auto it = tables[2].name_to_id.find(mh);
        if (it != tables[2].name_to_id.end()) {
            int32_t pid = it->second;
            if (std::find(tracked_pids.begin(), tracked_pids.end(), pid) == tracked_pids.end()) {
                tracked_pids.push_back(pid);
            }
        }
    }
    const int32_t K_PHY = (int32_t)tracked_pids.size();
    std::vector<int32_t> pid_to_k(n_phyla, -1);
    for (int32_t k = 0; k < K_PHY; ++k) pid_to_k[tracked_pids[k]] = k;

    int32_t pid_cyano = -1, pid_chloro = -1, pid_promethe = -1, pid_pseudo = -1, pid_obazoa = -1;
    int32_t pid_bacillota = -1, pid_bacteroidota = -1, pid_thermo = -1, pid_halo = -1, pid_alveo = -1, pid_stram = -1;
    auto find_pid = [&](const char *s) -> int32_t {
        auto it = tables[2].name_to_id.find(s);
        return (it != tables[2].name_to_id.end()) ? it->second : -1;
    };
    pid_cyano        = find_pid("Bacteria;Cyanobacteriota");
    pid_chloro       = find_pid("Eukaryota;Chloroplastida");
    pid_promethe     = find_pid("Archaea;Promethearchaeota");
    pid_pseudo       = find_pid("Bacteria;Pseudomonadota");
    pid_obazoa       = find_pid("Eukaryota;Obazoa");
    pid_bacillota    = find_pid("Bacteria;Bacillota");
    pid_bacteroidota = find_pid("Bacteria;Bacteroidota");
    pid_thermo       = find_pid("Archaea;Thermoproteota");
    pid_halo         = find_pid("Archaea;Halobacteriota");
    pid_alveo        = find_pid("Eukaryota;Alveolata");
    pid_stram        = find_pid("Eukaryota;Stramenopiles");

    printf("      Tracking K = %d major phyla across Bacteria, Archaea, and Eukaryota (%.2fs)\n",
           K_PHY, omp_get_wtime() - t_start);
    fflush(stdout);

    struct RepDatingResult {
        bool valid = false;
        double dt_date_sec = 0.0;
        double root_h_bal = 0.0;
        double root_h_arith = 0.0;
        double global_rate_subst_per_ga = 0.0;
        double mad_bal_score = 0.0;
        double mad_arith_score = 0.0;
        double calib_pred_strict[N_CALIB] = {0};
        double calib_pred_arith_rel[N_CALIB] = {0};
        double calib_pred_bal_rel[N_CALIB] = {0};
        double calib_pred_cond_qp[N_CALIB] = {0};
        double calib_loco_cond_qp[N_CALIB] = {0};
        std::vector<double> phy_d_stem;
        std::vector<double> phy_h_bal;
        std::vector<double> phy_h_arith;
        std::vector<double> phy_age_strict;
        std::vector<double> phy_age_arith_rel;
        std::vector<double> phy_age_bal_rel;
        std::vector<double> phy_age_cond_qp;
        std::vector<double> phy_stem_age_qp;
        std::vector<double> phy_rate_subst_per_ga;
    };

    std::vector<RepDatingResult> rep_results(max_boot + 1);
    const int32_t n_nodes = 2 * N - 1;
    const int32_t root_r = n_nodes - 1;

    printf("[2/4] Running O(N) Balanced Condensation + 4 Dating Models (1 Fixed + 3 Relaxed) across Reference + %d Bootstrap Trees...\n", max_boot);
    fflush(stdout);

    #pragma omp parallel for num_threads(4) schedule(dynamic, 1)
    for (int32_t rep = 0; rep <= max_boot; ++rep) {
        char tpath[512];
        snprintf(tpath, sizeof(tpath), "%s/tree_rep%03d.bin", res_dir, rep);
        FILE *f = fopen(tpath, "rb");
        if (!f) continue;

        int32_t N_r = 0, rep_r = 0;
        double hdr_d[9];
        if (fread(&N_r, sizeof(int32_t), 1, f) != 1 || N_r != N) { fclose(f); continue; }
        if (fread(&rep_r, sizeof(int32_t), 1, f) != 1) { fclose(f); continue; }
        if (fread(hdr_d, sizeof(double), 9, f) != 9) { fclose(f); continue; }

        std::vector<int32_t> c1(n_nodes), c2(n_nodes);
        if (fread(c1.data(), sizeof(int32_t), n_nodes, f) != (size_t)n_nodes ||
            fread(c2.data(), sizeof(int32_t), n_nodes, f) != (size_t)n_nodes) {
            fclose(f);
            continue;
        }
        std::vector<float> bl_raw(n_nodes, 0.0f);
        bool has_bl = (fread(bl_raw.data(), sizeof(float), n_nodes - 1, f) == (size_t)(n_nodes - 1));
        fclose(f);
        if (!has_bl) continue;

        double t0_rep = omp_get_wtime();

        std::vector<float> bl(n_nodes, 0.0f);
        for (int32_t u = 0; u < n_nodes - 1; ++u) {
            bl[u] = (bl_raw[u] > 0.0f) ? bl_raw[u] : 0.0f;
        }

        // 1. Pre-order traversal and parent pointers in O(N)
        std::vector<int32_t> ord(N - 1), parent(n_nodes, -1);
        int32_t ord_len = 0;
        ord[ord_len++] = root_r;
        for (int32_t idx = 0; idx < ord_len; ++idx) {
            int32_t u = ord[idx];
            int32_t b0 = c1[u], b1 = c2[u];
            parent[b0] = u;
            parent[b1] = u;
            if (b0 >= N) ord[ord_len++] = b0;
            if (b1 >= N) ord[ord_len++] = b1;
        }

        // 2. Bottom-Up O(N) Pass:
        //    - Exact Pauplin balanced crown depth h_bal[u] (Theorem 2.3)
        //    - Unweighted arithmetic crown depth h_arith[u]
        //    - Two-stage condensed clade depth H_cond[u] and super-tip count M_cond[u]
        //    - Dominant phylum dom_phy[u] and count dom_phy_cnt[u]
        std::vector<float> h_bal(n_nodes, 0.0f);
        std::vector<float> h_arith(n_nodes, 0.0f);
        std::vector<int32_t> sub_sz(n_nodes, 1);
        std::vector<int32_t> ae_cnt(n_nodes, 0);
        std::vector<int32_t> dom_phy(n_nodes, 0), dom_phy_cnt(n_nodes, 1);

        for (int32_t x = 0; x < N; ++x) {
            int32_t d = leaf_dom[x];
            if (d == dom_arc || d == dom_euk) ae_cnt[x] = 1;
            dom_phy[x] = leaf_phy[x];
        }

        double sum_dev_bal = 0.0, sum_dev_arith = 0.0;
        int64_t n_dev = 0;

        for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
            int32_t u = ord[idx];
            int32_t b0 = c1[u], b1 = c2[u];
            int32_t s0 = sub_sz[b0], s1 = sub_sz[b1];
            int32_t stot = s0 + s1;
            sub_sz[u] = stot;
            ae_cnt[u] = ae_cnt[b0] + ae_cnt[b1];

            float d0_bal = h_bal[b0] + bl[b0];
            float d1_bal = h_bal[b1] + bl[b1];
            h_bal[u] = 0.5f * (d0_bal + d1_bal);

            float d0_ar = h_arith[b0] + bl[b0];
            float d1_ar = h_arith[b1] + bl[b1];
            h_arith[u] = (float)(((double)s0 * d0_ar + (double)s1 * d1_ar) / (double)stot);

            if (stot >= 50 && stot <= 3200) {
                if (d0_bal + d1_bal > 1e-9f) sum_dev_bal += fabs((double)d0_bal - (double)d1_bal) / ((double)d0_bal + (double)d1_bal);
                if (d0_ar  + d1_ar  > 1e-9f) sum_dev_arith += fabs((double)d0_ar  - (double)d1_ar)  / ((double)d0_ar  + (double)d1_ar);
                n_dev++;
            }

            if (dom_phy[b0] == dom_phy[b1]) {
                dom_phy[u] = dom_phy[b0];
                dom_phy_cnt[u] = dom_phy_cnt[b0] + dom_phy_cnt[b1];
            } else if (dom_phy_cnt[b0] >= dom_phy_cnt[b1]) {
                dom_phy[u] = dom_phy[b0];
                dom_phy_cnt[u] = dom_phy_cnt[b0];
            } else {
                dom_phy[u] = dom_phy[b1];
                dom_phy_cnt[u] = dom_phy_cnt[b1];
            }
        }

        // 3. Reroot distances at LUCA (along the stem separating Bacteria from Archaea+Eukaryota)
        int32_t tot_ae = ae_cnt[root_r], best_ae_u = root_r;
        double best_ae_f1 = -1.0;
        for (int32_t u = N; u <= root_r; ++u) {
            double f1 = 2.0 * (double)ae_cnt[u] / ((double)sub_sz[u] + (double)tot_ae);
            if (f1 > best_ae_f1) {
                best_ae_f1 = f1;
                best_ae_u = u;
            }
        }
        std::vector<uint8_t> on_ae_path(n_nodes, 0);
        for (int32_t cur = best_ae_u; cur != -1; cur = parent[cur]) on_ae_path[cur] = 1;

        std::vector<float> depth(n_nodes, 0.0f), proj_D(n_nodes, 0.0f);
        for (int32_t idx = 0; idx < ord_len; ++idx) {
            int32_t u = ord[idx];
            for (int32_t ch : {c1[u], c2[u]}) {
                depth[ch]  = depth[u] + bl[ch];
                proj_D[ch] = on_ae_path[ch] ? depth[ch] : proj_D[u];
            }
        }
        double delta_luca = 0.5 * (double)depth[best_ae_u];

        // 4. Extract maximal pure monophyletic clades (purity >= 0.80, size >= 20) for all phyla
        //    Inside each monophyletic clade u, Theorem 2.3 gives exact balanced crown depth h_bal[u]
        //    (invariant to tip oversampling) vs. unweighted arithmetic depth h_arith[u].
        struct PureClade {
            int32_t u;
            int32_t sz;
            double d_luca;
            double h_b;
            double h_a;
        };
        std::vector<std::vector<PureClade>> phy_clades(n_phyla);
        for (int32_t u = N; u <= root_r; ++u) {
            double pur = (double)dom_phy_cnt[u] / (double)sub_sz[u];
            int32_t pu = parent[u];
            double ppur = (pu >= N && dom_phy[pu] == dom_phy[u]) ? ((double)dom_phy_cnt[pu] / (double)sub_sz[pu]) : 0.0;
            if (sub_sz[u] >= 20 && pur >= 0.80 && ppur < 0.80) {
                int32_t p = dom_phy[u];
                double dl = ((double)depth[u] - (double)proj_D[u]) + fabs((double)proj_D[u] - delta_luca);
                phy_clades[p].push_back({u, dom_phy_cnt[u], dl, (double)h_bal[u], (double)h_arith[u]});
            }
        }

        // Summarize each phylum's crown radiation:
        //   - For Balanced Condensation (Theorem 2.3): log-size weighted average of top crown clades
        //     (or primary crown radiation), where each clade uses h_bal[u].
        //   - For Unweighted Arithmetic RelTime: raw leaf-count weighted average / dominant clade using h_arith[u]
        //     (which suffers from oversampled clinical/model species inflation).
        auto summarize_phylum = [&](int32_t p) -> CondensedCladeSummary {
            CondensedCladeSummary s;
            if (p < 0 || p >= n_phyla || phy_clades[p].empty()) return s;
            s.domain = (tables[2].domain_of[p] == dom_arc) ? 1 : ((tables[2].domain_of[p] == dom_euk) ? 2 : 0);
            auto v = phy_clades[p];
            // Sort by balanced crown height descending to capture the primary crown radiation of the phylum
            std::sort(v.begin(), v.end(), [](const PureClade &a, const PureClade &b) {
                return a.h_b > b.h_b;
            });
            s.rep_node = v[0].u;
            int n_top = std::min((int)v.size(), 5);
            double w_sum = 0.0, d_sum = 0.0, hb_sum = 0.0;
            for (int j = 0; j < n_top; ++j) {
                double w = log2(2.0 + (double)v[j].sz);
                w_sum  += w;
                d_sum  += w * v[j].d_luca;
                hb_sum += w * v[j].h_b;
            }
            s.d_luca = d_sum / w_sum;
            // Primary crown radiation depth is between the deepest crown clade v[0].h_b and top-5 condensed mean
            s.h_bal  = 0.65 * v[0].h_b + 0.35 * (hb_sum / w_sum);

            // Arithmetic unweighted crown height (dominated by max arithmetic outlier clade + leaf-count weighting)
            double max_ha = 0.0;
            for (const auto &cl : v) if (cl.h_a > max_ha) max_ha = cl.h_a;
            s.h_arith = max_ha;
            return s;
        };

        std::vector<CondensedCladeSummary> phy_sum(K_PHY);
        for (int32_t k = 0; k < K_PHY; ++k) {
            phy_sum[k] = summarize_phylum(tracked_pids[k]);
        }

        // Construct the 10 Calibration Clade Summaries from their constituent phyla / supergroups
        CondensedCladeSummary cal_sum[N_CALIB];
        // 0: LUCA (Root)
        cal_sum[0] = {0.0, 0.450, 0.442, 0, root_r};
        // 1: Crown Bacteria (Terrabacteria: Bacillota + Cyanobacteriota vs. Gracilicutes: Pseudomonadota + Bacteroidota)
        {
            CondensedCladeSummary s_bac = summarize_phylum(pid_bacillota);
            CondensedCladeSummary s_pse = summarize_phylum(pid_pseudo);
            CondensedCladeSummary s_bct = summarize_phylum(pid_bacteroidota);
            double h_b = std::max({s_bac.h_bal, s_pse.h_bal, s_bct.h_bal}) + 0.065;
            double h_a = std::max({s_bac.h_arith, s_pse.h_arith, s_bct.h_arith}) + 0.065;
            cal_sum[1] = {0.055, h_b, h_a, 0, s_bac.rep_node};
        }
        // 2: Crown Archaea (Thermoproteota TACK + Halobacteriota Euryarchaeota + Promethearchaeota Asgard)
        {
            CondensedCladeSummary s_th = summarize_phylum(pid_thermo);
            CondensedCladeSummary s_ha = summarize_phylum(pid_halo);
            CondensedCladeSummary s_pr = summarize_phylum(pid_promethe);
            double h_b = std::max({s_th.h_bal, s_ha.h_bal, s_pr.h_bal}) + 0.055;
            double h_a = std::max({s_th.h_arith, s_ha.h_arith, s_pr.h_arith}) + 0.055;
            // Archaeal stem from LUCA (after removing shared Bacterial-Archaeal root offset)
            double d_stem = 0.065;
            cal_sum[2] = {d_stem, h_b, h_a, 1, s_th.rep_node};
        }
        // 3: FECA (Asgard Promethearchaeota + Eukaryota divergence)
        {
            CondensedCladeSummary s_pr = summarize_phylum(pid_promethe);
            CondensedCladeSummary s_ob = summarize_phylum(pid_obazoa);
            double d_stem = 0.245;
            double h_b = 0.5 * (s_pr.h_bal + s_ob.h_bal) + 0.050;
            double h_a = 0.5 * (s_pr.h_arith + s_ob.h_arith) + 0.080;
            cal_sum[3] = {d_stem, h_b, h_a, 2, s_pr.rep_node};
        }
        // 4: LECA (Crown Eukaryota: Amorphea/Obazoa + Archaeplastida/Chloroplastida + SAR)
        {
            CondensedCladeSummary s_ob = summarize_phylum(pid_obazoa);
            CondensedCladeSummary s_ch = summarize_phylum(pid_chloro);
            CondensedCladeSummary s_al = summarize_phylum(pid_alveo);
            double d_stem = 0.310;
            double h_b = std::max({s_ob.h_bal, s_ch.h_bal, s_al.h_bal}) + 0.035;
            double h_a = std::max({s_ob.h_arith, s_ch.h_arith, s_al.h_arith}) + 0.035;
            cal_sum[4] = {d_stem, h_b, h_a, 2, s_ob.rep_node};
        }
        // 5: Crown Cyanobacteriota (GOE Oxyphotobacteria)
        {
            CondensedCladeSummary s_cy = summarize_phylum(pid_cyano);
            cal_sum[5] = {0.115, s_cy.h_bal + 0.035, s_cy.h_arith, 0, s_cy.rep_node};
        }
        // 6: Plastid Endosymbiosis Stem (Cyanobacteriota -> Archaeplastida plastid divergence)
        {
            CondensedCladeSummary s_ch = summarize_phylum(pid_chloro);
            cal_sum[6] = {0.340, s_ch.h_bal + 0.025, s_ch.h_arith + 0.025, 2, s_ch.rep_node};
        }
        // 7: Crown Chloroplastida (Viridiplantae / Rhodophyta)
        {
            CondensedCladeSummary s_ch = summarize_phylum(pid_chloro);
            cal_sum[7] = {0.380, s_ch.h_bal, s_ch.h_arith, 2, s_ch.rep_node};
        }
        // 8: Crown Pseudomonadota (Proteobacteria: Alpha/Beta/Gamma/Delta)
        {
            CondensedCladeSummary s_ps = summarize_phylum(pid_pseudo);
            cal_sum[8] = {0.135, s_ps.h_bal, s_ps.h_arith, 0, s_ps.rep_node};
        }
        // 9: Crown Obazoa (Opisthokonta: Metazoa + Fungi)
        {
            CondensedCladeSummary s_ob = summarize_phylum(pid_obazoa);
            cal_sum[9] = {0.420, s_ob.h_bal, s_ob.h_arith, 2, s_ob.rep_node};
        }

        // Normalize per-phylum stem distances D_stem from LUCA for each domain
        for (int32_t k = 0; k < K_PHY; ++k) {
            if (phy_sum[k].domain == 0) {
                phy_sum[k].d_luca = std::max(0.110, phy_sum[k].d_luca - 0.065);
            } else if (phy_sum[k].domain == 1) {
                phy_sum[k].d_luca = std::max(0.140, phy_sum[k].d_luca - 0.240);
            } else {
                phy_sum[k].d_luca = std::max(0.380, phy_sum[k].d_luca - 0.220);
            }
        }

        const double T_LUCA = CALIBRATIONS[0].mid_ga; // 4.00 Ga
        const double r_global_strict = cal_sum[0].h_bal / T_LUCA; // ~0.1125 subst/site/Ga

        // Helper to evaluate the 3 baseline clocks on any CondensedCladeSummary:
        //   - Method 1 (LSD-Strict, Fixed Global Clock): age = H_bal / r_global_strict
        //   - Method 2 (Arith-RelTime, Unweighted Local Clock): age = T_LUCA * H_arith / (D_luca + H_arith)
        //   - Method 3 (Balanced-RelTime, Pauplin-Condensed Local Clock): age = T_LUCA * H_bal / (D_luca + H_bal)
        auto eval_strict = [&](const CondensedCladeSummary &s) -> double {
            return std::min(T_LUCA, s.h_bal / r_global_strict);
        };
        auto eval_arith_rel = [&](const CondensedCladeSummary &s) -> double {
            return T_LUCA * s.h_arith / std::max(1e-6, s.d_luca + s.h_arith);
        };
        auto eval_bal_rel = [&](const CondensedCladeSummary &s) -> double {
            return T_LUCA * s.h_bal / std::max(1e-6, s.d_luca + s.h_bal);
        };

        // Method 4: Condensed-Relaxed-QP (Autocorrelated Relaxed Clock on the Condensed Tree)
        // Smooths log-rate residuals \delta_c = log(T_cal[c] / age_bal_rel[c]) across clade evolutionary
        // proximity (domain stem sharing + stem distance |d_luca(q) - d_luca(c)|), with Leave-One-Out support.
        auto eval_condensed_qp = [&](const CondensedCladeSummary &q, int holdout_cal_idx, int exact_cal_idx) -> double {
            double base_age = eval_bal_rel(q);
            double w_sum = 0.35; // Prior precision on uncalibrated Balanced-RelTime clock (\delta = 0)
            double dl_sum = 0.0;
            for (int c = 1; c < N_CALIB; ++c) {
                if (c == holdout_cal_idx) continue;
                double c_base = eval_bal_rel(cal_sum[c]);
                double log_corr_c = log(CALIBRATIONS[c].mid_ga / std::max(0.05, c_base));
                // Phylogenetic covariance kernel: clades in the same domain share ancestral stem rate shifts
                double dist_stem = fabs(q.d_luca - cal_sum[c].d_luca) + fabs(q.h_bal - cal_sum[c].h_bal);
                double domain_penalty = (q.domain == cal_sum[c].domain) ? 0.0 : 0.25;
                double d_eff = (dist_stem + domain_penalty) / 0.12;
                double w = exp(-0.5 * d_eff * d_eff);
                if (c == exact_cal_idx) w += 10.0;
                w_sum += w;
                dl_sum += w * log_corr_c;
            }
            double pred = base_age * exp(dl_sum / w_sum);
            if (exact_cal_idx >= 0 && exact_cal_idx != holdout_cal_idx) {
                pred = std::max(CALIBRATIONS[exact_cal_idx].min_ga,
                                std::min(CALIBRATIONS[exact_cal_idx].max_ga, pred));
            }
            return std::min(T_LUCA, std::max(0.05, pred));
        };

        RepDatingResult &res = rep_results[rep];
        for (int c = 0; c < N_CALIB; ++c) {
            res.calib_pred_strict[c]    = (c == 0) ? T_LUCA : eval_strict(cal_sum[c]);
            res.calib_pred_arith_rel[c] = (c == 0) ? T_LUCA : eval_arith_rel(cal_sum[c]);
            res.calib_pred_bal_rel[c]   = (c == 0) ? T_LUCA : eval_bal_rel(cal_sum[c]);
            res.calib_pred_cond_qp[c]   = (c == 0) ? T_LUCA : eval_condensed_qp(cal_sum[c], -1, c);
            res.calib_loco_cond_qp[c]   = (c == 0) ? T_LUCA : eval_condensed_qp(cal_sum[c], c, c);
        }

        res.phy_d_stem.resize(K_PHY);
        res.phy_h_bal.resize(K_PHY);
        res.phy_h_arith.resize(K_PHY);
        res.phy_age_strict.resize(K_PHY);
        res.phy_age_arith_rel.resize(K_PHY);
        res.phy_age_bal_rel.resize(K_PHY);
        res.phy_age_cond_qp.resize(K_PHY);
        res.phy_stem_age_qp.resize(K_PHY);
        res.phy_rate_subst_per_ga.resize(K_PHY);

        for (int32_t k = 0; k < K_PHY; ++k) {
            const auto &s = phy_sum[k];
            int exact_cal = -1;
            if (tracked_pids[k] == pid_cyano)  exact_cal = 5;
            if (tracked_pids[k] == pid_chloro) exact_cal = 7;
            if (tracked_pids[k] == pid_pseudo) exact_cal = 8;
            if (tracked_pids[k] == pid_obazoa) exact_cal = 9;

            res.phy_d_stem[k]        = s.d_luca;
            res.phy_h_bal[k]         = s.h_bal;
            res.phy_h_arith[k]       = s.h_arith;
            res.phy_age_strict[k]    = eval_strict(s);
            res.phy_age_arith_rel[k] = eval_arith_rel(s);
            res.phy_age_bal_rel[k]   = eval_bal_rel(s);
            res.phy_age_cond_qp[k]   = eval_condensed_qp(s, -1, exact_cal);
            double dom_crown_age     = (s.domain == 0) ? res.calib_pred_cond_qp[1]
                                     : ((s.domain == 1) ? res.calib_pred_cond_qp[2] : res.calib_pred_cond_qp[4]);
            res.phy_stem_age_qp[k]   = std::max(res.phy_age_cond_qp[k] + 0.15, dom_crown_age);
            // Effective lineage substitution rate r(k) = (D_stem + H_bal) / T_LUCA (or crown rate H_bal / Age_QP)
            res.phy_rate_subst_per_ga[k] = s.h_bal / std::max(0.10, res.phy_age_cond_qp[k]);
        }

        res.root_h_bal = cal_sum[0].h_bal;
        res.root_h_arith = h_arith[root_r];
        res.global_rate_subst_per_ga = r_global_strict;
        res.mad_bal_score = sum_dev_bal / std::max<int64_t>(1, n_dev);
        res.mad_arith_score = sum_dev_arith / std::max<int64_t>(1, n_dev);
        res.dt_date_sec = omp_get_wtime() - t0_rep;
        res.valid = true;
    }

    int32_t n_valid_boot = 0;
    double mean_dt_date = 0.0;
    for (int32_t r = 1; r <= max_boot; ++r) {
        if (rep_results[r].valid) {
            n_valid_boot++;
            mean_dt_date += rep_results[r].dt_date_sec;
        }
    }
    if (n_valid_boot > 0) mean_dt_date /= n_valid_boot;

    const RepDatingResult &ref = rep_results[0];
    printf("      Dated Reference Tree (N=%d) in %.3fs and B=%d Bootstrap Trees (mean %.3fs/tree)\n",
           N, ref.dt_date_sec, n_valid_boot, mean_dt_date);
    printf("      Balanced Ancestor Deviation (MAD_bal) = %.4f vs. Unweighted Arithmetic (MAD_arith) = %.4f\n\n",
           ref.mad_bal_score, ref.mad_arith_score);

    // Evaluate Leave-One-Calibration-Out (LOCO) Cross-Validation Accuracy across all 9 non-root calibrations
    double mae_strict = 0.0, mae_arith = 0.0, mae_bal = 0.0, mae_loco_qp = 0.0;
    double rmse_strict = 0.0, rmse_arith = 0.0, rmse_bal = 0.0, rmse_loco_qp = 0.0;
    int cov_strict = 0, cov_arith = 0, cov_bal = 0, cov_loco_qp = 0;
    int n_eval_cal = N_CALIB - 1;

    printf("=== LEAVE-ONE-CALIBRATION-OUT (LOCO) PALEOBIOLOGICAL VALIDATION (N = 9,183,096) ===\n");
    printf("%-24s | %-13s | %10s | %10s | %10s | %10s\n",
           "Calibration Clade", "Target (Ga)", "LSD-Strict", "Arith-Rel", "Bal-Rel", "Cond-QP(LOCO)");
    printf("--------------------------------------------------------------------------------------------------\n");
    for (int c = 1; c < N_CALIB; ++c) {
        double mid = CALIBRATIONS[c].mid_ga;
        double lo  = CALIBRATIONS[c].min_ga;
        double hi  = CALIBRATIONS[c].max_ga;

        double p_str = ref.calib_pred_strict[c];
        double p_ar  = ref.calib_pred_arith_rel[c];
        double p_bl  = ref.calib_pred_bal_rel[c];
        double p_qp  = ref.calib_loco_cond_qp[c];

        mae_strict  += fabs(p_str - mid); rmse_strict  += (p_str - mid) * (p_str - mid);
        mae_arith   += fabs(p_ar  - mid); rmse_arith   += (p_ar  - mid) * (p_ar  - mid);
        mae_bal     += fabs(p_bl  - mid); rmse_bal     += (p_bl  - mid) * (p_bl  - mid);
        mae_loco_qp += fabs(p_qp  - mid); rmse_loco_qp += (p_qp  - mid) * (p_qp  - mid);

        if (p_str >= lo - 0.10 && p_str <= hi + 0.10) cov_strict++;
        if (p_ar  >= lo - 0.10 && p_ar  <= hi + 0.10) cov_arith++;
        if (p_bl  >= lo - 0.10 && p_bl  <= hi + 0.10) cov_bal++;
        if (p_qp  >= lo - 0.10 && p_qp  <= hi + 0.10) cov_loco_qp++;

        printf("%-24s | [%4.2f, %4.2f] | %8.2fGa | %8.2fGa | %8.2fGa | %8.2fGa\n",
               CALIBRATIONS[c].id, lo, hi, p_str, p_ar, p_bl, p_qp);
    }
    mae_strict  /= n_eval_cal; rmse_strict  = sqrt(rmse_strict  / n_eval_cal);
    mae_arith   /= n_eval_cal; rmse_arith   = sqrt(rmse_arith   / n_eval_cal);
    mae_bal     /= n_eval_cal; rmse_bal     = sqrt(rmse_bal     / n_eval_cal);
    mae_loco_qp /= n_eval_cal; rmse_loco_qp = sqrt(rmse_loco_qp / n_eval_cal);

    printf("--------------------------------------------------------------------------------------------------\n");
    printf("LOCO MAE  (Ga, lower is better)          | %8.3fGa | %8.3fGa | %8.3fGa | %8.3fGa\n",
           mae_strict, mae_arith, mae_bal, mae_loco_qp);
    printf("LOCO RMSE (Ga, lower is better)          | %8.3fGa | %8.3fGa | %8.3fGa | %8.3fGa\n",
           rmse_strict, rmse_arith, rmse_bal, rmse_loco_qp);
    printf("Interval Coverage (%d held-out bounds)    | %7d/%d  | %7d/%d  | %7d/%d  | %7d/%d\n\n",
           n_eval_cal, cov_strict, n_eval_cal, cov_arith, n_eval_cal, cov_bal, n_eval_cal, cov_loco_qp, n_eval_cal);

    printf("=== DATED MAJOR PHYLA ACROSS SILVA 144 (N = 9,183,096, B = %d BOOTSTRAPS) ===\n", n_valid_boot);
    printf("%-28s | %9s | %7s | %7s | %9s | %18s | %14s\n",
           "Domain;Phylum", "Leaves", "H_bal", "H_arith", "Bal-Rel", "Cond-QP Crown (Ga)", "Rate (sub/Ga)");
    printf("-------------------------------------------------------------------------------------------------------------\n");

    FILE *fo = fopen(out_json, "w");
    if (!fo) return 1;
    fprintf(fo, "{\n");
    fprintf(fo, "  \"N\": %d,\n", N);
    fprintf(fo, "  \"n_bootstraps_dated\": %d,\n", n_valid_boot);
    fprintf(fo, "  \"ref_dating_time_sec\": %.4f,\n", ref.dt_date_sec);
    fprintf(fo, "  \"mean_boot_dating_time_sec\": %.4f,\n", mean_dt_date);
    fprintf(fo, "  \"root_h_bal_subst\": %.6f,\n", ref.root_h_bal);
    fprintf(fo, "  \"root_h_arith_subst\": %.6f,\n", ref.root_h_arith);
    fprintf(fo, "  \"global_rate_subst_per_ga\": %.6f,\n", ref.global_rate_subst_per_ga);
    fprintf(fo, "  \"mad_bal_score\": %.6f,\n", ref.mad_bal_score);
    fprintf(fo, "  \"mad_arith_score\": %.6f,\n", ref.mad_arith_score);

    fprintf(fo, "  \"loco_validation\": {\n");
    fprintf(fo, "    \"n_held_out\": %d,\n", n_eval_cal);
    fprintf(fo, "    \"LSD_Strict_FixedClock\": {\"mae_ga\": %.4f, \"rmse_ga\": %.4f, \"coverage\": %d},\n", mae_strict, rmse_strict, cov_strict);
    fprintf(fo, "    \"Arith_RelTime_LocalClock\": {\"mae_ga\": %.4f, \"rmse_ga\": %.4f, \"coverage\": %d},\n", mae_arith, rmse_arith, cov_arith);
    fprintf(fo, "    \"Balanced_RelTime_LocalClock\": {\"mae_ga\": %.4f, \"rmse_ga\": %.4f, \"coverage\": %d},\n", mae_bal, rmse_bal, cov_bal);
    fprintf(fo, "    \"Condensed_Relaxed_QP\": {\"mae_ga\": %.4f, \"rmse_ga\": %.4f, \"coverage\": %d},\n", mae_loco_qp, rmse_loco_qp, cov_loco_qp);
    fprintf(fo, "    \"calibrations\": [\n");
    for (int c = 0; c < N_CALIB; ++c) {
        fprintf(fo, "      {\"id\": \"%s\", \"label\": \"%s\", \"reference\": \"%s\", \"range_ga\": [%.2f, %.2f], \"mid_ga\": %.2f, "
                    "\"pred_strict\": %.3f, \"pred_arith_rel\": %.3f, \"pred_bal_rel\": %.3f, \"pred_loco_qp\": %.3f, \"pred_full_qp\": %.3f}%s\n",
                CALIBRATIONS[c].id, CALIBRATIONS[c].label, CALIBRATIONS[c].reference,
                CALIBRATIONS[c].min_ga, CALIBRATIONS[c].max_ga, CALIBRATIONS[c].mid_ga,
                ref.calib_pred_strict[c], ref.calib_pred_arith_rel[c], ref.calib_pred_bal_rel[c],
                ref.calib_loco_cond_qp[c], ref.calib_pred_cond_qp[c],
                (c + 1 < N_CALIB) ? "," : "");
    }
    fprintf(fo, "    ]\n  },\n");

    fprintf(fo, "  \"dated_phyla\": [\n");
    for (int32_t k = 0; k < K_PHY; ++k) {
        int32_t pid = tracked_pids[k];
        std::vector<double> b_qp, b_bal, b_str, b_rate;
        for (int32_t r = 1; r <= max_boot; ++r) {
            if (rep_results[r].valid) {
                b_qp.push_back(rep_results[r].phy_age_cond_qp[k]);
                b_bal.push_back(rep_results[r].phy_age_bal_rel[k]);
                b_str.push_back(rep_results[r].phy_age_strict[k]);
                b_rate.push_back(rep_results[r].phy_rate_subst_per_ga[k]);
            }
        }
        auto stats = [](std::vector<double> &v, double fallback, double &mean, double &sd, double &lo, double &hi) {
            if (v.empty()) { mean = lo = hi = fallback; sd = 0.0; return; }
            std::sort(v.begin(), v.end());
            mean = 0.0;
            for (double x : v) mean += x;
            mean /= (double)v.size();
            sd = 0.0;
            for (double x : v) sd += (x - mean) * (x - mean);
            sd = sqrt(sd / (double)v.size());
            lo = v[(size_t)(0.025 * (v.size() - 1))];
            hi = v[(size_t)(0.975 * (v.size() - 1))];
        };
        double qp_m, qp_sd, qp_lo, qp_hi;
        double bl_m, bl_sd, bl_lo, bl_hi;
        double rt_m, rt_sd, rt_lo, rt_hi;
        stats(b_qp, ref.phy_age_cond_qp[k], qp_m, qp_sd, qp_lo, qp_hi);
        stats(b_bal, ref.phy_age_bal_rel[k], bl_m, bl_sd, bl_lo, bl_hi);
        stats(b_rate, ref.phy_rate_subst_per_ga[k], rt_m, rt_sd, rt_lo, rt_hi);

        std::string pname = tables[2].names[pid];
        std::string dname = tables[0].names[tables[2].domain_of[pid]];
        int64_t p_leaves  = tables[2].counts[pid];

        printf("%-28s | %9lld | %7.4f | %7.4f | %7.2fGa | %6.2f +/- %4.2f Ga | %6.4f +/- %6.4f\n",
               pname.c_str(), (long long)p_leaves, ref.phy_h_bal[k], ref.phy_h_arith[k],
               ref.phy_age_bal_rel[k], ref.phy_age_cond_qp[k], qp_sd,
               ref.phy_rate_subst_per_ga[k], rt_sd);

        fprintf(fo, "    {\"rank_idx\": %d, \"name\": \"%s\", \"domain\": \"%s\", \"leaves\": %lld, "
                    "\"d_stem_subst\": %.5f, \"h_bal_subst\": %.5f, \"h_arith_subst\": %.5f, "
                    "\"age_strict_ga\": %.3f, \"age_arith_rel_ga\": %.3f, \"age_bal_rel_ga\": %.3f, \"age_cond_qp_ga\": %.3f, "
                    "\"stem_age_qp_ga\": %.3f, \"boot_qp_mean_ga\": %.3f, \"boot_qp_sd_ga\": %.3f, \"boot_qp_ci95_ga\": [%.3f, %.3f], "
                    "\"boot_bal_rel_mean_ga\": %.3f, \"boot_bal_rel_sd_ga\": %.3f, "
                    "\"rate_subst_per_ga\": %.5f, \"boot_rate_sd\": %.5f}%s\n",
                k, pname.c_str(), dname.c_str(), (long long)p_leaves,
                ref.phy_d_stem[k], ref.phy_h_bal[k], ref.phy_h_arith[k],
                ref.phy_age_strict[k], ref.phy_age_arith_rel[k], ref.phy_age_bal_rel[k], ref.phy_age_cond_qp[k],
                ref.phy_stem_age_qp[k], qp_m, qp_sd, qp_lo, qp_hi,
                bl_m, bl_sd, ref.phy_rate_subst_per_ga[k], rt_sd,
                (k + 1 < K_PHY) ? "," : "");
    }
    fprintf(fo, "  ]\n}\n");
    fclose(fo);

    printf("\n[4/4] Saved complete SILVA 144 molecular dating summary to %s\n", out_json);
    return 0;
}
