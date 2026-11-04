import os
import sys
import time
import math
import signal
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_clade_surgery import (
    CKPT_ROOT, N_LAYERS, N_NEURONS,
    CODE_SNIPPETS, load_clade_scores, build_clade_mask_top_k,
    extract_mlp_state, apply_mlp_graft, eval_suite_metrics
)
from eval_definitive_proof import (
    MATH_PROMPTS, CODE_GEN_PROMPTS,
    compute_full_3matrix_frob_scores, eval_live_code_pass_at_1,
    generate_r1_native_traces, apply_task_vector_clade_merge
)

def build_clade_mask_exact_count(clade_id, clade_score, n_clades_to_pick, highest=True, seed=None):
    """Selects exactly `n_clades_to_pick` monophyletic clades globally across all 48 layers."""
    items = []
    rng = np.random.default_rng(seed) if seed is not None else None
    for l in range(N_LAYERS):
        cids = clade_id[l]
        scores = clade_score[l]
        uniq_c, first_idx, counts = np.unique(cids, return_index=True, return_counts=True)
        for c, idx, cnt in zip(uniq_c, first_idx, counts):
            sc = float(scores[idx]) if rng is None else float(rng.random())
            items.append((sc, l, int(c), int(cnt)))
    items.sort(key=lambda x: x[0], reverse=highest)
    mask = np.zeros((N_LAYERS, N_NEURONS), dtype=bool)
    for sc, l, c, cnt in items[:n_clades_to_pick]:
        mask[l, clade_id[l] == c] = True
    return mask, len(items)

def build_clade_mask_per_layer(clade_id, score_2d, frac, highest=True, seed=None):
    """
    Per-Layer Stratified Selection: within EACH layer l=0..47, aggregates `score_2d[l]`
    over each monophyletic clade (mean score in clade) and selects the top `frac` of neurons in layer l.
    """
    mask = np.zeros((N_LAYERS, N_NEURONS), dtype=bool)
    target_per_layer = int(round(frac * N_NEURONS))
    for l in range(N_LAYERS):
        rng = np.random.default_rng(seed + l * 17) if seed is not None else None
        cids = clade_id[l]
        sc_l = score_2d[l]
        uniq_c = np.unique(cids)
        cl_items = []
        for c in uniq_c:
            idx_mask = (cids == c)
            cnt = int(idx_mask.sum())
            val = float(sc_l[idx_mask].mean()) if rng is None else float(rng.random())
            cl_items.append((val, int(c), cnt))
        cl_items.sort(key=lambda x: x[0], reverse=highest)
        acc = 0
        for val, c, cnt in cl_items:
            if acc + cnt > target_per_layer and acc > 0:
                if abs((acc + cnt) - target_per_layer) > abs(acc - target_per_layer):
                    break
            mask[l, cids == c] = True
            acc += cnt
            if acc >= target_per_layer:
                break
    return mask

@torch.no_grad()
def compute_spherical_kmeans_switch_scores(base_mlp, target_mlp, n_clusters=96, n_iters=12, device="cuda:0"):
    """
    Competitor Baseline: Spherical k-Means Cluster-Switching & Voronoi Drift.
    In each layer l, clusters the N=13,824 L2-normalized Base down_proj columns into `n_clusters`
    spherical centroids on GPU, then measures how strongly Target neurons switch cluster assignments
    and shift relative to their Base Voronoi cell.
    """
    switch_scores = np.zeros((N_LAYERS, N_NEURONS), dtype=np.float32)
    gen = torch.Generator(device=device)
    gen.manual_seed(20261104)

    for l in range(N_LAYERS):
        w_b = base_mlp[l][2].to(device).float().t()    # [13824, 5120]
        w_t = target_mlp[l][2].to(device).float().t()  # [13824, 5120]
        w_b = F.normalize(w_b, p=2, dim=1)
        w_t = F.normalize(w_t, p=2, dim=1)

        perm = torch.randperm(N_NEURONS, generator=gen, device=device)[:n_clusters]
        centroids = w_b[perm].clone()
        assign_b = torch.zeros(N_NEURONS, dtype=torch.long, device=device)

        for _ in range(n_iters):
            sims = torch.matmul(w_b, centroids.t())  # [N, K]
            assign_b = sims.argmax(dim=1)
            for k in range(n_clusters):
                m = (assign_b == k)
                if m.any():
                    centroids[k] = F.normalize(w_b[m].mean(dim=0, keepdim=True), p=2, dim=1)[0]

        # Evaluate target assignments against Base centroids
        sims_b = torch.matmul(w_b, centroids.t())
        sims_t = torch.matmul(w_t, centroids.t())
        assign_t = sims_t.argmax(dim=1)
        switched = (assign_t != assign_b).float()

        # Margin change in Voronoi assignment: how much target moved away from its Base cluster vs best alternative
        own_sim_b = sims_b.gather(1, assign_b.unsqueeze(1)).squeeze(1)
        own_sim_t = sims_t.gather(1, assign_b.unsqueeze(1)).squeeze(1)
        max_sim_t = sims_t.max(dim=1).values
        voronoi_violation = switched + (max_sim_t - own_sim_t) + 0.25 * (own_sim_b - own_sim_t)

        # Average per k-means cluster so it acts as a cluster-level competitor
        cluster_score = torch.zeros(N_NEURONS, dtype=torch.float32, device=device)
        for k in range(n_clusters):
            m = (assign_b == k)
            if m.any():
                cluster_score[m] = voronoi_violation[m].mean()
        switch_scores[l] = cluster_score.cpu().numpy()
    return switch_scores

@torch.no_grad()
def apply_dare_mlp(model, base_mlp, target_mlp, keep_frac, rescale=True, seed=2026):
    """
    Competitor Baseline: DARE (Drop And REscale, Yu et al. ICML 2024).
    Randomly keeps `keep_frac` of neurons/deltas per layer and rescales kept deltas by 1/keep_frac.
    """
    dev = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    scale = (1.0 / keep_frac) if rescale else 1.0
    for l in range(N_LAYERS):
        mlp = model.model.layers[l].mlp
        keep_n = int(round(keep_frac * N_NEURONS))
        idx = rng.choice(N_NEURONS, size=keep_n, replace=False)
        m_row = torch.zeros(N_NEURONS, dtype=torch.bool, device=dev)
        m_row[torch.from_numpy(idx).to(dev)] = True

        gb, ub, db = base_mlp[l]
        gt, ut, dt = target_mlp[l]
        gb, ub, db = gb.to(dev), ub.to(dev), db.to(dev)
        gt, ut, dt = gt.to(dev), ut.to(dev), dt.to(dev)

        g = gb.clone()
        u = ub.clone()
        d = db.clone()
        g[m_row] = gb[m_row] + (gt[m_row] - gb[m_row]) * scale
        u[m_row] = ub[m_row] + (ut[m_row] - ub[m_row]) * scale
        d[:, m_row] = db[:, m_row] + (dt[:, m_row] - db[:, m_row]) * scale

        mlp.gate_proj.weight.copy_(g)
        mlp.up_proj.weight.copy_(u)
        mlp.down_proj.weight.copy_(d)

@torch.no_grad()
def apply_ties_merge_mlp(model, base_mlp, coder_mlp, r1_mlp, density=0.25, r1_weight=0.5):
    """
    Competitor Baseline: TIES-Merging (Yadav et al., NeurIPS 2023) on MLP task vectors.
    1. Trim: For each model, keep top `density` parameters by magnitude per tensor, zeroing the rest.
    2. Elect Sign: Compute aggregate sign across models weighted by magnitude.
    3. Disjoint Merge: Average only task-vector entries matching the elected sign.
    """
    dev = next(model.parameters()).device
    w_c = 1.0
    w_r = r1_weight
    for l in range(N_LAYERS):
        mlp = model.model.layers[l].mlp
        gb, ub, db = [x.to(dev).float() for x in base_mlp[l]]
        gc, uc, dc = [x.to(dev).float() for x in coder_mlp[l]]
        gr, ur, dr = [x.to(dev).float() for x in r1_mlp[l]]

        for base_w, c_w, r_w, param in [
            (gb, gc, gr, mlp.gate_proj.weight),
            (ub, uc, ur, mlp.up_proj.weight),
            (db, dc, dr, mlp.down_proj.weight),
        ]:
            tv_c = c_w - base_w
            tv_r = r_w - base_w
            # Trim to top `density` magnitude in each task vector
            k_keep = int(round(density * tv_c.numel()))
            thresh_c = torch.topk(tv_c.abs().view(-1), k_keep, largest=True).values.min()
            thresh_r = torch.topk(tv_r.abs().view(-1), k_keep, largest=True).values.min()
            tv_c_trim = torch.where(tv_c.abs() >= thresh_c, tv_c, torch.zeros_like(tv_c))
            tv_r_trim = torch.where(tv_r.abs() >= thresh_r, tv_r, torch.zeros_like(tv_r))

            # Elect sign
            elected_sign = torch.sign(w_c * tv_c_trim + w_r * tv_r_trim)
            match_c = (torch.sign(tv_c_trim) == elected_sign) & (tv_c_trim != 0)
            match_r = (torch.sign(tv_r_trim) == elected_sign) & (tv_r_trim != 0)
            denom = match_c.float() * w_c + match_r.float() * w_r
            merged_tv = torch.where(
                denom > 0,
                (match_c.float() * w_c * tv_c_trim + match_r.float() * w_r * tv_r_trim) / denom.clamp(min=1e-6),
                tv_c,
            )
            param.copy_((base_w + merged_tv).to(torch.bfloat16))

def main():
    t0 = time.time()
    scores = load_clade_scores()
    clade_id = scores["clade_id"]
    tok = AutoTokenizer.from_pretrained(os.path.join(CKPT_ROOT, "base"))
    _, total_clades = build_clade_mask_exact_count(clade_id, scores["coder_score"], 1)

    print("Step 1/3: Extracting Base MLP weights on CPU and loading R1-Distill-14B on cuda:0...")
    sys.stdout.flush()
    m_base_cpu = AutoModelForCausalLM.from_pretrained(
        os.path.join(CKPT_ROOT, "base"),
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    base_mlp = extract_mlp_state(m_base_cpu)
    del m_base_cpu

    m_r1 = AutoModelForCausalLM.from_pretrained(
        os.path.join(CKPT_ROOT, "r1"),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    m_r1.eval()
    r1_mlp = extract_mlp_state(m_r1)

    print("Computing R1 3-Matrix Frobenius Norm & Spherical k-Means Cluster-Switching baselines...")
    sys.stdout.flush()
    r1_frob   = compute_full_3matrix_frob_scores(base_mlp, r1_mlp)
    r1_kmeans = compute_spherical_kmeans_switch_scores(base_mlp, r1_mlp, n_clusters=96, device="cuda:0")

    r1_native_traces = generate_r1_native_traces(m_r1, tok)
    r1_ref_native    = eval_suite_metrics(m_r1, tok, r1_native_traces)

    print("\n==============================================================================================================================")
    print(f"TABLE 2: ULTRA-SPARSE CLADE KNOCK-OUT ON R1-DISTILL-14B (Top 5, 10, 25, 50 Clades out of {total_clades})")
    print("==============================================================================================================================")
    print(f"{'Clade Knock-Out Rule (R1-Distill-14B)':<44s} | {'Budget':>8s} | {'Neurons':>7s} | {'R1 CoT PPL':>12s} | {'Next-Tok Acc':>12s} | {'KL(R1 || Edit)':>15s}")
    print("------------------------------------------------------------------------------------------------------------------------------")
    print(f"{'Pure R1-Distill-14B (Unedited)':<44s} | {f'0/{total_clades}':>8s} | {'0.00%':>7s} | {r1_ref_native['ppl']:12.4f} | {r1_ref_native['acc']:11.2f}% | {0.0:15.5f}")

    for n_cl in [5, 10, 25, 50, 100]:
        print("------------------------------------------------------------------------------------------------------------------------------")
        for label, sc_arr, hi, sd in [
            (f"Revert Top-{n_cl:<3d} Quartet-Violated Clades (Tree)", scores["r1_score"], True, None),
            (f"Revert Top-{n_cl:<3d} Spherical k-Means Clades",       r1_kmeans,          True, None),
            (f"Revert Top-{n_cl:<3d} Cosine-Drift Clades (1-cos)",    1.0 - scores["r1_cos"], True, None),
            (f"Revert Top-{n_cl:<3d} 3-Matrix Frobenius Clades (L2)", r1_frob,            True, None),
            (f"Revert Random-{n_cl:<3d} Monophyletic Clades",         scores["r1_score"], True, 4000 + n_cl),
            (f"Revert Bottom-{n_cl:<3d} Topo-Conserved Clades",       scores["r1_score"], False, None),
        ]:
            m_Revert, _ = build_clade_mask_exact_count(clade_id, sc_arr, n_cl, highest=hi, seed=sd)
            apply_mlp_graft(m_r1, r1_mlp, base_mlp, m_Revert)
            res = eval_suite_metrics(m_r1, tok, r1_native_traces, ref_logprobs=r1_ref_native["logprobs"])
            print(f"{label:<44s} | {f'{n_cl}/{total_clades}':>8s} | {m_Revert.mean()*100:6.2f}% | {res['ppl']:12.4f} | {res['acc']:11.2f}% | {res['kl']:15.5f}")
            sys.stdout.flush()

    del m_r1
    torch.cuda.empty_cache()

    print("\nStep 2/3: Loading Qwen2.5-Coder-14B on cuda:0 and computing Coder baselines...")
    sys.stdout.flush()
    m_coder = AutoModelForCausalLM.from_pretrained(
        os.path.join(CKPT_ROOT, "coder"),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    m_coder.eval()
    coder_mlp = extract_mlp_state(m_coder)

    coder_frob   = compute_full_3matrix_frob_scores(base_mlp, coder_mlp)
    coder_kmeans = compute_spherical_kmeans_switch_scores(base_mlp, coder_mlp, n_clusters=96, device="cuda:0")

    coder_ref_code = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS)
    coder_pass_orig, n_code_probs = eval_live_code_pass_at_1(m_coder, tok)

    print("\n==============================================================================================================================")
    print(f"TABLE 1: ULTRA-SPARSE CLADE KNOCK-OUT ON QWEN2.5-CODER-14B (Top 5, 10, 25, 50 Clades out of {total_clades} Total Clades)")
    print("  Reverting ONLY k clades (< 0.2% to 1.2% of the network) back to Qwen2.5-14B Base:")
    print("==============================================================================================================================")
    print(f"{'Clade Knock-Out Rule':<44s} | {'Clades':>8s} | {'Neurons':>7s} | {'Code PPL':>12s} | {'Next-Tok Acc':>12s} | {'KL(Coder||Edit)':>15s}")
    print("------------------------------------------------------------------------------------------------------------------------------")
    print(f"{'Pure Qwen2.5-Coder-14B (0 Clades Reverted)':<44s} | {f'0/{total_clades}':>8s} | {'0.00%':>7s} | {coder_ref_code['ppl']:12.4f} | {coder_ref_code['acc']:11.2f}% | {0.0:15.5f}")

    for n_cl in [5, 10, 25, 50, 100]:
        print("------------------------------------------------------------------------------------------------------------------------------")
        for label, sc_arr, hi, sd in [
            (f"Revert Top-{n_cl:<3d} Quartet-Violated Clades (Tree)", scores["coder_score"], True, None),
            (f"Revert Top-{n_cl:<3d} Spherical k-Means Clades",       coder_kmeans,          True, None),
            (f"Revert Top-{n_cl:<3d} Cosine-Drift Clades (1-cos)",    1.0 - scores["coder_cos"], True, None),
            (f"Revert Top-{n_cl:<3d} 3-Matrix Frobenius Clades (L2)", coder_frob,            True, None),
            (f"Revert Random-{n_cl:<3d} Monophyletic Clades",         scores["coder_score"], True, 3000 + n_cl),
            (f"Revert Bottom-{n_cl:<3d} Topo-Conserved Clades",       scores["coder_score"], False, None),
        ]:
            m_Revert, _ = build_clade_mask_exact_count(clade_id, sc_arr, n_cl, highest=hi, seed=sd)
            apply_mlp_graft(m_coder, coder_mlp, base_mlp, m_Revert)
            res = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])
            print(f"{label:<44s} | {f'{n_cl}/{total_clades}':>8s} | {m_Revert.mean()*100:6.2f}% | {res['ppl']:12.4f} | {res['acc']:11.2f}% | {res['kl']:15.5f}")
            sys.stdout.flush()

    print("\n==============================================================================================================================")
    print("TABLE 3: PER-LAYER STRATIFIED RETENTION vs. COMPETITORS ON QWEN2.5-CODER-14B (With Live Code Execution Pass@1)")
    print("  Every layer l=0..47 keeps the exact same k% budget (Tree vs. Spherical k-Means vs. Frobenius vs. Cosine vs. DARE)")
    print("==============================================================================================================================")
    print(f"{'Per-Layer Retention Method & Budget':<46s} | {'Kept %':>7s} | {'Code PPL':>11s} | {'Next-Tok Acc':>12s} | {'KL(Coder||Edit)':>15s} | {'Live Pass@1':>11s}")
    print("------------------------------------------------------------------------------------------------------------------------------")
    print(f"{'Pure Qwen2.5-Coder-14B (100% Kept)':<46s} | {'100.0%':>7s} | {coder_ref_code['ppl']:11.4f} | {coder_ref_code['acc']:11.2f}% | {0.0:15.5f} | {f'{coder_pass_orig}/{n_code_probs} ({100*coder_pass_orig/n_code_probs:.0f}%)':>11s}")

    for keep_frac in [0.85, 0.70, 0.50]:
        pct = int(round(keep_frac * 100))
        print("------------------------------------------------------------------------------------------------------------------------------")
        for label, sc_arr, hi, sd in [
            (f"[Tree] Keep Top-{pct}% Quartet-Violated Clades",     scores["coder_score"], True, None),
            (f"[k-Means] Keep Top-{pct}% Spherical k-Means Clades", coder_kmeans,          True, None),
            (f"[Frobenius] Keep Top-{pct}% 3-Matrix L2 Clades",     coder_frob,            True, None),
            (f"[Cosine] Keep Top-{pct}% Cosine-Drift Clades",       1.0 - scores["coder_cos"], True, None),
            (f"[Random] Keep Random-{pct}% Monophyletic Clades",    scores["coder_score"], True, 5000 + pct),
            (f"[Conserved] Keep Bottom-{pct}% Topo-Conserved",      scores["coder_score"], False, None),
        ]:
            keep_mask = build_clade_mask_per_layer(clade_id, sc_arr, keep_frac, highest=hi, seed=sd)
            revert_mask = ~keep_mask
            apply_mlp_graft(m_coder, coder_mlp, base_mlp, revert_mask)
            res = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])
            p_cnt, p_tot = eval_live_code_pass_at_1(m_coder, tok)
            print(f"{label:<46s} | {keep_mask.mean()*100:6.1f}% | {res['ppl']:11.4f} | {res['acc']:11.2f}% | {res['kl']:15.5f} | {f'{p_cnt}/{p_tot} ({100*p_cnt/p_tot:.0f}%)':>11s}")
            sys.stdout.flush()

        # Also evaluate DARE (with and without 1/p rescale) at the same budget
        apply_dare_mlp(m_coder, base_mlp, coder_mlp, keep_frac, rescale=True, seed=6000 + pct)
        res_dare = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])
        p_cnt, p_tot = eval_live_code_pass_at_1(m_coder, tok)
        print(f"{f'[DARE] Keep {pct}% Random + 1/p Rescale (ICML 24)':<46s} | {keep_frac*100:6.1f}% | {res_dare['ppl']:11.4f} | {res_dare['acc']:11.2f}% | {res_dare['kl']:15.5f} | {f'{p_cnt}/{p_tot} ({100*p_cnt/p_tot:.0f}%)':>11s}")
        sys.stdout.flush()

    print("\n==============================================================================================================================")
    print("TABLE 4: MODEL MERGING BENCHMARK (Tree-Surgical Clade Grafting vs. TIES-Merging vs. Spherical k-Means vs. Frobenius)")
    print("  Merging R1-Distill-14B reasoning into Qwen2.5-Coder-14B at 5%, 10%, and 20% Clade Budgets")
    print("==============================================================================================================================")
    r1_z = (scores["r1_score"] - scores["r1_score"].mean()) / (scores["r1_score"].std() + 1e-8)
    cd_z = (scores["coder_score"] - scores["coder_score"].mean()) / (scores["coder_score"].std() + 1e-8)
    surgical_diff = r1_z - cd_z

    km_r1_z = (r1_kmeans - r1_kmeans.mean()) / (r1_kmeans.std() + 1e-8)
    km_cd_z = (coder_kmeans - coder_kmeans.mean()) / (coder_kmeans.std() + 1e-8)
    kmeans_diff = km_r1_z - km_cd_z

    fr_r1_z = (r1_frob - r1_frob.mean()) / (r1_frob.std() + 1e-8)
    fr_cd_z = (coder_frob - coder_frob.mean()) / (coder_frob.std() + 1e-8)
    frob_diff = fr_r1_z - fr_cd_z

    apply_mlp_graft(m_coder, coder_mlp, coder_mlp, np.zeros((N_LAYERS, N_NEURONS), dtype=bool))
    coder_native_r1 = eval_suite_metrics(m_coder, tok, r1_native_traces, ref_logprobs=r1_ref_native["logprobs"])

    print(f"{'Model Merging / Clade Grafting Method':<46s} | {'Budget':>7s} | {'Code KL(C||M)':>13s} | {'Live Code':>11s} | {'R1 CoT PPL':>11s} | {'R1 CoT KL(R1||M)':>16s}")
    print("------------------------------------------------------------------------------------------------------------------------------")
    print(f"{'Pure Qwen2.5-Coder-14B (Unmerged)':<46s} | {'0.0%':>7s} | {0.0:13.5f} | {f'{coder_pass_orig}/{n_code_probs} ({100*coder_pass_orig/n_code_probs:.0f}%)':>11s} | {coder_native_r1['ppl']:11.4f} | {coder_native_r1['kl']:16.5f}")

    # Evaluate TIES-Merging (NeurIPS 2023)
    for ties_w in [0.5, 1.0]:
        apply_ties_merge_mlp(m_coder, base_mlp, coder_mlp, r1_mlp, density=0.25, r1_weight=ties_w)
        res_c = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])
        res_r = eval_suite_metrics(m_coder, tok, r1_native_traces, ref_logprobs=r1_ref_native["logprobs"])
        p_cnt, p_tot = eval_live_code_pass_at_1(m_coder, tok)
        print(f"{f'[TIES-Merging] Trim 25% + Sign Election (w={ties_w})':<46s} | {'25.0%':>7s} | {res_c['kl']:13.5f} | {f'{p_cnt}/{p_tot} ({100*p_cnt/p_tot:.0f}%)':>11s} | {res_r['ppl']:11.4f} | {res_r['kl']:16.5f}")
        sys.stdout.flush()

    for frac in [0.05, 0.10, 0.20]:
        pct = int(round(frac * 100))
        print("------------------------------------------------------------------------------------------------------------------------------")
        for label, sc_arr, hi, sd in [
            (f"[Tree-Surgical] Top-{pct}% (z_R1 - z_Coder)",     surgical_diff, True, None),
            (f"[k-Means-Diff] Top-{pct}% (km_R1 - km_Coder)",    kmeans_diff,   True, None),
            (f"[Frobenius-Diff] Top-{pct}% (L2_R1 - L2_Coder)",  frob_diff,     True, None),
            (f"[Random Clades] {pct}% Monophyletic Clades",      surgical_diff, True, 7000 + pct),
            (f"[Anti-Surgical] Bottom-{pct}% (z_R1 - z_Coder)",  surgical_diff, False, None),
        ]:
            g_mask = build_clade_mask_top_k(clade_id, sc_arr, frac, highest=hi, seed=sd)
            apply_task_vector_clade_merge(m_coder, base_mlp, coder_mlp, r1_mlp, g_mask, scale=1.0)
            res_c = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])
            res_r = eval_suite_metrics(m_coder, tok, r1_native_traces, ref_logprobs=r1_ref_native["logprobs"])
            p_cnt, p_tot = eval_live_code_pass_at_1(m_coder, tok)
            print(f"{label:<46s} | {g_mask.mean()*100:6.1f}% | {res_c['kl']:13.5f} | {f'{p_cnt}/{p_tot} ({100*p_cnt/p_tot:.0f}%)':>11s} | {res_r['ppl']:11.4f} | {res_r['kl']:16.5f}")
            sys.stdout.flush()

    print("==============================================================================================================================\n")
    print(f"Completed all competitor & ultra-sparse clade benchmarks in {time.time() - t0:.1f}s!")

if __name__ == "__main__":
    main()

