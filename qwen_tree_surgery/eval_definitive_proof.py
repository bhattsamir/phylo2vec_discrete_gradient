import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_clade_surgery import (
    CKPT_ROOT, N_LAYERS, N_NEURONS,
    CODE_SNIPPETS, load_clade_scores, build_clade_mask_top_k,
    extract_mlp_state, apply_mlp_graft, eval_suite_metrics, eval_suite_metrics_from_saved
)

# Math & Logic Prompts where we both generate reference R1-Distill CoT traces and test live accuracy
MATH_PROMPTS = [
    ("Compute the sum of all positive integers n <= 1000 such that gcd(n, 30) = 1? Wait: first find how many such integers exist. Put the count inside \\boxed{}.", "266"),
    ("If x + 1/x = 3, compute the exact value of x^5 + 1/x^5. Put your final numerical answer inside \\boxed{}.", "123"),
    ("A sequence satisfies a_1 = 2 and a_{n+1} = 3a_n - 2 for all n >= 1. Find the exact value of a_8. Put your final answer inside \\boxed{}.", "2188"),
    ("How many trailing zeros does 2025! have in base 10? Put your final numerical answer inside \\boxed{}.", "505"),
    ("In a triangle with side lengths 13, 14, and 15, find the radius of the inscribed circle (inradius). Put your final answer inside \\boxed{}.", "4"),
    ("Four fair six-sided dice are rolled. For how many outcomes is the sum of the four dice equal to 7? Put your final answer inside \\boxed{}.", "20"),
    ("Find the remainder when 2^100 is divided by 101. Put your final answer inside \\boxed{}.", "1"),
    ("How many 4-digit positive integers have strictly increasing digits from left to right (e.g. 1358)? Put your final answer inside \\boxed{}.", "126"),
]

CODE_GEN_PROMPTS = [
    (
        "def longest_increasing_subsequence(nums: list[int]) -> int:\n"
        "    \"\"\"Return the length of the longest strictly increasing subsequence in O(n log n) time.\"\"\"\n",
        "assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4\n"
        "assert longest_increasing_subsequence([0, 1, 0, 3, 2, 3]) == 4\n"
        "assert longest_increasing_subsequence([7, 7, 7, 7]) == 1\n"
    ),
    (
        "def min_edit_distance(word1: str, word2: str) -> int:\n"
        "    \"\"\"Return the Levenshtein edit distance between word1 and word2.\"\"\"\n",
        "assert min_edit_distance('horse', 'ros') == 3\n"
        "assert min_edit_distance('intention', 'execution') == 5\n"
        "assert min_edit_distance('', 'abc') == 3\n"
    ),
    (
        "def subarray_sum_equals_k(nums: list[int], k: int) -> int:\n"
        "    \"\"\"Return the total number of contiguous subarrays whose sum equals k.\"\"\"\n",
        "assert subarray_sum_equals_k([1, 1, 1], 2) == 2\n"
        "assert subarray_sum_equals_k([1, 2, 3, -3, 1, 2], 3) == 5\n"
    ),
    (
        "def is_valid_parentheses(s: str) -> bool:\n"
        "    \"\"\"Return True iff the bracket string s containing '()[]{}' is valid and balanced.\"\"\"\n",
        "assert is_valid_parentheses('()[]{}') is True\n"
        "assert is_valid_parentheses('([)]') is False\n"
        "assert is_valid_parentheses('{[]}') is True\n"
    ),
    (
        "def max_subarray_sum(nums: list[int]) -> int:\n"
        "    \"\"\"Return the maximum sum of any non-empty contiguous subarray (Kadane's algorithm).\"\"\"\n",
        "assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\n"
        "assert max_subarray_sum([-5, -2, -9]) == -2\n"
    ),
    (
        "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n"
        "    \"\"\"Merge all overlapping intervals and return sorted non-overlapping intervals.\"\"\"\n",
        "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
        "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
    ),
    (
        "def coin_change_min_coins(coins: list[int], amount: int) -> int:\n"
        "    \"\"\"Return fewest coins needed to make amount, or -1 if impossible.\"\"\"\n",
        "assert coin_change_min_coins([1, 2, 5], 11) == 3\n"
        "assert coin_change_min_coins([2], 3) == -1\n"
        "assert coin_change_min_coins([1], 0) == 0\n"
    ),
    (
        "def topological_sort_kahn(n: int, edges: list[tuple[int, int]]) -> list[int]:\n"
        "    \"\"\"Return a valid topological ordering of 0..n-1 given directed edges (u, v), or [] if cycle.\"\"\"\n",
        "order = topological_sort_kahn(4, [(0, 1), (0, 2), (1, 3), (2, 3)])\n"
        "assert len(order) == 4 and order[0] == 0 and order[-1] == 3\n"
        "assert topological_sort_kahn(2, [(0, 1), (1, 0)]) == []\n"
    ),
]

@torch.no_grad()
def compute_full_3matrix_frob_scores(src_mlp_state, dst_mlp_state):
    """
    Computes for every neuron (l, i) the exact Frobenius L2 norm squared of its coupled weight change:
      ||Delta gate_proj[i, :]||_2^2 + ||Delta up_proj[i, :]||_2^2 + ||Delta down_proj[:, i]||_2^2
    """
    frob = np.zeros((N_LAYERS, N_NEURONS), dtype=np.float32)
    for l in range(N_LAYERS):
        g0, u0, d0 = src_mlp_state[l]
        g1, u1, d1 = dst_mlp_state[l]
        dg = (g1.float() - g0.float()).pow(2).sum(dim=1)
        du = (u1.float() - u0.float()).pow(2).sum(dim=1)
        dd = (d1.float() - d0.float()).pow(2).sum(dim=0)
        frob[l] = (dg + du + dd).sqrt().numpy()
    return frob

@torch.no_grad()
def eval_live_code_pass_at_1(model, tok):
    import signal
    def _timeout_handler(signum, frame):
        raise TimeoutError("Execution timed out")

    dev = next(model.parameters()).device
    orig_pad_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    prompts = [p for p, _ in CODE_GEN_PROMPTS]
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    in_len = enc["input_ids"].shape[1]
    outs = model.generate(
        **enc,
        max_new_tokens=160,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    tok.padding_side = orig_pad_side

    passed = 0
    for idx, (prompt, test_code) in enumerate(CODE_GEN_PROMPTS):
        gen = tok.decode(outs[idx][in_len:], skip_special_tokens=True)
        lines = []
        for line in gen.splitlines():
            if line.startswith("def ") or line.startswith("class ") or line.startswith("if __name__") or line.startswith("# Test") or line.startswith("print("):
                break
            lines.append(line)
        full_code = prompt + "\n".join(lines) + "\n" + test_code
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(1)
        try:
            ns = {}
            exec(full_code, ns, ns)
            passed += 1
        except Exception:
            pass
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    return passed, len(CODE_GEN_PROMPTS)

@torch.no_grad()
def generate_r1_native_traces(m_r1, tok):
    """Generates native Chain-of-Thought traces directly from Unedited R1-Distill-14B (batched)."""
    dev = next(m_r1.parameters()).device
    orig_pad_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    prompts = [f"<｜begin▁of▁sentence｜><｜User｜>{q}<｜Assistant｜><think>\n" for q, _ in MATH_PROMPTS]
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    outs = m_r1.generate(
        **enc,
        max_new_tokens=240,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    tok.padding_side = orig_pad_side
    return [tok.decode(outs[i], skip_special_tokens=True) for i in range(len(prompts))]

@torch.no_grad()
def apply_task_vector_clade_merge(model, base_mlp, coder_mlp, r1_mlp, r1_mask_2d, scale=1.0):
    """
    Starts from Coder-14B MLP weights, and on the selected R1 clades (`r1_mask_2d`),
    adds the R1-Distill task vector: W_merged = W_coder + scale * (W_r1 - W_base).
    """
    dev = next(model.parameters()).device
    for l in range(N_LAYERS):
        mlp = model.model.layers[l].mlp
        m_row = torch.from_numpy(r1_mask_2d[l]).to(dev)
        gb, ub, db = base_mlp[l]
        gc, uc, dc = coder_mlp[l]
        gr, ur, dr = r1_mlp[l]

        g = gc.to(dev).clone()
        u = uc.to(dev).clone()
        d = dc.to(dev).clone()
        if m_row.any():
            dg = (gr.to(dev)[m_row] - gb.to(dev)[m_row]) * scale
            du = (ur.to(dev)[m_row] - ub.to(dev)[m_row]) * scale
            dd = (dr.to(dev)[:, m_row] - db.to(dev)[:, m_row]) * scale
            g[m_row] = g[m_row] + dg
            u[m_row] = u[m_row] + du
            d[:, m_row] = d[:, m_row] + dd
        mlp.gate_proj.weight.copy_(g)
        mlp.up_proj.weight.copy_(u)
        mlp.down_proj.weight.copy_(d)

def main():
    t0 = time.time()
    scores = load_clade_scores()
    clade_id = scores["clade_id"]
    tok = AutoTokenizer.from_pretrained(os.path.join(CKPT_ROOT, "base"))

    print("Loading Base, R1-Distill-14B (cuda:0), and Coder-14B (cuda:1)...")
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

    m_coder = AutoModelForCausalLM.from_pretrained(
        os.path.join(CKPT_ROOT, "coder"),
        dtype=torch.bfloat16,
        device_map={"": "cuda:1"},
    )
    m_coder.eval()
    coder_mlp = extract_mlp_state(m_coder)

    print("Computing exact 3-matrix Frobenius norm changes ||Delta W_MLP||_F per neuron...")
    coder_frob = compute_full_3matrix_frob_scores(base_mlp, coder_mlp)
    r1_frob    = compute_full_3matrix_frob_scores(base_mlp, r1_mlp)

    print("Generating native R1-Distill-14B Chain-of-Thought traces on the 8 Math problems...")
    sys.stdout.flush()
    r1_native_traces = generate_r1_native_traces(m_r1, tok)
    r1_ref_native    = eval_suite_metrics(m_r1, tok, r1_native_traces)
    del m_r1
    torch.cuda.empty_cache()

    coder_ref_code   = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS)
    coder_pass_orig, n_code_probs = eval_live_code_pass_at_1(m_coder, tok)

    print("\n========================================================================================================================")
    print("EXPERIMENT 2: SPARSE POST-TRAINING RETENTION CURVE ON QWEN2.5-CODER-14B (With Live Code Execution Pass@1)")
    print("  Comparing Phylogenetic Quartet-Violated Clades vs. 3-Matrix Frobenius Norm vs. Cosine Drift vs. Random Clades")
    print("========================================================================================================================")
    print(f"{'Retention Method & Budget':<44s} | {'Kept %':>8s} | {'Code PPL':>11s} | {'Next-Tok Acc':>12s} | {'KL(Coder||Edit)':>15s} | {'Live Pass@1':>11s}")
    print("------------------------------------------------------------------------------------------------------------------------")
    print(f"{'Pure Qwen2.5-Coder-14B (100% Kept)':<44s} | {'100.0%':>8s} | {coder_ref_code['ppl']:11.4f} | {coder_ref_code['acc']:11.2f}% | {0.0:15.5f} | {f'{coder_pass_orig}/{n_code_probs} ({100*coder_pass_orig/n_code_probs:.0f}%)':>11s}")

    for keep_frac in [0.85, 0.70, 0.50]:
        pct = int(round(keep_frac * 100))
        print("------------------------------------------------------------------------------------------------------------------------")
        for label, sc_arr, hi, sd in [
            (f"Keep Top-{pct}% Quartet-Violated Clades (Tree)", scores["coder_score"], True, None),
            (f"Keep Top-{pct}% 3-Matrix Frobenius Clades (L2)", coder_frob,            True, None),
            (f"Keep Top-{pct}% Cosine-Drift Clades (1-cos)",    1.0 - scores["coder_cos"], True, None),
            (f"Keep Random-{pct}% Monophyletic Clades",         scores["coder_score"], True, 900 + pct),
            (f"Keep Bottom-{pct}% Topo-Conserved Clades",       scores["coder_score"], False, None),
        ]:
            keep_mask = build_clade_mask_top_k(clade_id, sc_arr, keep_frac, highest=hi, seed=sd)
            revert_mask = ~keep_mask
            apply_mlp_graft(m_coder, coder_mlp, base_mlp, revert_mask)
            res = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])
            p_cnt, p_tot = eval_live_code_pass_at_1(m_coder, tok)
            print(f"{label:<44s} | {keep_mask.mean()*100:7.1f}% | {res['ppl']:11.4f} | {res['acc']:11.2f}% | {res['kl']:15.5f} | {f'{p_cnt}/{p_tot} ({100*p_cnt/p_tot:.0f}%)':>11s}")
            sys.stdout.flush()

    print("\n========================================================================================================================")
    print("EXPERIMENT 3: CLADE-SURGICAL TASK-VECTOR GRAFTING (Injecting R1-Distill Task Vector into Qwen2.5-Coder-14B)")
    print("  W_merged = W_Coder + (W_R1 - W_Base) on 25% of clades: Tree-Surgical vs. Random vs. Anti-Surgical")
    print("========================================================================================================================")
    r1_z = (scores["r1_score"] - scores["r1_score"].mean()) / (scores["r1_score"].std() + 1e-8)
    cd_z = (scores["coder_score"] - scores["coder_score"].mean()) / (scores["coder_score"].std() + 1e-8)
    surgical_diff = r1_z - cd_z

    apply_mlp_graft(m_coder, coder_mlp, coder_mlp, np.zeros((N_LAYERS, N_NEURONS), dtype=bool))
    coder_native_r1 = eval_suite_metrics(m_coder, tok, r1_native_traces, ref_logprobs=r1_ref_native["logprobs"])

    print(f"{'Task-Vector Clade Grafting Method':<44s} | {'Grafted':>8s} | {'Code KL(C||M)':>13s} | {'Live Code':>11s} | {'R1 CoT PPL':>11s} | {'R1 CoT KL(R1||M)':>16s}")
    print("------------------------------------------------------------------------------------------------------------------------")
    print(f"{'Pure Qwen2.5-Coder-14B (0% R1 Task Vector)':<44s} | {'0.0%':>8s} | {0.0:13.5f} | {f'{coder_pass_orig}/{n_code_probs} ({100*coder_pass_orig/n_code_probs:.0f}%)':>11s} | {coder_native_r1['ppl']:11.4f} | {coder_native_r1['kl']:16.5f}")
    for label, sc_arr, hi, sd in [
        ("Tree-Surgical Clades (High z_R1 - z_Coder)", surgical_diff, True, None),
        ("Top R1 Quartet-Violated Clades (High z_R1)", scores["r1_score"], True, None),
        ("Random 25% Monophyletic Clades",             surgical_diff, True, 2026),
        ("Anti-Surgical Clades (Low z_R1 - z_Coder)",  surgical_diff, False, None),
    ]:
        g_mask = build_clade_mask_top_k(clade_id, sc_arr, 0.25, highest=hi, seed=sd)
        apply_task_vector_clade_merge(m_coder, base_mlp, coder_mlp, r1_mlp, g_mask, scale=1.0)
        res_c = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])
        res_r = eval_suite_metrics(m_coder, tok, r1_native_traces, ref_logprobs=r1_ref_native["logprobs"])
        p_cnt, p_tot = eval_live_code_pass_at_1(m_coder, tok)
        print(f"{label:<44s} | {g_mask.mean()*100:7.1f}% | {res_c['kl']:13.5f} | {f'{p_cnt}/{p_tot} ({100*p_cnt/p_tot:.0f}%)':>11s} | {res_r['ppl']:11.4f} | {res_r['kl']:16.5f}")
        sys.stdout.flush()
    print("========================================================================================================================\n")
    print(f"All definitive proof-of-principle experiments finished in {time.time() - t0:.1f}s!")

if __name__ == "__main__":
    main()

