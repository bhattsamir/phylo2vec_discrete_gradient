import os
import sys
import re
import time
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT_ROOT = "/tmp/qwen_tree_exp/checkpoints"
SCORES_BIN = "/tmp/qwen_tree_exp/clade_scores_48L.bin"
N_LAYERS = 48
N_NEURONS = 13824
DIM = 5120

# ==============================================================================
# 1. Evaluation Benchmarks:
#    (A) Exact Teacher-Forcing Cross-Entropy (NLL) & KL Divergence wrt Specialist
#    (B) Live Autoregressive Generation (Math Reasoning + Code Synthesis Pass@1)
# ==============================================================================

REASONING_TRACES = [
    """Problem: Find the number of positive integers n <= 1000 such that gcd(n, 30) = 1.
<think>
Wait, we need to count integers n in {1, 2, ..., 1000} that are coprime to 30.
First, factorize 30 = 2 * 3 * 5.
By Euler's totient function, in every block of 30 consecutive integers, there are phi(30) = 30 * (1 - 1/2) * (1 - 1/3) * (1 - 1/5) = 8 integers coprime to 30.
Let's divide 1000 by 30:
1000 = 33 * 30 + 10.
In the first 33 full blocks (from 1 to 990), there are 33 * 8 = 264 integers coprime to 30.
Now check the remaining 10 integers from 991 to 1000, which correspond to residues 1, 2, ..., 10 mod 30:
The residues in {1, ..., 10} coprime to 2, 3, 5 are:
- 1 (coprime)
- 7 (coprime)
All others (2, 3, 4, 5, 6, 8, 9, 10) are divisible by 2, 3, or 5.
So there are 2 additional integers (991 and 997).
Total count = 264 + 2 = 266.
</think>
Final Answer: 266""",

    """Problem: If x + 1/x = 3, compute x^5 + 1/x^5.
<think>
Let S_k = x^k + 1/x^k. We are given S_1 = 3.
First compute S_2:
S_2 = (x + 1/x)^2 - 2 = 3^2 - 2 = 7.
Next compute S_3:
S_3 = S_1 * S_2 - S_1 = 3 * 7 - 3 = 18.
Next compute S_5:
Notice that S_2 * S_3 = (x^2 + 1/x^2)(x^3 + 1/x^3) = (x^5 + 1/x^5) + (x + 1/x) = S_5 + S_1.
Therefore, S_5 = S_2 * S_3 - S_1 = 7 * 18 - 3 = 126 - 3 = 123.
Let's double check via S_4: S_4 = S_2^2 - 2 = 49 - 2 = 47, and S_5 = S_4 * S_1 - S_3 = 47 * 3 - 18 = 141 - 18 = 123. Both give 123.
</think>
Final Answer: 123""",

    """Problem: A sequence satisfies a_1 = 2 and a_{n+1} = 3a_n - 2 for all n >= 1. Find a_10.
<think>
Wait, let's find the fixed point c of the recurrence c = 3c - 2 -> 2c = 2 -> c = 1.
Subtract 1 from both sides:
a_{n+1} - 1 = 3(a_n - 1).
So b_n = a_n - 1 is a geometric progression with b_1 = a_1 - 1 = 2 - 1 = 1 and common ratio 3.
Thus b_n = 3^{n-1}, which means a_n = 3^{n-1} + 1.
For n = 10:
a_10 = 3^9 + 1.
Since 3^4 = 81, 3^8 = 6561, and 3^9 = 19683, we get a_10 = 19683 + 1 = 19684.
</think>
Final Answer: 19684""",

    """Problem: How many trailing zeros does 2025! have in base 10?
<think>
By Legendre's formula, the number of trailing zeros of n! is the exponent of 5 in the prime factorization of n!:
E_5(2025!) = floor(2025/5) + floor(2025/25) + floor(2025/125) + floor(2025/625).
Let's compute each term:
- floor(2025 / 5) = 405
- floor(2025 / 25) = 81
- floor(2025 / 125) = 16 (since 16 * 125 = 2000)
- floor(2025 / 625) = 3 (since 3 * 625 = 1875)
- floor(2025 / 3125) = 0
Summing these up: 405 + 81 + 16 + 3 = 505.
</think>
Final Answer: 505""",

    """Problem: Find the sum of all real roots of (x^2 - 5x + 5)^{x^2 - 9x + 20} = 1.
<think>
An exponential equation A^B = 1 holds in the real numbers under three cases:
Case 1: B = 0 and A != 0.
x^2 - 9x + 20 = 0 => (x - 4)(x - 5) = 0 => x = 4 or x = 5.
Check A != 0: for x=4, A = 16 - 20 + 5 = 1 != 0; for x=5, A = 25 - 25 + 5 = 5 != 0. Both x=4 and x=5 are valid.
Case 2: A = 1.
x^2 - 5x + 5 = 1 => x^2 - 5x + 4 = 0 => (x - 1)(x - 4) = 0 => x = 1 or x = 4.
(Note x = 4 was already counted in Case 1, so distinct roots so far: 1, 4, 5.)
Case 3: A = -1 and B is an even integer.
x^2 - 5x + 5 = -1 => x^2 - 5x + 6 = 0 => (x - 2)(x - 3) = 0 => x = 2 or x = 3.
Check whether B = x^2 - 9x + 20 is even:
- For x = 2: B = 4 - 18 + 20 = 6 (even, valid).
- For x = 3: B = 9 - 27 + 20 = 2 (even, valid).
So the distinct real roots are {1, 2, 3, 4, 5}.
Their sum is 1 + 2 + 3 + 4 + 5 = 15.
</think>
Final Answer: 15""",

    """Problem: In a triangle ABC, AB = 13, BC = 14, AC = 15. Find the inradius r.
<think>
First compute the semiperimeter s:
s = (13 + 14 + 15) / 2 = 42 / 2 = 21.
By Heron's formula, the area K of triangle ABC is:
K = sqrt(s(s - a)(s - b)(s - c)) = sqrt(21 * (21 - 14) * (21 - 15) * (21 - 13))
  = sqrt(21 * 7 * 6 * 8) = sqrt(7056) = 84.
The inradius r satisfies K = r * s, so:
r = K / s = 84 / 21 = 4.
</think>
Final Answer: 4""",

    """Problem: Determine the remainder when 2^{100} is divided by 101.
<think>
Wait, is 101 a prime number?
Yes, 101 is prime because it is not divisible by 2, 3, 5, 7 (and 11^2 = 121 > 101).
By Fermat's Little Theorem, since gcd(2, 101) = 1:
2^{101 - 1} = 2^{100} === 1 (mod 101).
Therefore, the remainder is 1.
</think>
Final Answer: 1""",

    """Problem: Four fair six-sided dice are rolled. What is the number of outcomes where the sum of the four dice is 7?
<think>
We want the number of positive integer solutions to x_1 + x_2 + x_3 + x_4 = 7 with 1 <= x_i <= 6.
Since x_i >= 1, if three variables are at least 1, the fourth variable is at most 7 - 3 = 4 <= 6, so the upper bound x_i <= 6 is automatically satisfied.
By stars and bars, the number of positive integer solutions to x_1 + x_2 + x_3 + x_4 = 7 is:
C(7 - 1, 4 - 1) = C(6, 3) = (6 * 5 * 4) / (3 * 2 * 1) = 20.
</think>
Final Answer: 20""",
]

CODE_SNIPPETS = [
    """def longest_increasing_subsequence(nums: list[int]) -> int:
    import bisect
    tails = []
    for x in nums:
        idx = bisect.bisect_left(tails, x)
        if idx == len(tails):
            tails.append(x)
        else:
            tails[idx] = x
    return len(tails)
""",
    """#include <vector>
#include <queue>
#include <limits>

std::vector<long long> dijkstra_shortest_paths(
    int n, int src, const std::vector<std::vector<std::pair<int, int>>>& adj
) {
    const long long INF = std::numeric_limits<long long>::max() / 4;
    std::vector<long long> dist(n, INF);
    using P = std::pair<long long, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    dist[src] = 0;
    pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
""",
    """class FenwickTree:
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i: int, delta: int) -> None:
        i += 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i: int) -> int:
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def range_query(self, left: int, right: int) -> int:
        if left > right:
            return 0
        return self.query(right) - (self.query(left - 1) if left > 0 else 0)
""",
    """WITH RECURSIVE OrgHierarchy AS (
    SELECT employee_id, manager_id, department_id, salary, 1 AS depth_level
    FROM employees
    WHERE manager_id IS NULL
    UNION ALL
    SELECT e.employee_id, e.manager_id, e.department_id, e.salary, h.depth_level + 1
    FROM employees e
    INNER JOIN OrgHierarchy h ON e.manager_id = h.employee_id
)
SELECT
    department_id,
    COUNT(*) AS headcount,
    AVG(salary) AS avg_salary,
    MAX(depth_level) AS max_reporting_depth
FROM OrgHierarchy
GROUP BY department_id
HAVING COUNT(*) >= 5
ORDER BY avg_salary DESC;
""",
    """def min_window_substring(s: str, t: str) -> str:
    from collections import Counter
    if not s or not t or len(s) < len(t):
        return ""
    need = Counter(t)
    missing = len(t)
    left = 0
    best_start, best_len = 0, float("inf")
    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1
        while missing == 0:
            if right - left + 1 < best_len:
                best_start = left
                best_len = right - left + 1
            left_ch = s[left]
            need[left_ch] += 1
            if need[left_ch] > 0:
                missing += 1
            left += 1
    return "" if best_len == float("inf") else s[best_start:best_start + best_len]
""",
    """def tarjan_scc(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    index = 0
    indices = [-1] * n
    lowlink = [-1] * n
    on_stack = [False] * n
    stack = []
    sccs = []

    def strongconnect(u: int):
        nonlocal index
        indices[u] = lowlink[u] = index
        index += 1
        stack.append(u)
        on_stack[u] = True
        for v in adj[u]:
            if indices[v] == -1:
                strongconnect(v)
                lowlink[u] = min(lowlink[u], lowlink[v])
            elif on_stack[v]:
                lowlink[u] = min(lowlink[u], indices[v])
        if lowlink[u] == indices[u]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == u:
                    break
            sccs.append(comp)

    for i in range(n):
        if indices[i] == -1:
            strongconnect(i)
    return sccs
""",
    """def edit_distance_space_optimized(word1: str, word2: str) -> int:
    if len(word1) < len(word2):
        word1, word2 = word2, word1
    m, n = len(word1), len(word2)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]
""",
    """def serialize_and_deserialize_bst(root_vals: list[int]) -> list[int]:
    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

    def insert(node, val):
        if node is None:
            return Node(val)
        if val < node.val:
            node.left = insert(node.left, val)
        else:
            node.right = insert(node.right, val)
        return node

    root = None
    for v in root_vals:
        root = insert(root, v)

    preorder = []
    def dfs(node):
        if not node:
            return
        preorder.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return preorder
""",
]

def load_clade_scores():
    raw = np.fromfile(SCORES_BIN, dtype=np.uint32)
    total = N_LAYERS * N_NEURONS
    assert len(raw) == 7 * total, f"Expected {7*total}, got {len(raw)}"
    clade_id    = raw[0*total : 1*total].view(np.int32).reshape(N_LAYERS, N_NEURONS)
    coder_score = raw[1*total : 2*total].view(np.float32).reshape(N_LAYERS, N_NEURONS)
    r1_score    = raw[2*total : 3*total].view(np.float32).reshape(N_LAYERS, N_NEURONS)
    coder_spr   = raw[3*total : 4*total].view(np.float32).reshape(N_LAYERS, N_NEURONS)
    r1_spr      = raw[4*total : 5*total].view(np.float32).reshape(N_LAYERS, N_NEURONS)
    coder_cos   = raw[5*total : 6*total].view(np.float32).reshape(N_LAYERS, N_NEURONS)
    r1_cos      = raw[6*total : 7*total].view(np.float32).reshape(N_LAYERS, N_NEURONS)
    return {
        "clade_id": clade_id,
        "coder_score": coder_score,
        "r1_score": r1_score,
        "coder_spr": coder_spr,
        "r1_spr": r1_spr,
        "coder_cos": coder_cos,
        "r1_cos": r1_cos,
    }

def build_clade_mask_top_k(clade_id, clade_score, frac, highest=True, seed=None):
    """Selects whole monophyletic clades globally across all 48 layers up to `frac` of total neurons."""
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
    target_neurons = int(round(frac * N_LAYERS * N_NEURONS))
    mask = np.zeros((N_LAYERS, N_NEURONS), dtype=bool)
    acc = 0
    for sc, l, c, cnt in items:
        if acc + cnt > target_neurons and acc > 0:
            if abs((acc + cnt) - target_neurons) > abs(acc - target_neurons):
                break
        mask[l, clade_id[l] == c] = True
        acc += cnt
        if acc >= target_neurons:
            break
    return mask

def extract_mlp_state(model):
    """Snapshots all 48 layers' (gate_proj, up_proj, down_proj) to CPU BF16 tensors."""
    state = []
    for l in range(N_LAYERS):
        mlp = model.model.layers[l].mlp
        state.append((
            mlp.gate_proj.weight.detach().cpu().clone(),
            mlp.up_proj.weight.detach().cpu().clone(),
            mlp.down_proj.weight.detach().cpu().clone(),
        ))
    return state

@torch.no_grad()
def apply_mlp_graft(model, src_mlp_state, dst_mlp_state, mask_2d):
    """
    Sets neuron (l, i) to dst_mlp_state if mask_2d[l, i] is True, else src_mlp_state.
    Each neuron i is grafted as a complete coupled residual-stream rank-1 unit:
      gate_proj[i, :], up_proj[i, :], and down_proj[:, i].
    """
    dev = next(model.parameters()).device
    for l in range(N_LAYERS):
        mlp = model.model.layers[l].mlp
        m_row = torch.from_numpy(mask_2d[l]).to(dev)
        g_src, u_src, d_src = src_mlp_state[l]
        g_dst, u_dst, d_dst = dst_mlp_state[l]

        g = g_src.to(dev)
        u = u_src.to(dev)
        d = d_src.to(dev)
        if m_row.any():
            g[m_row] = g_dst.to(dev)[m_row]
            u[m_row] = u_dst.to(dev)[m_row]
            d[:, m_row] = d_dst.to(dev)[:, m_row]
        mlp.gate_proj.weight.copy_(g)
        mlp.up_proj.weight.copy_(u)
        mlp.down_proj.weight.copy_(d)

@torch.no_grad()
def eval_suite_metrics(model, tok, texts, ref_logprobs=None):
    """Computes exact teacher-forcing NLL, Perplexity, Next-Token Accuracy, and KL(P_ref || P_model)."""
    dev = next(model.parameters()).device
    total_nll = 0.0
    total_kl  = 0.0
    total_correct = 0
    total_tokens = 0
    out_logprobs = []

    for idx, txt in enumerate(texts):
        enc = tok(txt, return_tensors="pt").to(dev)
        input_ids = enc["input_ids"]
        logits = model(input_ids=input_ids).logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        log_p = F.log_softmax(logits, dim=-1)

        if ref_logprobs is None:
            out_logprobs.append(log_p.cpu())
        else:
            ref_lp = ref_logprobs[idx].to(dev)
            ref_p  = ref_lp.exp()
            kl = (ref_p * (ref_lp - log_p)).sum(dim=-1).sum().item()
            total_kl += kl

        nll = F.nll_loss(log_p.reshape(-1, log_p.size(-1)), targets.reshape(-1), reduction="sum").item()
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).sum().item()
        n_tok = targets.numel()

        total_nll += nll
        total_correct += correct
        total_tokens += n_tok

    avg_nll = total_nll / total_tokens
    ppl = math.exp(min(avg_nll, 20.0))
    acc = 100.0 * total_correct / total_tokens
    avg_kl = (total_kl / total_tokens) if ref_logprobs is not None else 0.0
    return {"nll": avg_nll, "ppl": ppl, "acc": acc, "kl": avg_kl, "logprobs": out_logprobs}

def main():
    t0 = time.time()
    print("Loading 48-layer phylogenetic clade scores...")
    scores = load_clade_scores()
    clade_id = scores["clade_id"]

    tok = AutoTokenizer.from_pretrained(os.path.join(CKPT_ROOT, "base"))

    print("Loading Base, R1-Distill-14B, and Coder-14B MLP states & initializing GPU models...")
    sys.stdout.flush()

    # Load Base MLP state on CPU first
    m_base_cpu = AutoModelForCausalLM.from_pretrained(
        os.path.join(CKPT_ROOT, "base"),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    base_mlp = extract_mlp_state(m_base_cpu)

    # Evaluate Pure Base on cuda:0 before replacing it with R1-Distill-14B
    m_base_gpu = m_base_cpu.to("cuda:0")
    m_base_gpu.eval()
    base_reas = eval_suite_metrics(m_base_gpu, tok, REASONING_TRACES)
    base_code = eval_suite_metrics(m_base_gpu, tok, CODE_SNIPPETS)
    del m_base_gpu, m_base_cpu
    torch.cuda.empty_cache()

    # Load R1-Distill-14B on cuda:0 and Coder-14B on cuda:1
    m_r1 = AutoModelForCausalLM.from_pretrained(
        os.path.join(CKPT_ROOT, "r1"),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
    )
    m_r1.eval()
    r1_mlp = extract_mlp_state(m_r1)

    m_coder = AutoModelForCausalLM.from_pretrained(
        os.path.join(CKPT_ROOT, "coder"),
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:1"},
    )
    m_coder.eval()
    coder_mlp = extract_mlp_state(m_coder)

    r1_ref_reas    = eval_suite_metrics(m_r1, tok, REASONING_TRACES)
    r1_ref_code    = eval_suite_metrics(m_r1, tok, CODE_SNIPPETS)
    coder_ref_code = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS)
    coder_ref_reas = eval_suite_metrics(m_coder, tok, REASONING_TRACES)

    base_kl_to_r1    = eval_suite_metrics_from_saved(base_reas["logprobs"], r1_ref_reas["logprobs"])
    base_kl_to_coder = eval_suite_metrics_from_saved(base_code["logprobs"], coder_ref_code["logprobs"])

    print("\n==========================================================================================================")
    print("PART 1: CLADE KNOCK-OUT ABLATION ON R1-DISTILL-14B (Revert k% of MLP Clades back to Qwen2.5-14B Base)")
    print("==========================================================================================================")
    print(f"{'Intervention / Clade Selection':<42s} | {'Reverted':>8s} | {'Reasoning PPL':>13s} | {'Next-Tok Acc':>12s} | {'KL(R1 || Edit)':>14s}")
    print("----------------------------------------------------------------------------------------------------------")
    print(f"{'Pure R1-Distill-14B (Unedited Specialist)':<42s} | {'0.0%':>8s} | {r1_ref_reas['ppl']:13.4f} | {r1_ref_reas['acc']:11.2f}% | {0.0:14.5f}")
    print(f"{'Pure Qwen2.5-14B (100% Reverted to Base)':<42s} | {'100.0%':>8s} | {base_reas['ppl']:13.4f} | {base_reas['acc']:11.2f}% | {base_kl_to_r1:14.5f}")
    print("----------------------------------------------------------------------------------------------------------")

    for frac in [0.15, 0.30, 0.50]:
        # 1. Revert Top-k% Topological Quartet-Violated Clades
        mask_viol = build_clade_mask_top_k(clade_id, scores["r1_score"], frac, highest=True)
        apply_mlp_graft(m_r1, r1_mlp, base_mlp, mask_viol)
        res_viol = eval_suite_metrics(m_r1, tok, REASONING_TRACES, ref_logprobs=r1_ref_reas["logprobs"])

        # 2. Revert Bottom-k% Topologically Conserved (Clean) Clades
        mask_clean = build_clade_mask_top_k(clade_id, scores["r1_score"], frac, highest=False)
        apply_mlp_graft(m_r1, r1_mlp, base_mlp, mask_clean)
        res_clean = eval_suite_metrics(m_r1, tok, REASONING_TRACES, ref_logprobs=r1_ref_reas["logprobs"])

        # 3. Revert Random-k% Clades
        mask_rand = build_clade_mask_top_k(clade_id, scores["r1_score"], frac, highest=True, seed=42 + int(frac*100))
        apply_mlp_graft(m_r1, r1_mlp, base_mlp, mask_rand)
        res_rand = eval_suite_metrics(m_r1, tok, REASONING_TRACES, ref_logprobs=r1_ref_reas["logprobs"])

        pct = 100.0 * frac
        print(f"{f'Revert Top-{int(pct)}% Topo-Violated Clades':<42s} | {mask_viol.mean()*100:7.1f}% | {res_viol['ppl']:13.4f} | {res_viol['acc']:11.2f}% | {res_viol['kl']:14.5f}")
        print(f"{f'Revert Random-{int(pct)}% Monophyletic Clades':<42s} | {mask_rand.mean()*100:7.1f}% | {res_rand['ppl']:13.4f} | {res_rand['acc']:11.2f}% | {res_rand['kl']:14.5f}")
        print(f"{f'Revert Bottom-{int(pct)}% Topo-Conserved Clades':<42s} | {mask_clean.mean()*100:7.1f}% | {res_clean['ppl']:13.4f} | {res_clean['acc']:11.2f}% | {res_clean['kl']:14.5f}")
        print("----------------------------------------------------------------------------------------------------------")
        sys.stdout.flush()

    print("\n==========================================================================================================")
    print("PART 2: CLADE KNOCK-OUT ABLATION ON QWEN2.5-CODER-14B (Revert k% of MLP Clades back to Qwen2.5-14B Base)")
    print("==========================================================================================================")
    print(f"{'Intervention / Clade Selection':<42s} | {'Reverted':>8s} | {'Code Suite PPL':>13s} | {'Next-Tok Acc':>12s} | {'KL(Coder||Edit)':>14s}")
    print("----------------------------------------------------------------------------------------------------------")
    print(f"{'Pure Qwen2.5-Coder-14B (Unedited)':<42s} | {'0.0%':>8s} | {coder_ref_code['ppl']:13.4f} | {coder_ref_code['acc']:11.2f}% | {0.0:14.5f}")
    print(f"{'Pure Qwen2.5-14B (100% Reverted to Base)':<42s} | {'100.0%':>8s} | {base_code['ppl']:13.4f} | {base_code['acc']:11.2f}% | {base_kl_to_coder:14.5f}")
    print("----------------------------------------------------------------------------------------------------------")

    for frac in [0.15, 0.30, 0.50]:
        mask_viol = build_clade_mask_top_k(clade_id, scores["coder_score"], frac, highest=True)
        apply_mlp_graft(m_coder, coder_mlp, base_mlp, mask_viol)
        res_viol = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])

        mask_clean = build_clade_mask_top_k(clade_id, scores["coder_score"], frac, highest=False)
        apply_mlp_graft(m_coder, coder_mlp, base_mlp, mask_clean)
        res_clean = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])

        mask_rand = build_clade_mask_top_k(clade_id, scores["coder_score"], frac, highest=True, seed=142 + int(frac*100))
        apply_mlp_graft(m_coder, coder_mlp, base_mlp, mask_rand)
        res_rand = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])

        pct = 100.0 * frac
        print(f"{f'Revert Top-{int(pct)}% Topo-Violated Clades':<42s} | {mask_viol.mean()*100:7.1f}% | {res_viol['ppl']:13.4f} | {res_viol['acc']:11.2f}% | {res_viol['kl']:14.5f}")
        print(f"{f'Revert Random-{int(pct)}% Monophyletic Clades':<42s} | {mask_rand.mean()*100:7.1f}% | {res_rand['ppl']:13.4f} | {res_rand['acc']:11.2f}% | {res_rand['kl']:14.5f}")
        print(f"{f'Revert Bottom-{int(pct)}% Topo-Conserved Clades':<42s} | {mask_clean.mean()*100:7.1f}% | {res_clean['ppl']:13.4f} | {res_clean['acc']:11.2f}% | {res_clean['kl']:14.5f}")
        print("----------------------------------------------------------------------------------------------------------")
        sys.stdout.flush()

    print("\n==========================================================================================================")
    print("PART 3: CLADE-SPARSE POST-TRAINING RETENTION (50% Neuron Budget on R1-Distill-14B & Coder-14B)")
    print("==========================================================================================================")
    print(f"{'Sparse Post-Training Retention (50% Budget)':<42s} | {'Kept %':>8s} | {'Specialist PPL':>13s} | {'Next-Tok Acc':>12s} | {'KL(Spec||Edit)':>14s}")
    print("----------------------------------------------------------------------------------------------------------")
    for label, score_arr, hi, sd in [
        ("[R1] Keep Top-50% Quartet-Violated Clades", scores["r1_score"], True, None),
        ("[R1] Keep Top-50% Euclidean-Drift Clades",  1.0 - scores["r1_cos"], True, None),
        ("[R1] Keep Random-50% Monophyletic Clades",  scores["r1_score"], True, 777),
        ("[R1] Keep Bottom-50% Topo-Conserved Clades", scores["r1_score"], False, None),
    ]:
        keep_mask = build_clade_mask_top_k(clade_id, score_arr, 0.50, highest=hi, seed=sd)
        revert_mask = ~keep_mask
        apply_mlp_graft(m_r1, r1_mlp, base_mlp, revert_mask)
        res = eval_suite_metrics(m_r1, tok, REASONING_TRACES, ref_logprobs=r1_ref_reas["logprobs"])
        print(f"{label:<42s} | {keep_mask.mean()*100:7.1f}% | {res['ppl']:13.4f} | {res['acc']:11.2f}% | {res['kl']:14.5f}")
    print("----------------------------------------------------------------------------------------------------------")
    for label, score_arr, hi, sd in [
        ("[Coder] Keep Top-50% Quartet-Violated Clades", scores["coder_score"], True, None),
        ("[Coder] Keep Top-50% Euclidean-Drift Clades",  1.0 - scores["coder_cos"], True, None),
        ("[Coder] Keep Random-50% Monophyletic Clades",  scores["coder_score"], True, 888),
        ("[Coder] Keep Bottom-50% Topo-Conserved Clades", scores["coder_score"], False, None),
    ]:
        keep_mask = build_clade_mask_top_k(clade_id, score_arr, 0.50, highest=hi, seed=sd)
        revert_mask = ~keep_mask
        apply_mlp_graft(m_coder, coder_mlp, base_mlp, revert_mask)
        res = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])
        print(f"{label:<42s} | {keep_mask.mean()*100:7.1f}% | {res['ppl']:13.4f} | {res['acc']:11.2f}% | {res['kl']:14.5f}")

    print("\n==========================================================================================================")
    print("PART 4: CROSS-SPECIALIST CLADE-SURGICAL GRAFTING (Grafting R1-Distill Rewired Clades into Coder-14B)")
    print("  Goal: Inject k% of R1-Distill-14B's topologically rewired reasoning clades into Qwen2.5-Coder-14B")
    print("        while avoiding Coder-14B's heavily rewired syntax clades!")
    print("==========================================================================================================")
    # Standardize clade scores per model so we can select clades with highest (z_R1 - z_Coder)
    r1_z = (scores["r1_score"] - scores["r1_score"].mean()) / (scores["r1_score"].std() + 1e-8)
    cd_z = (scores["coder_score"] - scores["coder_score"].mean()) / (scores["coder_score"].std() + 1e-8)
    surgical_diff = r1_z - cd_z  # High when R1 rewired the clade AND Coder left the clade relatively conserved

    print(f"{'Coder-14B + 20% R1-Distill Clade Graft':<42s} | {'R1 Graft':>8s} | {'Code KL(C||M)':>13s} | {'Reas PPL':>10s} | {'Reas KL Gain':>12s}")
    print("----------------------------------------------------------------------------------------------------------")
    coder_kl_to_r1 = eval_suite_metrics_from_saved(coder_ref_reas["logprobs"], r1_ref_reas["logprobs"])
    print(f"{'Pure Qwen2.5-Coder-14B (0% R1 Graft)':<42s} | {'0.0%':>8s} | {0.0:13.5f} | {coder_ref_reas['ppl']:10.4f} | {0.0:12.5f}")
    for label, sc_arr, hi, sd in [
        ("Tree-Surgical Graft (High z_R1 - z_Coder)", surgical_diff, True, None),
        ("Random 20% Clade Graft",                    surgical_diff, True, 999),
        ("Anti-Surgical Graft (Low z_R1 - z_Coder)",  surgical_diff, False, None),
    ]:
        g_mask = build_clade_mask_top_k(clade_id, sc_arr, 0.20, highest=hi, seed=sd)
        apply_mlp_graft(m_coder, coder_mlp, r1_mlp, g_mask)
        res_c = eval_suite_metrics(m_coder, tok, CODE_SNIPPETS, ref_logprobs=coder_ref_code["logprobs"])
        res_r = eval_suite_metrics(m_coder, tok, REASONING_TRACES, ref_logprobs=r1_ref_reas["logprobs"])
        kl_gain = coder_kl_to_r1 - res_r["kl"]
        print(f"{label:<42s} | {g_mask.mean()*100:7.1f}% | {res_c['kl']:13.5f} | {res_r['ppl']:10.4f} | {kl_gain:+12.5f}")
    print("==========================================================================================================\n")
    print(f"Completed all surgical evaluations in {time.time() - t0:.1f}s!")

def eval_suite_metrics_from_saved(edit_lps, ref_lps):
    total_kl = 0.0
    total_toks = 0
    for elp, rlp in zip(edit_lps, ref_lps):
        rp = rlp.exp()
        total_kl += (rp * (rlp - elp)).sum().item()
        total_toks += rlp.shape[1]
    return total_kl / total_toks

if __name__ == "__main__":
    main()
