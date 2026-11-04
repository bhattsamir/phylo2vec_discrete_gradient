import os, sys, time, random, subprocess
import numpy as np
from test_rnasim_baseline import parse_newick_iterative, induce_subtree, compute_rf_distance

def to_newick(children, name, root):
    """Iterative Newick serializer (topology only)."""
    order = []
    st = [root]
    while st:
        u = st.pop()
        order.append(u)
        for c in children[u]:
            st.append(c)
    s_map = [""] * len(children)
    for u in reversed(order):
        if not children[u]:
            s_map[u] = name[u]
        else:
            s_map[u] = "(" + ",".join(s_map[c] for c in children[u]) + ")"
    return s_map[root] + ";"

def load_fasta_subset(fa_path, keep_set):
    seqs = {}
    curr_name = None
    curr_seq = []
    with open(fa_path) as f:
        for line in f:
            if line.startswith(">"):
                if curr_name is not None and curr_name in keep_set:
                    seqs[curr_name] = "".join(curr_seq)
                    if len(seqs) == len(keep_set):
                        break
                curr_name = line[1:].strip().split()[0]
                curr_seq = []
            else:
                if curr_name in keep_set:
                    curr_seq.append(line.strip())
        if curr_name is not None and curr_name in keep_set and curr_name not in seqs:
            seqs[curr_name] = "".join(curr_seq)
    return seqs

if __name__ == "__main__":
    base_dir = "/home/sjbhatt/phylo2vec_discrete_gradient/rnasim_data"
    vft_bin = os.path.join(base_dir, "veryfasttree/build/VeryFastTree")
    with open(os.path.join(base_dir, "true_topo.tree")) as f:
        nwk_full = f.read().strip()
    ch_full, nm_full, ln_full, rt_full = parse_newick_iterative(nwk_full)
    all_leaves = [nm for ch, nm in zip(ch_full, nm_full) if not ch]

    for N in [1000, 5000, 10000]:
        rng = random.Random(42)
        subset = rng.sample(all_leaves, N)
        keep_set = set(subset)
        ch_t, nm_t, ln_t, rt_t = induce_subtree(ch_full, nm_full, ln_full, rt_full, keep_set)
        seqs = load_fasta_subset(os.path.join(base_dir, "aln_dna.fa"), keep_set)

        sub_fa = os.path.join(base_dir, f"sub_{N}.fa")
        with open(sub_fa, "w") as f:
            for k in subset:
                f.write(f">{k}\n{seqs[k]}\n")

        # 1. Run VeryFastTree -nt -noml (Minimum Evolution only: NJ + ME NNI/SPR)
        out_vft_me = os.path.join(base_dir, f"vft_me_{N}.nwk")
        t0 = time.time()
        subprocess.run([vft_bin, "-nt", "-noml", "-threads", "64", "-quiet", "-out", out_vft_me, sub_fa], check=True)
        t_vft_me = time.time() - t0
        with open(out_vft_me) as f:
            ch_vme, nm_vme, _, rt_vme = parse_newick_iterative(f.read().strip())
        w_vme, tot_splits, nrf_vme = compute_rf_distance(ch_t, nm_t, rt_t, ch_vme, nm_vme, rt_vme)

        # 2. Run VeryFastTree -nt -fast (Fast mode ME + ML)
        out_vft_fast = os.path.join(base_dir, f"vft_fast_{N}.nwk")
        t0 = time.time()
        subprocess.run([vft_bin, "-nt", "-fast", "-threads", "64", "-quiet", "-out", out_vft_fast, sub_fa], check=True)
        t_vft_fast = time.time() - t0
        with open(out_vft_fast) as f:
            ch_vf, nm_vf, _, rt_vf = parse_newick_iterative(f.read().strip())
        w_vf, _, nrf_vf = compute_rf_distance(ch_t, nm_t, rt_t, ch_vf, nm_vf, rt_vf)

        # 3. Run VeryFastTree -nt (Default full ME + ML)
        out_vft_ml = os.path.join(base_dir, f"vft_ml_{N}.nwk")
        t0 = time.time()
        subprocess.run([vft_bin, "-nt", "-threads", "64", "-quiet", "-out", out_vft_ml, sub_fa], check=True)
        t_vft_ml = time.time() - t0
        with open(out_vft_ml) as f:
            ch_vml, nm_vml, _, rt_vml = parse_newick_iterative(f.read().strip())
        w_vml, _, nrf_vml = compute_rf_distance(ch_t, nm_t, rt_t, ch_vml, nm_vml, rt_vml)

        print(f"N={N:6d} | Splits={tot_splits:6d} | "
              f"VFT-ME(-noml): RF={w_vme:5d} ({nrf_vme*100:5.2f}%) in {t_vft_me:5.2f}s | "
              f"VFT-fast: RF={w_vf:5d} ({nrf_vf*100:5.2f}%) in {t_vft_fast:5.2f}s | "
              f"VFT-Full: RF={w_vml:5d} ({nrf_vml*100:5.2f}%) in {t_vft_ml:5.2f}s")
