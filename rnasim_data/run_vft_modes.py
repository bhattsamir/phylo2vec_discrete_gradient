import os, time, random, subprocess
from test_rnasim_baseline import parse_newick_iterative, induce_subtree, compute_rf_distance

base_dir = "/home/sjbhatt/phylo2vec_discrete_gradient/rnasim_data"
vft_bin = os.path.join(base_dir, "veryfasttree/build/VeryFastTree")

with open(os.path.join(base_dir, "true_topo.tree")) as f:
    ch_full, nm_full, ln_full, rt_full = parse_newick_iterative(f.read().strip())
all_leaves = [nm for ch, nm in zip(ch_full, nm_full) if not ch]

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

for N in [1000, 5000, 10000]:
    rng = random.Random(42)
    subset = rng.sample(all_leaves, N)
    keep_set = set(subset)
    ch_t, nm_t, ln_t, rt_t = induce_subtree(ch_full, nm_full, ln_full, rt_full, keep_set)
    sub_fa = os.path.join(base_dir, f"sub_{N}.fa")
    if not os.path.exists(sub_fa):
        seqs = load_fasta_subset(os.path.join(base_dir, "aln_dna.fa"), keep_set)
        with open(sub_fa, "w") as f:
            for k in subset:
                f.write(f">{k}\n{seqs[k]}\n")

    results = {}
    for label, flags in [
        ("VFT-InitNJ (-nome -noml)", ["-nt", "-nome", "-noml", "-nosupport", "-threads", "64", "-quiet"]),
        ("VFT-ME (-noml)",           ["-nt", "-noml", "-nosupport", "-threads", "64", "-quiet"]),
        ("VFT-Fastest-ME",           ["-nt", "-fastest", "-noml", "-nosupport", "-threads", "64", "-quiet"]),
        ("VFT-Full (ME+ML)",         ["-nt", "-nosupport", "-threads", "64", "-quiet"]),
    ]:
        out_nwk = os.path.join(base_dir, f"tree_{N}_{label.split()[0]}.nwk")
        t0 = time.time()
        subprocess.run([vft_bin] + flags + ["-out", out_nwk, sub_fa], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dt = time.time() - t0
        with open(out_nwk) as f:
            ch_v, nm_v, _, rt_v = parse_newick_iterative(f.read().strip())
        w, tot, nrf = compute_rf_distance(ch_t, nm_t, rt_t, ch_v, nm_v, rt_v)
        results[label] = (w, tot, nrf, dt)

    print(f"\n=== RNASim N = {N} (Total Internal Splits = {results['VFT-ME (-noml)'][1]}) ===")
    for label, (w, tot, nrf, dt) in results.items():
        print(f"  {label:26s} : Wrong Splits = {w:5d} / {tot:5d} (nRF = {nrf*100:5.2f}%, Recov = {(1-nrf)*100:5.2f}%) | Time = {dt:6.2f} s")
