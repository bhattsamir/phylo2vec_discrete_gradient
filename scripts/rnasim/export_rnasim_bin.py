import os, sys, struct, time, random, subprocess
import numpy as np
from test_rnasim_baseline import parse_newick_iterative, induce_subtree, compute_rf_distance

def tree_to_rooted_arrays(children, name, root, leaf_to_id):
    """
    Convert a parsed Newick tree (possibly trifurcating at root) into a strict
    rooted binary tree with leaves 0..N-1 and internal nodes N..2N-2 in post-order
    (so c1_r[u] < u and c2_r[u] < u for all u in N..2N-2, with root = 2N-2).
    """
    N = len(leaf_to_id)
    # Traversal order
    order = []
    st = [root]
    while st:
        u = st.pop()
        order.append(u)
        for c in children[u]:
            st.append(c)

    c1_list = []
    c2_list = []
    mapped = {}
    for u in reversed(order):
        ch = children[u]
        if not ch:
            mapped[u] = leaf_to_id[name[u]]
        else:
            mc = [mapped[c] for c in ch]
            curr = mc[0]
            for nxt in mc[1:]:
                nid = N + len(c1_list)
                c1_list.append(curr)
                c2_list.append(nxt)
                curr = nid
            mapped[u] = curr

    assert len(c1_list) == N - 1, f"Expected {N-1} internals, got {len(c1_list)}"
    c1_r = np.zeros(2 * N - 1, dtype=np.int32)
    c2_r = np.zeros(2 * N - 1, dtype=np.int32)
    c1_r[N:] = np.array(c1_list, dtype=np.int32)
    c2_r[N:] = np.array(c2_list, dtype=np.int32)
    return c1_r, c2_r

def pack_sequences_bits(seqs_list):
    """
    Pack N aligned sequences of length L=1596 into 25 uint64 words per channel:
    [N, 5, 25] uint64 where channels are: 0=A, 1=C, 2=G, 3=T/U, 4=valid(A|C|G|T).
    Vectorized in batches of 4096 sequences for high speed.
    """
    N = len(seqs_list)
    L = len(seqs_list[0])
    W = (L + 63) // 64
    packed = np.zeros((N, 5, W), dtype=np.uint64)
    batch_sz = 4096
    for b_start in range(0, N, batch_sz):
        b_end = min(N, b_start + batch_sz)
        b_len = b_end - b_start
        raw_bytes = "".join(s.upper() for s in seqs_list[b_start:b_end]).encode('ascii')
        arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(b_len, L)
        for w in range(W):
            start = w * 64
            end = min(L, start + 64)
            chunk = arr[:, start:end]
            shifts = np.arange(end - start, dtype=np.uint64)[None, :]
            is_A = (chunk == ord('A')).astype(np.uint64)
            is_C = (chunk == ord('C')).astype(np.uint64)
            is_G = (chunk == ord('G')).astype(np.uint64)
            is_T = ((chunk == ord('T')) | (chunk == ord('U'))).astype(np.uint64)
            is_V = is_A | is_C | is_G | is_T
            packed[b_start:b_end, 0, w] = np.bitwise_or.reduce(is_A << shifts, axis=1)
            packed[b_start:b_end, 1, w] = np.bitwise_or.reduce(is_C << shifts, axis=1)
            packed[b_start:b_end, 2, w] = np.bitwise_or.reduce(is_G << shifts, axis=1)
            packed[b_start:b_end, 3, w] = np.bitwise_or.reduce(is_T << shifts, axis=1)
            packed[b_start:b_end, 4, w] = np.bitwise_or.reduce(is_V << shifts, axis=1)
    return packed, L, W

if __name__ == "__main__":
    base_dir = "/home/sjbhatt/phylo2vec_discrete_gradient/rnasim_data"
    vft_bin = os.path.join(base_dir, "veryfasttree/build/VeryFastTree")

    with open(os.path.join(base_dir, "true_topo.tree")) as f:
        ch_full, nm_full, ln_full, rt_full = parse_newick_iterative(f.read().strip())
    all_leaves = [nm for ch, nm in zip(ch_full, nm_full) if not ch]

    target_sizes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [1000, 5000, 10000, 50000, 100000]
    for N in target_sizes:
        bin_path = os.path.join(base_dir, f"rnasim_{N}.bin")
        if os.path.exists(bin_path):
            print(f"Already exists: {bin_path} ({os.path.getsize(bin_path)/1e6:.2f} MB)")
            continue
        rng = random.Random(42)
        subset = rng.sample(all_leaves, N)
        keep_set = set(subset)
        ch_t, nm_t, ln_t, rt_t = induce_subtree(ch_full, nm_full, ln_full, rt_full, keep_set)

        sub_fa = os.path.join(base_dir, f"sub_{N}.fa")
        if not os.path.exists(sub_fa):
            seqs = {}
            curr_name = None
            curr_seq = []
            with open(os.path.join(base_dir, "aln_dna.fa")) as f:
                for line in f:
                    if line.startswith(">"):
                        if curr_name is not None and curr_name in keep_set:
                            seqs[curr_name] = "".join(curr_seq)
                        curr_name = line[1:].strip().split()[0]
                        curr_seq = []
                    else:
                        if curr_name in keep_set:
                            curr_seq.append(line.strip())
                if curr_name is not None and curr_name in keep_set:
                    seqs[curr_name] = "".join(curr_seq)
            with open(sub_fa, "w") as f:
                for k in subset:
                    f.write(f">{k}\n{seqs[k]}\n")

        # Load sequences in deterministic subset order
        seqs_map = {}
        curr_name = None
        curr_seq = []
        with open(sub_fa) as f:
            for line in f:
                if line.startswith(">"):
                    if curr_name is not None:
                        seqs_map[curr_name] = "".join(curr_seq)
                    curr_name = line[1:].strip().split()[0]
                    curr_seq = []
                else:
                    curr_seq.append(line.strip())
            if curr_name is not None:
                seqs_map[curr_name] = "".join(curr_seq)

        init_nwk_path = os.path.join(base_dir, f"tree_{N}_VFT-InitNJ.nwk")
        me_nwk_path = os.path.join(base_dir, f"tree_{N}_VFT-ME.nwk")
        if not os.path.exists(init_nwk_path):
            t0 = time.time()
            subprocess.run([vft_bin, "-nt", "-nome", "-noml", "-nosupport", "-threads", "64", "-quiet", "-out", init_nwk_path, sub_fa], check=True)
            print(f"Generated VFT-InitNJ for N={N} in {time.time()-t0:.2f}s", flush=True)
        if not os.path.exists(me_nwk_path):
            t0 = time.time()
            subprocess.run([vft_bin, "-nt", "-fastest", "-noml", "-nosupport", "-threads", "64", "-quiet", "-out", me_nwk_path, sub_fa], check=True)
            print(f"Generated VFT-ME for N={N} in {time.time()-t0:.2f}s", flush=True)

        leaf_to_id = {name: i for i, name in enumerate(subset)}
        seqs_list = [seqs_map[name] for name in subset]
        packed, L, W = pack_sequences_bits(seqs_list)

        true_c1, true_c2 = tree_to_rooted_arrays(ch_t, nm_t, rt_t, leaf_to_id)
        with open(init_nwk_path) as f:
            ch_i, nm_i, _, rt_i = parse_newick_iterative(f.read().strip())
        init_c1, init_c2 = tree_to_rooted_arrays(ch_i, nm_i, rt_i, leaf_to_id)
        with open(me_nwk_path) as f:
            ch_m, nm_m, _, rt_m = parse_newick_iterative(f.read().strip())
        me_c1, me_c2 = tree_to_rooted_arrays(ch_m, nm_m, rt_m, leaf_to_id)

        bin_path = os.path.join(base_dir, f"rnasim_{N}.bin")
        with open(bin_path, "wb") as bf:
            bf.write(struct.pack("<iii", N, L, W))
            bf.write(packed.tobytes())
            bf.write(true_c1.tobytes())
            bf.write(true_c2.tobytes())
            bf.write(init_c1.tobytes())
            bf.write(init_c2.tobytes())
            bf.write(me_c1.tobytes())
            bf.write(me_c2.tobytes())
        print(f"Exported {bin_path} ({os.path.getsize(bin_path)/1e6:.2f} MB)")
