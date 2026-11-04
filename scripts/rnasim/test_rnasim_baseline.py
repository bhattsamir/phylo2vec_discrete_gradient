import os, sys, time, random, subprocess

def parse_newick_iterative(nwk_str):
    """Iterative O(N) Newick parser returning (children, name, length, root)."""
    children = [[]]
    name = [""]
    length = [0.0]
    parent = [-1]
    curr = 0
    i = 0
    n = len(nwk_str)
    while i < n:
        ch = nwk_str[i]
        if ch == '(':
            nid = len(children)
            children.append([])
            name.append("")
            length.append(0.0)
            parent.append(curr)
            children[curr].append(nid)
            curr = nid
            i += 1
        elif ch == ',':
            curr = parent[curr]
            nid = len(children)
            children.append([])
            name.append("")
            length.append(0.0)
            parent.append(curr)
            children[curr].append(nid)
            curr = nid
            i += 1
        elif ch == ')':
            curr = parent[curr]
            i += 1
        elif ch == ';':
            break
        elif ch.isspace():
            i += 1
        else:
            j = i
            while j < n and nwk_str[j] not in '(),;':
                j += 1
            token = nwk_str[i:j].strip()
            if ':' in token:
                nm, ln = token.split(':', 1)
                name[curr] = nm.strip()
                try:
                    length[curr] = float(ln.strip())
                except ValueError:
                    length[curr] = 0.0
            else:
                name[curr] = token
            i = j
    return children, name, length, 0

def induce_subtree(children, name, length, root, keep_set):
    """Bottom-up O(N) induced binary subtree on keep_set."""
    order = []
    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for c in children[u]:
            stack.append(c)
    # Post-order mapping
    mapped = {}
    new_children = []
    new_name = []
    new_len = []
    for u in reversed(order):
        if not children[u]:
            if name[u] in keep_set:
                nid = len(new_children)
                new_children.append([])
                new_name.append(name[u])
                new_len.append(length[u])
                mapped[u] = nid
        else:
            kept_c = [mapped[c] for c in children[u] if c in mapped]
            if len(kept_c) == 1:
                only = kept_c[0]
                new_len[only] += length[u]
                mapped[u] = only
            elif len(kept_c) >= 2:
                # If multifurcating (>2), binarize with 0-length edges
                curr_c = kept_c[0]
                for nxt_c in kept_c[1:-1]:
                    nid = len(new_children)
                    new_children.append([curr_c, nxt_c])
                    new_name.append("")
                    new_len.append(0.0)
                    curr_c = nid
                nid = len(new_children)
                new_children.append([curr_c, kept_c[-1]])
                new_name.append("")
                new_len.append(length[u])
                mapped[u] = nid
    return new_children, new_name, new_len, mapped[root]

def compute_rf_distance(c1, nm1, r1, c2, nm2, r2):
    """Exact O(N) unrooted Robinson-Foulds distance using 64-bit universal XOR hashing."""
    # Collect leaf names
    leaves = [nm for ch, nm in zip(c1, nm1) if not ch]
    rng = random.Random(123456789)
    hmap = {}
    xor_all = 0
    for x in leaves[:-1]:
        h = rng.getrandbits(64)
        hmap[x] = h
        xor_all ^= h
    hmap[leaves[-1]] = xor_all

    def get_splits(children, name, root):
        order = []
        st = [root]
        while st:
            u = st.pop()
            order.append(u)
            for c in children[u]:
                st.append(c)
        nhash = [0] * len(children)
        for u in reversed(order):
            if not children[u]:
                nhash[u] = hmap[name[u]]
            else:
                x = 0
                for c in children[u]:
                    x ^= nhash[c]
                nhash[u] = x
        splits = set()
        for u in order:
            if u == root or not children[u]:
                continue
            ifnh = nhash[u]
            if ifnh != 0:
                splits.add(ifnh)
        return splits

    s1 = get_splits(c1, nm1, r1)
    s2 = get_splits(c2, nm2, r2)
    common = len(s1 & s2)
    total1 = len(s1)
    total2 = len(s2)
    wrong = total1 - common
    return wrong, total1, wrong / max(1, total1)

if __name__ == "__main__":
    t0 = time.time()
    with open("/home/sjbhatt/phylo2vec_discrete_gradient/rnasim_data/true_topo.tree") as f:
        nwk = f.read().strip()
    ch_full, nm_full, ln_full, rt_full = parse_newick_iterative(nwk)
    all_leaves = [nm for ch, nm in zip(ch_full, nm_full) if not ch]
    print(f"Parsed 200k tree in {time.time()-t0:.2f}s, leaves={len(all_leaves)}")
