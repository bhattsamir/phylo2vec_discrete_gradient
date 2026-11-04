import json
import os

TOKENIZER_PATH = "/home/sjbhatt/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9/tokenizer.json"
SAMPLE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47]

def load_id_to_token():
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok = json.load(f)
    vocab = tok["model"]["vocab"]
    id2tok = {v: k for k, v in vocab.items()}
    for added in tok.get("added_tokens", []):
        id2tok[added["id"]] = added["content"]
    return id2tok

def clean_tok(s):
    if s is None:
        return "<UNK>"
    return repr(s.replace("Ġ", " ").replace("Ċ", "\\n").replace("ĉ", "\\t"))

def decode_vocab_file(path, id2tok, label):
    if not os.path.exists(path):
        return
    print(f"\n=== {label}: Top Quartet-Violated Token Clades & Migrating Tokens ({path}) ===")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    clades = [l for l in lines if l.startswith("CLADE")]
    migrates = [l for l in lines if l.startswith("MIGRATE")]

    print("Top 10 Rewired Token Clades:")
    for line in clades[:10]:
        parts = dict(kv.split("=", 1) for kv in line.split()[1:])
        leaf_ids = [int(x) for x in parts["leaves"].split(",")[:10]]
        toks = [f"{lid}:{clean_tok(id2tok.get(lid))}" for lid in leaf_ids]
        print(f"  Rank {parts['rank']:>2s} (size={parts['size']:>4s}, viols={parts['viols']:>3s}, margin={parts['margin']}): " + ", ".join(toks))

    # Sort migrating tokens by lowest self_cos (largest direct vector rotation + topological jump)
    mig_parsed = []
    for line in migrates:
        parts = dict(kv.split("=", 1) for kv in line.split()[1:])
        lid = int(parts["leaf"])
        sc = float(parts["self_cos"])
        mig_parsed.append((sc, lid, parts["old_v"], parts["new_v"]))
    mig_parsed.sort(key=lambda x: x[0])
    print("Top 12 Migrating Vocabulary Tokens (sorted by largest cosine shift among Phylo2Vec branch jumpers):")
    for sc, lid, ov, nv in mig_parsed[:12]:
        print(f"  Token {lid:>6d} {clean_tok(id2tok.get(lid)):<24s} | self_cos={sc:.5f} | v_base={ov} -> v_target={nv}")

def decode_layer_clades(path, width, label):
    if not os.path.exists(path):
        return
    print(f"\n=== {label}: Layer Distribution of Top Rewired Clades ({path}) ===")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and line.startswith("CLADE")]
    for line in lines[:8]:
        parts = dict(kv.split("=", 1) for kv in line.split()[1:])
        leaf_ids = [int(x) for x in parts["leaves"].split(",")[:12]]
        layer_tags = [f"L{SAMPLE_LAYERS[lid // width]}:{lid % width}" for lid in leaf_ids]
        print(f"  Rank {parts['rank']:>2s} (size={parts['size']:>4s}, viols={parts['viols']:>3s}, margin={parts['margin']}): " + ", ".join(layer_tags))

if __name__ == "__main__":
    id2tok = load_id_to_token()
    decode_vocab_file("/tmp/qwen_tree_exp/lm_head_rewired_coder.txt", id2tok, "Coder-14B vs Base (lm_head)")
    decode_vocab_file("/tmp/qwen_tree_exp/lm_head_rewired_r1.txt", id2tok, "R1-Distill-14B vs Base (lm_head)")
    decode_layer_clades("/tmp/qwen_tree_exp/all_attn_rewired_coder.txt", 5120, "Coder-14B vs Base (ALL 13 ATTNs)")
    decode_layer_clades("/tmp/qwen_tree_exp/all_attn_rewired_r1.txt", 5120, "R1-Distill-14B vs Base (ALL 13 ATTNs)")
    decode_layer_clades("/tmp/qwen_tree_exp/all_mlp_rewired_coder.txt", 13824, "Coder-14B vs Base (ALL 13 MLPs)")
    decode_layer_clades("/tmp/qwen_tree_exp/all_mlp_rewired_r1.txt", 13824, "R1-Distill-14B vs Base (ALL 13 MLPs)")

