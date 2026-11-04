import os, sys, json, struct, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
from huggingface_hub import hf_hub_download, hf_hub_url

OUT_DIR = "/tmp/qwen_tree_exp"
SAMPLE_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47]
N_LAYERS = len(SAMPLE_LAYERS)

MODELS = {
    "base": "Qwen/Qwen2.5-14B",
    "coder": "Qwen/Qwen2.5-Coder-14B",
    "r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

# If base_mlp_663k.bf16 exists, slice out the 13 layers and remove the 6.4GB file
old_base_mlp = os.path.join(OUT_DIR, "base_mlp_663k.bf16")
new_base_mlp = os.path.join(OUT_DIR, "base_mlp_13L.bf16")
layer_mlp_bytes = 13824 * 5120 * 2
if os.path.exists(old_base_mlp) and os.path.getsize(old_base_mlp) == 48 * layer_mlp_bytes:
    print("Extracting base_mlp_13L.bf16 from existing base_mlp_663k.bf16...")
    sys.stdout.flush()
    with open(old_base_mlp, "rb") as fin, open(new_base_mlp, "wb") as fout:
        for l in SAMPLE_LAYERS:
            fin.seek(l * layer_mlp_bytes)
            fout.write(fin.read(layer_mlp_bytes))
    os.remove(old_base_mlp)
    print("Extracted base_mlp_13L.bf16 and removed 6.4GB base_mlp_663k.bf16.")
    sys.stdout.flush()

def resolve_shard_cdn(repo_id, shard_files):
    info = {}
    for sf in shard_files:
        url = hf_hub_url(repo_id=repo_id, filename=sf)
        r = requests.get(url, headers={"Range": "bytes=0-7"}, allow_redirects=True, timeout=30)
        r.raise_for_status()
        cdn_url = r.url
        hdr_len = struct.unpack("<Q", r.content)[0]
        r2 = requests.get(cdn_url, headers={"Range": f"bytes=8-{8+hdr_len-1}"}, timeout=30)
        r2.raise_for_status()
        meta = json.loads(r2.content.decode("utf-8"))
        info[sf] = (cdn_url, 8 + hdr_len, meta)
    return info

def fetch_range(cdn_url, base_off, b0, b1):
    for attempt in range(5):
        try:
            r = requests.get(
                cdn_url,
                headers={"Range": f"bytes={base_off+b0}-{base_off+b1-1}"},
                timeout=90,
            )
            r.raise_for_status()
            if len(r.content) == b1 - b0:
                return r.content
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("Failed range fetch")

def download_model(tag, repo_id):
    attn_path = os.path.join(OUT_DIR, f"{tag}_attn_13L.bf16")
    mlp_path  = os.path.join(OUT_DIR, f"{tag}_mlp_13L.bf16")
    lm_path   = os.path.join(OUT_DIR, f"{tag}_lm_head_152k.bf16")

    attn_bytes = N_LAYERS * 5120 * 5120 * 2
    mlp_bytes  = N_LAYERS * 13824 * 5120 * 2
    lm_bytes   = 152064 * 5120 * 2

    have_attn = os.path.exists(attn_path) and os.path.getsize(attn_path) == attn_bytes
    have_mlp  = os.path.exists(mlp_path)  and os.path.getsize(mlp_path)  == mlp_bytes
    have_lm   = os.path.exists(lm_path)   and os.path.getsize(lm_path)   == lm_bytes

    if have_attn and have_mlp and have_lm:
        print(f"[{tag}] Already downloaded and verified.")
        sys.stdout.flush()
        return

    t0 = time.time()
    idx_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
    with open(idx_path) as f:
        wmap = json.load(f)["weight_map"]

    needed_tensors = ["lm_head.weight"]
    for l in SAMPLE_LAYERS:
        needed_tensors.append(f"model.layers.{l}.self_attn.o_proj.weight")
        if not have_mlp:
            needed_tensors.append(f"model.layers.{l}.mlp.down_proj.weight")

    shards = sorted(set(wmap[t] for t in needed_tensors))
    print(f"[{tag}] Resolving CDN URLs for {len(shards)} shards of {repo_id}...")
    sys.stdout.flush()
    s_info = resolve_shard_cdn(repo_id, shards)
    print(f"[{tag}] CDN URLs resolved in {time.time() - t0:.1f}s. Starting parallel range download...")
    sys.stdout.flush()

    attn_chunks = [None] * N_LAYERS
    mlp_chunks  = [None] * N_LAYERS

    def fetch_slot(slot_idx):
        l = SAMPLE_LAYERS[slot_idx]
        if not have_attn:
            ta_name = f"model.layers.{l}.self_attn.o_proj.weight"
            sf_a = wmap[ta_name]
            cdn_a, base_a, meta_a = s_info[sf_a]
            b0, b1 = meta_a[ta_name]["data_offsets"]
            raw_a = fetch_range(cdn_a, base_a, b0, b1)
            arr_a = np.frombuffer(raw_a, dtype=np.uint16).reshape(5120, 5120).T
            attn_chunks[slot_idx] = arr_a.tobytes()

        if not have_mlp:
            tm_name = f"model.layers.{l}.mlp.down_proj.weight"
            sf_m = wmap[tm_name]
            cdn_m, base_m, meta_m = s_info[sf_m]
            b0, b1 = meta_m[tm_name]["data_offsets"]
            raw_m = fetch_range(cdn_m, base_m, b0, b1)
            arr_m = np.frombuffer(raw_m, dtype=np.uint16).reshape(5120, 13824).T
            mlp_chunks[slot_idx] = arr_m.tobytes()
        return l

    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch_slot, i) for i in range(N_LAYERS)]
        for fut in as_completed(futs):
            _ = fut.result()
            done += 1
            if done % 4 == 0 or done == N_LAYERS:
                print(f"  [{tag}] {done}/{N_LAYERS} layer pairs fetched ({time.time() - t0:.1f}s)")
                sys.stdout.flush()

    if not have_attn:
        with open(attn_path, "wb") as f:
            for ch in attn_chunks:
                f.write(ch)
    if not have_mlp:
        with open(mlp_path, "wb") as f:
            for ch in mlp_chunks:
                f.write(ch)

    if not have_lm:
        print(f"  [{tag}] Fetching lm_head.weight (152,064 x 5,120 BF16)...")
        sys.stdout.flush()
        sf_l = wmap["lm_head.weight"]
        cdn_l, base_l, meta_l = s_info[sf_l]
        b0, b1 = meta_l["lm_head.weight"]["data_offsets"]
        raw_l = fetch_range(cdn_l, base_l, b0, b1)
        with open(lm_path, "wb") as f:
            f.write(raw_l)

    print(f"[{tag}] Completed in {time.time() - t0:.1f}s!")
    sys.stdout.flush()

if __name__ == "__main__":
    for tag, repo_id in MODELS.items():
        download_model(tag, repo_id)
