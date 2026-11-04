import os, sys, json, struct, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import torch
from huggingface_hub import hf_hub_download, hf_hub_url

OUT_DIR = "/tmp/qwen_tree_exp"
MODELS = {
    "base": "Qwen/Qwen2.5-14B",
    "coder": "Qwen/Qwen2.5-Coder-14B",
    "r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

def get_shard_headers(repo_id, shard_files):
    headers_map = {}
    session = requests.Session()
    for sf in shard_files:
        url = hf_hub_url(repo_id=repo_id, filename=sf)
        r = session.get(url, headers={"Range": "bytes=0-7"}, timeout=30)
        r.raise_for_status()
        hdr_len = struct.unpack("<Q", r.content)[0]
        r2 = session.get(url, headers={"Range": f"bytes=8-{8+hdr_len-1}"}, timeout=30)
        r2.raise_for_status()
        headers_map[sf] = (url, 8 + hdr_len, json.loads(r2.content.decode("utf-8")))
    return headers_map

def fetch_tensor_bytes(url, data_base, start_off, end_off):
    for attempt in range(5):
        try:
            r = requests.get(
                url,
                headers={"Range": f"bytes={data_base+start_off}-{data_base+end_off-1}"},
                timeout=120,
            )
            r.raise_for_status()
            if len(r.content) == end_off - start_off:
                return r.content
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("Failed to fetch tensor bytes")

def download_model_tensors(tag, repo_id):
    mlp_path = os.path.join(OUT_DIR, f"{tag}_mlp_663k.bf16")
    lm_path = os.path.join(OUT_DIR, f"{tag}_lm_head_152k.bf16")
    if os.path.exists(mlp_path) and os.path.getsize(mlp_path) == 663552 * 5120 * 2 and \
       os.path.exists(lm_path) and os.path.getsize(lm_path) == 152064 * 5120 * 2:
        print(f"[{tag}] Already downloaded and verified.")
        return

    t0 = time.time()
    idx_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
    with open(idx_path) as f:
        wmap = json.load(f)["weight_map"]

    needed_tensors = [f"model.layers.{l}.mlp.down_proj.weight" for l in range(48)] + ["lm_head.weight"]
    needed_shards = sorted(set(wmap[t] for t in needed_tensors))
    print(f"[{tag}] Reading headers for {len(needed_shards)} shards from {repo_id}...")
    s_headers = get_shard_headers(repo_id, needed_shards)

    # Pre-allocate output files
    with open(mlp_path, "wb") as f:
        f.truncate(663552 * 5120 * 2)

    def process_layer(l):
        tname = f"model.layers.{l}.mlp.down_proj.weight"
        sf = wmap[tname]
        url, data_base, meta = s_headers[sf]
        b0, b1 = meta[tname]["data_offsets"]
        raw = fetch_tensor_bytes(url, data_base, b0, b1)
        # down_proj.weight has shape [5120, 13824] in BF16; transpose to [13824, 5120] so each neuron is a contiguous 5120-dim row
        t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(5120, 13824).t().contiguous()
        out_bytes = bytes(t.untyped_storage())
        layer_offset = l * 13824 * 5120 * 2
        with open(mlp_path, "r+b") as f:
            f.seek(layer_offset)
            f.write(out_bytes)
        return l

    print(f"[{tag}] Streaming & transposing 48 MLP down_proj layers (663,552 x 5,120 BF16)...")
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_layer, l) for l in range(48)]
        for fut in as_completed(futs):
            _ = fut.result()
            done += 1
            if done % 12 == 0 or done == 48:
                print(f"  [{tag}] {done}/48 MLP layers complete ({time.time() - t0:.1f}s)")
                sys.stdout.flush()

    # Fetch lm_head.weight [152064, 5120]
    print(f"[{tag}] Streaming lm_head.weight (152,064 x 5,120 BF16)...")
    sf = wmap["lm_head.weight"]
    url, data_base, meta = s_headers[sf]
    b0, b1 = meta["lm_head.weight"]["data_offsets"]
    raw_lm = fetch_tensor_bytes(url, data_base, b0, b1)
    with open(lm_path, "wb") as f:
        f.write(raw_lm)
    print(f"[{tag}] Done in {time.time() - t0:.1f}s!")
    sys.stdout.flush()

if __name__ == "__main__":
    tok_path = hf_hub_download(repo_id="Qwen/Qwen2.5-14B", filename="tokenizer.json")
    print(f"Tokenizer cached at: {tok_path}")
    for tag, repo_id in MODELS.items():
        download_model_tensors(tag, repo_id)
