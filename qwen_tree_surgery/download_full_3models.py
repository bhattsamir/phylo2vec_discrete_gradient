import os, sys, json, struct, shutil, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import requests
from huggingface_hub import hf_hub_download, hf_hub_url
from safetensors import safe_open

OUT_DIR = "/tmp/qwen_tree_exp"
CKPT_ROOT = "/tmp/qwen_tree_exp/checkpoints"

MODELS = {
    "base": "Qwen/Qwen2.5-14B",
    "coder": "Qwen/Qwen2.5-Coder-14B",
    "r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

META_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB chunks

def download_shard_fast(repo_id, filename, dest_path):
    url = hf_hub_url(repo_id=repo_id, filename=filename)
    # Resolve redirect and get total size via Range: bytes=0-0
    r = requests.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=30)
    r.raise_for_status()
    cdn_url = r.url
    content_range = r.headers.get("Content-Range", "")
    if "/" not in content_range:
        raise RuntimeError(f"Missing Content-Range for {filename}: {content_range}")
    total_size = int(content_range.split("/")[-1])

    if os.path.exists(dest_path) and os.path.getsize(dest_path) == total_size:
        return total_size, 0.0

    tmp_path = dest_path + ".part"
    with open(tmp_path, "wb") as f:
        f.truncate(total_size)

    ranges = []
    for start in range(0, total_size, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_size)
        ranges.append((start, end))

    t0 = time.time()
    def fetch_and_write(rng):
        b0, b1 = rng
        for attempt in range(6):
            try:
                resp = requests.get(
                    cdn_url,
                    headers={"Range": f"bytes={b0}-{b1-1}"},
                    timeout=90,
                )
                resp.raise_for_status()
                data = resp.content
                if len(data) == b1 - b0:
                    fd = os.open(tmp_path, os.O_WRONLY)
                    try:
                        os.pwrite(fd, data, b0)
                    finally:
                        os.close(fd)
                    return b1 - b0
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Failed chunk {b0}-{b1}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch_and_write, rng) for rng in ranges]
        for fut in as_completed(futs):
            _ = fut.result()

    os.replace(tmp_path, dest_path)
    return total_size, time.time() - t0

def process_model(tag, repo_id):
    model_dir = os.path.join(CKPT_ROOT, tag)
    os.makedirs(model_dir, exist_ok=True)
    t_start = time.time()

    for mf in META_FILES:
        try:
            src = hf_hub_download(repo_id=repo_id, filename=mf)
            dst = os.path.join(model_dir, mf)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        except Exception:
            pass

    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(idx_path, "r") as f:
        wmap = json.load(f)["weight_map"]
    shards = sorted(set(wmap.values()))

    print(f"[{tag}] Downloading {len(shards)} shards of {repo_id} into {model_dir}...")
    sys.stdout.flush()
    for i, sf in enumerate(shards, 1):
        dst = os.path.join(model_dir, sf)
        sz, dt = download_shard_fast(repo_id, sf, dst)
        mb = sz / (1024 * 1024)
        spd = mb / max(dt, 1e-3) if dt > 0 else 0.0
        print(f"  [{tag}] Shard {i}/{len(shards)}: {sf} ({mb:.1f} MB) in {dt:.1f}s ({spd:.1f} MB/s)")
        sys.stdout.flush()

    # Build 48-layer transposed down_proj BF16 file for fast GPU tree/quartet construction
    mlp48_path = os.path.join(OUT_DIR, f"{tag}_mlp_48L.bf16")
    expected_bytes = 48 * 13824 * 5120 * 2
    if not (os.path.exists(mlp48_path) and os.path.getsize(mlp48_path) == expected_bytes):
        print(f"  [{tag}] Extracting 48-layer transposed down_proj matrix to {mlp48_path}...")
        sys.stdout.flush()
        with open(mlp48_path, "wb") as fout:
            for l in range(48):
                tname = f"model.layers.{l}.mlp.down_proj.weight"
                sf = os.path.join(model_dir, wmap[tname])
                with open(sf, "rb") as f_s:
                    hdr_len = struct.unpack("<Q", f_s.read(8))[0]
                    meta = json.loads(f_s.read(hdr_len).decode("utf-8"))
                    b0, b1 = meta[tname]["data_offsets"]
                    f_s.seek(8 + hdr_len + b0)
                    raw = f_s.read(b1 - b0)
                arr = np.frombuffer(raw, dtype=np.uint16).reshape(5120, 13824).T
                fout.write(arr.tobytes())
    print(f"[{tag}] Ready in {time.time() - t_start:.1f}s!")
    sys.stdout.flush()

if __name__ == "__main__":
    # Clean up old 13L temporary files to keep /tmp tidy
    for old_f in os.listdir(OUT_DIR):
        if old_f.endswith("_13L.bf16") or old_f.endswith("_152k.bf16"):
            try:
                os.remove(os.path.join(OUT_DIR, old_f))
            except Exception:
                pass
    for tag, repo_id in MODELS.items():
        process_model(tag, repo_id)
