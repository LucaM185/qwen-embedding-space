"""Export a browser-friendly, 4-bit subset of Qwen3.5-2B's token embeddings.

Only tokens made purely of ASCII letters (``[A-Za-z]+``, no leading-space marker)
are kept. Any letters-only word is tokenized into such pieces, so this subset is
enough to embed every word the page accepts.

Outputs (in --out):
  embed.q4.bin    rows of [1024 bytes packed int4 | 64 x float16 scales]
  tokenizer.json  {"tokens": [...row order...], "merges": ["a b", ...]}
  meta.json       shapes and quantization parameters

Usage:
  python tools/export.py --model-dir <HF snapshot dir of Qwen/Qwen3.5-2B> --out data
  python tools/export.py --out data          # downloads via huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np

MODEL_ID = "Qwen/Qwen3.5-2B"
TENSOR_KEY = "model.language_model.embed_tokens.weight"
GROUP = 32
LETTERS = re.compile(r"[A-Za-z]+")


def find_model_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            MODEL_ID, allow_patterns=["*.safetensors", "*.json", "merges.txt"]
        )
    )


def load_embedding_rows(model_dir: Path, rows: np.ndarray) -> np.ndarray:
    """Read selected rows of the bf16 embedding matrix straight from safetensors."""
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        shard = json.loads(index.read_text())["weight_map"][TENSOR_KEY]
    else:
        shard = "model.safetensors"
    path = model_dir / shard
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    info = header[TENSOR_KEY]
    assert info["dtype"] == "BF16", info["dtype"]
    n, d = info["shape"]
    start, end = info["data_offsets"]
    raw = np.memmap(path, dtype=np.uint16, mode="r", offset=8 + header_len + start, shape=(n, d))
    bf16 = np.asarray(raw[rows], dtype=np.uint32)
    return (bf16 << 16).view(np.float32)


def quantize_q4(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric 4-bit, one float16 scale per group of GROUP values."""
    n, d = w.shape
    g = w.reshape(n, d // GROUP, GROUP)
    scale = (np.abs(g).max(axis=2) / 7.0).astype(np.float16)
    s = scale.astype(np.float32)[..., None]
    q = np.clip(np.round(g / np.where(s == 0, 1, s)), -7, 7).astype(np.int8) + 8  # 1..15
    q = q.reshape(n, d).astype(np.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)  # low nibble = even index
    return packed, scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    model_dir = find_model_dir(args.model_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = json.loads((model_dir / "tokenizer.json").read_text())
    vocab: dict[str, int] = tok["model"]["vocab"]
    keep = sorted((i, t) for t, i in vocab.items() if LETTERS.fullmatch(t))
    ids = np.array([i for i, _ in keep], dtype=np.int64)
    tokens = [t for _, t in keep]
    kept = set(tokens)

    merges = []
    for m in tok["model"]["merges"]:
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        if a in kept and b in kept and (a + b) in kept:
            merges.append(f"{a} {b}")

    w = load_embedding_rows(model_dir, ids)
    packed, scale = quantize_q4(w)
    rows = np.concatenate([packed, scale.view(np.uint8)], axis=1)
    (out / "embed.q4.bin").write_bytes(rows.tobytes())

    (out / "tokenizer.json").write_text(
        json.dumps({"tokens": tokens, "merges": merges}, separators=(",", ":"))
    )
    meta = {
        "model_id": MODEL_ID,
        "tensor": TENSOR_KEY,
        "rows": len(tokens),
        "dim": int(w.shape[1]),
        "group": GROUP,
        "row_bytes": int(rows.shape[1]),
        "layout": "per row: dim/2 bytes packed int4 (low nibble = even index, value-8), then dim/group float16 scales",
        "token_filter": "^[A-Za-z]+$",
        "original_ids": ids.tolist(),
    }
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    print(f"{len(tokens)} tokens, {len(merges)} merges, {rows.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
