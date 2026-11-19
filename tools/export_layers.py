"""Precompute Qwen3.5-2B hidden states for a small fixed word list.

Each word is fed as  "Word:" + " <word>"  (a short fixed prefix keeps the word off
position 0, where LLMs park huge "attention sink" activations) and we keep the hidden
state at the word's last token, which has attended to the whole word.

Outputs in --out (default data/layers):
  words.json       the vocabulary, row order
  L<n>.f32.bin     float32 rows, little endian, one per word
  meta.json        layers and shapes

Needs transformers >= 5 (Qwen3.5 support) and torch.
  python tools/export_layers.py --model-dir <snapshot dir or Qwen/Qwen3.5-2B>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX = "Word:"
WORDS = """
king queen prince princess emperor empress royal throne crown palace
man woman boy girl male female gentleman lady husband wife
father mother dad mom son daughter brother sister uncle aunt
grandfather grandmother grandson granddaughter baby child kid teenager adult
old young elderly ancient new modern
friend stranger enemy neighbor
rich poor wealthy luxury cheap expensive money
hot cold warm cool fire ice sun snow water steam
big small huge tiny elephant mouse whale ant
fast slow rocket turtle car bicycle
happy sad love hate good bad
doctor nurse teacher student soldier farmer
dog cat horse lion tiger wolf
city village house castle
""".split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--out", default="data/layers")
    ap.add_argument("--layers", default="12,24")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Without the fla/causal-conv1d kernels, transformers' PyTorch fallback for the linear
    # attention layers pads every sequence to 64-token chunks. Our inputs are 3-8 tokens, so
    # the token-by-token recurrent form (same math; outputs match to ~1e-6) is far lighter.
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen3_5

    qwen3_5.torch_chunk_gated_delta_rule = qwen3_5.torch_recurrent_gated_delta_rule

    words = list(dict.fromkeys(WORDS))
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    prefix = tok(PREFIX, add_special_tokens=False)["input_ids"]
    seqs = [prefix + tok(" " + w, add_special_tokens=False)["input_ids"] for w in words]
    width = max(map(len, seqs))
    ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs])  # left pad
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs])
    with torch.no_grad():
        # model.model skips the LM head (logits over the 248k vocabulary are not needed).
        hs = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states

    meta = {
        "model_id": "Qwen/Qwen3.5-2B",
        "input": f"{PREFIX} <word>",
        "pooling": "hidden state at the word's last token",
        "words": len(words),
        "dim": model.config.hidden_size,
        "num_layers": model.config.num_hidden_layers,
        "layers": {},
    }
    (out / "words.json").write_text(json.dumps(words))
    for li in layers:
        rows = hs[li][:, -1].float().numpy().astype("<f4")
        (out / f"L{li}.f32.bin").write_bytes(rows.tobytes())
        meta["layers"][str(li)] = {"file": f"L{li}.f32.bin"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(words)} words, layers {layers}")


if __name__ == "__main__":
    main()
