"""Draw mdbr-leaf-ir vectors as 24 x 32 heatmaps for the page (assets/leaf_heatmaps.png).

Each word is embedded like on the page (mean pooling + dense + normalize), then the mean
of 1000 common English words is subtracted: about 95% of every single-word vector is that
shared "an English word" component, which otherwise hides everything else.

Needs torch, transformers, safetensors, huggingface_hub, wordfreq, matplotlib.
  python tools/heatmaps.py
"""

from __future__ import annotations

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

OUT = "assets/leaf_heatmaps.png"
BG, INK, MUTED = "#0f1117", "#e8ecf4", "#8b96a8"
# Diverging map for a dark page: dark neutral at 0, getting brighter toward both extremes.
CMAP = LinearSegmentedColormap.from_list(
    "div", ["#bcd7f7", "#3987e5", "#1d3a63", "#20242e", "#5e2626", "#e66767", "#f7c2c2"]
)


def main() -> None:
    from wordfreq import top_n_list

    d = snapshot_download("MongoDB/mdbr-leaf-ir")
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModel.from_pretrained(d).eval()
    dense = load_file(f"{d}/2_Dense/model.safetensors")

    def embed(words: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(words), 128):
            x = tok(words[i : i + 128], padding=True, return_tensors="pt")
            with torch.no_grad():
                h = model(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            v = v @ dense["linear.weight"].T + dense["linear.bias"]
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.concatenate(out)

    background = [w for w in top_n_list("en", 2000) if w.isalpha() and len(w) > 1][:1000]
    mean = embed(background).mean(0)
    words = ["man", "woman", "king", "queen", "apple"]
    c = {w: v - mean for w, v in zip(words, embed(words))}
    panels = [
        ("man − woman", c["man"] - c["woman"]),
        ("king", c["king"]),
        ("queen", c["queen"]),
        ("king − man + woman", c["king"] - c["man"] + c["woman"]),
        ("apple  (control)", c["apple"]),
    ]
    lim = float(np.percentile(np.abs(np.concatenate([p for _, p in panels])), 99))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig, axes = plt.subplots(3, 2, figsize=(9, 9.6), facecolor=BG)
    for ax, (title, vec) in zip(axes.flat, panels):
        im = ax.imshow(vec.reshape(24, 32), cmap=CMAP, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest")
        ax.set_title(title, loc="left", color=INK, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes.flat[5].axis("off")
    cb = fig.colorbar(im, ax=axes.flat[5], orientation="horizontal", fraction=0.12, pad=0.0, aspect=18)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.set_label(f"value (blue < 0 < red, clipped at ±{lim:.3f})", color=MUTED, fontsize=9)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
    print(OUT)


if __name__ == "__main__":
    main()
