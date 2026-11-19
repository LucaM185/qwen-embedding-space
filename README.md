# Qwen embedding space

Interactive 3D view of word vectors, running entirely in the browser:
**live demo → <https://lucam185.github.io/qwen-embedding-space/>**

Two kinds of embeddings, switchable or side by side:

| model | what a word's vector is | size |
|---|---|---|
| [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) token embeddings | the LLM's input embedding rows (4-bit), averaged over the word's tokens; 2048-d | 42 MB, in this repo |
| [MongoDB/mdbr-leaf-ir](https://huggingface.co/MongoDB/mdbr-leaf-ir) | output of a 23M-parameter retrieval (RAG) model, run with [transformers.js](https://huggingface.co/docs/transformers.js); 768-d, unit length | 46 MB fp16, fetched from Hugging Face on first use |

The first is what an LLM sees before any layer runs; the second is a model trained so that
related texts land close together, which is what RAG search uses.

You define three directions with word arithmetic (for example `friend - stranger`,
`male - female`, `rich - poor`). Every word or expression is placed at its projection on
those directions, and a tracked expression such as `mom + gender * k` moves through the
space as you drag the `k` slider, leaving a trail.

## How it works

- **Coordinates.** For an axis direction `d`, a vector `v` is placed at `dot(v, d / ‖d‖)`.
- **Words (Qwen).** The mean of the embedding rows of the word's tokens
  (`stranger` → `str` + `anger`), tokenized exactly like the original Qwen tokenizer.
  ASCII letters only.
- **Words (mdbr-leaf-ir).** Each word or quoted phrase is embedded as a document (no query
  prompt). The fp16 ONNX model matches the fp32 sentence-transformers output to about 1e-4
  on projections; the int8 one was off by ~0.04, so it is not used.
- **Expressions.** `+ - * ( )`, numbers, `k` (the slider), quoted words or phrases, and
  named definitions (`gender = male - female`).
- **Side by side.** The two models live on very different scales, so each view is scaled to
  put its farthest reference point at the same distance; the numbers under each view are
  the unscaled values.

## Data

`data/` holds a 4-bit subset of the embedding matrix (42 MB):

| file | contents |
|---|---|
| `embed.q4.bin` | 36,555 rows × 2048 dims, symmetric int4 with one float16 scale per 32 values |
| `tokenizer.json` | the kept tokens (row order) and the BPE merges between them |
| `meta.json` | shapes, quantization layout, and the original token id of every row |

Only tokens made of plain ASCII letters (`^[A-Za-z]+$`) are kept. Any letters-only word
tokenizes into such pieces, so this subset covers every word the page accepts; words with
accents, digits or symbols are rejected. The JS tokenizer matches the Hugging Face tokenizer
on all 295,974 test words (the macOS word list plus capitalized, uppercase and random
strings).

Quantization error: mean cosine similarity between the original and 4-bit rows is 0.995,
and projections differ from full precision by less than 0.01 on the default example.

To regenerate:

```bash
pip install numpy huggingface_hub
python tools/export.py --out data
```

## Run locally

Any static file server works:

```bash
python -m http.server 8000
```

then open <http://localhost:8000>. The page loads three.js and transformers.js from
jsDelivr, and the mdbr-leaf-ir weights from Hugging Face.

## Deploy

The site is static. On GitHub Pages: Settings → Pages → deploy from the `main` branch,
root folder.

## License

The embedding data in `data/` is derived from Qwen/Qwen3.5-2B and is subject to that
model's license. mdbr-leaf-ir is Apache-2.0 and is not redistributed here.
