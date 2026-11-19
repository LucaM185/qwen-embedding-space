# Qwen embedding space

Interactive 3D view of word vectors, running entirely in the browser:
**live demo → <https://lucam185.github.io/qwen-embedding-space/>**

Several embedding spaces, one at a time or side by side (cameras move together). By default
all four are shown: Qwen layer 0 (token table), layer 12, layer 24, and the RAG model.

| space | what a word's vector is | size |
|---|---|---|
| Qwen layer 0 (token table) | rows of [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)'s input embedding table (4-bit), averaged over the word's tokens; any ASCII word; 2048-d | 42 MB |
| Qwen layer 12 / 24 | Qwen3.5-2B's hidden state for `Word: <word>` halfway through (layer 12 of 24) and after the last layer; precomputed for a list of ~100 words, other words are left out of these two views; 2048-d | 0.8 MB per layer |
| [MongoDB/mdbr-leaf-ir](https://huggingface.co/MongoDB/mdbr-leaf-ir) | output of a 23M-parameter retrieval (RAG) model, run with [transformers.js](https://huggingface.co/docs/transformers.js); 768-d, unit length | 46 MB fp16, fetched from Hugging Face on first use |

The token table and layer 0 are what an LLM sees before any processing; layers 12 and 24 show
what it has built from the word by then; the retrieval model is trained so that related
texts land close together, which is what RAG search uses. Each file is downloaded only when
a view that needs it is shown.

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
- **Words (Qwen layers 12 / 24).** Each word is run through Qwen3.5-2B as `Word: <word>`
  (the prefix keeps the word off position 0, where LLMs park huge "attention sink"
  activations) and the hidden state at the word's last token, which has seen the whole word,
  is kept. Only words in [`data/layers/words.json`](data/layers/words.json) are available
  (case-insensitive); edit the list in `tools/export_layers.py` and re-run it to change it.
- **Zoom.** The spaces differ in scale by up to ~250× (layer 0 vs layer 24), so every view
  is zoomed to put its farthest reference point at the same distance; the numbers under
  each view are the unscaled values.

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

`data/layers/` holds `words.json` and one float32 file per layer (`L12.f32.bin`, `L24.f32.bin`).

To regenerate:

```bash
pip install numpy huggingface_hub
python tools/export.py --out data
pip install "transformers>=5" torch
python tools/export_layers.py --out data/layers
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
