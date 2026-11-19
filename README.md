# Qwen embedding space

Interactive 3D view of the input token embeddings of
[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), running entirely in the browser.

You define three directions with word arithmetic (for example `friend - stranger`,
`male - female`, `rich - poor`). Every word or expression is placed at its projection on
those directions, and a tracked expression such as `mom + gender * k` moves through the
space as you drag the `k` slider, leaving a trail.

## How it works

- **Coordinates.** For an axis direction `d`, a vector `v` is placed at `dot(v, d / ‖d‖)`.
- **Words.** A word's vector is the mean of the embedding rows of its tokens
  (`stranger` → `str` + `anger`), tokenized exactly like the original Qwen tokenizer.
- **Expressions.** `+ - * ( )`, numbers, `k` (the slider), quoted words, and named
  definitions (`gender = male - female`).

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

then open <http://localhost:8000>. The page loads three.js from jsDelivr.

## Deploy

The site is static. On GitHub Pages: Settings → Pages → deploy from the `main` branch,
root folder.

## License

The embedding data in `data/` is derived from Qwen/Qwen3.5-2B and is subject to that
model's license.
