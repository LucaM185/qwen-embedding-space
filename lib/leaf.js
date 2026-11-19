// MongoDB/mdbr-leaf-ir: a small retrieval (RAG) embedding model run in the browser with
// transformers.js. Each word or phrase is embedded as a document (no query prompt) and the
// model's own pooling + dense + normalize head gives a unit-length 768-d vector.
import { AutoModel, AutoTokenizer } from "@huggingface/transformers";
import { ParseError } from "./engine.js";

export const LEAF_MODEL_ID = "MongoDB/mdbr-leaf-ir";

export class LeafEmbeddings {
  static async load(onProgress) {
    const files = new Map();
    const progress_callback = (p) => {
      if (p.status === "progress" && p.total) {
        files.set(p.file, { loaded: p.loaded, total: p.total });
        let loaded = 0;
        let total = 0;
        for (const f of files.values()) {
          loaded += f.loaded;
          total += f.total;
        }
        onProgress?.(loaded, total);
      }
    };
    const [tokenizer, model] = await Promise.all([
      AutoTokenizer.from_pretrained(LEAF_MODEL_ID, { progress_callback }),
      // fp16 matches the fp32 reference to ~1e-4 on projections; q8 is off by ~0.04.
      AutoModel.from_pretrained(LEAF_MODEL_ID, { dtype: "fp16", progress_callback }),
    ]);
    return new LeafEmbeddings(tokenizer, model);
  }

  constructor(tokenizer, model) {
    this.tokenizer = tokenizer;
    this.model = model;
    this.cache = new Map();
  }

  async prepare(words) {
    const missing = [...new Set(words)].filter((w) => !this.cache.has(w) && w.trim());
    for (let i = 0; i < missing.length; i += 32) {
      const batch = missing.slice(i, i + 32);
      const inputs = await this.tokenizer(batch, { padding: true, truncation: true });
      const { sentence_embedding } = await this.model(inputs);
      const [n, dim] = sentence_embedding.dims;
      const data = sentence_embedding.data;
      for (let j = 0; j < n; j++) {
        this.cache.set(batch[j], Float32Array.from(data.subarray(j * dim, (j + 1) * dim)));
      }
    }
  }

  word(word) {
    const vec = this.cache.get(word);
    if (!vec) throw new ParseError(`"${word}" has not been embedded yet`);
    return { vec, pieces: [word] };
  }
}
