// Embedding lookup, BPE tokenizer and expression language for the 3D projection view.
// Runs in the browser and in Node (no DOM access here).

export class ParseError extends Error {}

const LETTERS = /^[A-Za-z]+$/;

// --- Tokenizer -------------------------------------------------------------
// Byte-level BPE restricted to letters-only tokens. For an ASCII-letters word the
// Qwen pre-tokenizer yields a single chunk and byte-level mapping is the identity,
// so plain BPE over the kept merges reproduces the original tokenizer exactly.
export class Tokenizer {
  constructor({ tokens, merges }) {
    this.tokens = tokens;
    this.index = new Map(tokens.map((t, i) => [t, i]));
    this.ranks = new Map(merges.map((m, i) => [m, i]));
    this.cache = new Map();
  }

  tokenize(word) {
    if (!LETTERS.test(word)) {
      throw new ParseError(`"${word}": only plain ASCII letters are supported`);
    }
    const hit = this.cache.get(word);
    if (hit) return hit;
    let parts = [...word];
    while (parts.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const a = parts[best];
      const b = parts[best + 1];
      const merged = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
          merged.push(a + b);
          i++;
        } else {
          merged.push(parts[i]);
        }
      }
      parts = merged;
    }
    const rows = parts.map((p) => {
      const r = this.index.get(p);
      if (r === undefined) throw new ParseError(`"${word}": piece "${p}" not in vocabulary`);
      return r;
    });
    const out = { pieces: parts, rows };
    this.cache.set(word, out);
    return out;
  }
}

// Fixed vocabulary of precomputed word vectors (one row per word, case-insensitive).
export class WordTable {
  constructor(words, size) {
    this.index = new Map(words.map((w, i) => [w, i]));
    this.size = size;
  }

  tokenize(word) {
    const r = this.index.get(word.toLowerCase());
    if (r === undefined) {
      throw new ParseError(`"${word}" is not in the ${this.size}-word list`);
    }
    return { pieces: [word], rows: [r] };
  }
}

// Precomputed float32 word vectors (row i = words[i]).
export class DenseWords {
  constructor(buffer, dim, table) {
    this.data = new Float32Array(buffer);
    this.dim = dim;
    this.table = table;
  }

  async prepare() {
    /* all data is already in memory */
  }

  word(word) {
    const r = this.table.tokenize(word).rows[0];
    return { vec: this.data.subarray(r * this.dim, (r + 1) * this.dim), pieces: [word] };
  }
}

// --- Embeddings --------------------------------------------------------------
function f16(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export class Embeddings {
  constructor(buffer, meta, tokenizer) {
    this.bytes = new Uint8Array(buffer);
    this.view = new DataView(buffer);
    this.dim = meta.dim;
    this.group = meta.group;
    this.rowBytes = meta.row_bytes;
    this.tokenizer = tokenizer;
    this.rowCache = new Map();
    this.wordCache = new Map();
  }

  row(r) {
    let v = this.rowCache.get(r);
    if (v) return v;
    const { dim, group, rowBytes, bytes, view } = this;
    const base = r * rowBytes;
    const scaleBase = base + dim / 2;
    v = new Float32Array(dim);
    for (let gi = 0; gi < dim / group; gi++) {
      const s = f16(view.getUint16(scaleBase + gi * 2, true));
      for (let j = gi * group; j < (gi + 1) * group; j += 2) {
        const b = bytes[base + (j >> 1)];
        v[j] = ((b & 15) - 8) * s;
        v[j + 1] = ((b >> 4) - 8) * s;
      }
    }
    this.rowCache.set(r, v);
    return v;
  }

  async prepare() {
    /* all data is already in memory */
  }

  // Same as the Python playground: mean of the word's token vectors.
  word(word) {
    let hit = this.wordCache.get(word);
    if (hit) return hit;
    const { pieces, rows } = this.tokenizer.tokenize(word);
    const v = new Float32Array(this.dim);
    for (const r of rows) {
      const x = this.row(r);
      for (let i = 0; i < v.length; i++) v[i] += x[i];
    }
    for (let i = 0; i < v.length; i++) v[i] /= rows.length;
    hit = { vec: v, pieces };
    this.wordCache.set(word, hit);
    return hit;
  }
}

// --- Vector helpers ----------------------------------------------------------
const add = (a, b) => a.map((x, i) => x + b[i]);
const sub = (a, b) => a.map((x, i) => x - b[i]);
const scale = (a, s) => a.map((x) => x * s);
export const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};
export const norm = (a) => Math.sqrt(dot(a, a));

// --- Expression language (port of the playground grammar) ---------------------
const NUM_RE = /^-?\d+(?:\.\d+)?/;
const WORD_RE = /^[a-zA-Z_][a-zA-Z0-9_]*/;

function lex(expr) {
  const toks = [];
  let i = 0;
  while (i < expr.length) {
    const c = expr[i];
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    if ("+-*/()".includes(c)) {
      toks.push({ kind: "op", value: c });
      i++;
      continue;
    }
    if (c === '"' || c === "'") {
      const j = expr.indexOf(c, i + 1);
      if (j < 0) throw new ParseError("Unclosed string");
      toks.push({ kind: "word", value: expr.slice(i + 1, j) });
      i = j + 1;
      continue;
    }
    const rest = expr.slice(i);
    let m = rest.match(NUM_RE);
    if (m) {
      toks.push({ kind: "num", value: parseFloat(m[0]) });
      i += m[0].length;
      continue;
    }
    m = rest.match(WORD_RE);
    if (m) {
      toks.push(m[0] === "k" ? { kind: "k", value: "k" } : { kind: "word", value: m[0] });
      i += m[0].length;
      continue;
    }
    throw new ParseError(`Unexpected character "${c}"`);
  }
  return toks;
}

class Parser {
  constructor(toks, k, env, emb, words) {
    Object.assign(this, { toks, k, env, emb, words, pos: 0 });
  }
  peek() {
    return this.toks[this.pos];
  }
  eat(kind, value) {
    const t = this.peek();
    if (!t) throw new ParseError("Unexpected end of expression");
    if (kind && t.kind !== kind) throw new ParseError(`Expected ${kind}, got "${t.value}"`);
    if (value && t.value !== value) throw new ParseError(`Expected "${value}"`);
    this.pos++;
    return t;
  }
  parse() {
    const v = this.parseAdd();
    if (this.peek()) throw new ParseError("Trailing tokens after expression");
    return v;
  }
  parseAdd() {
    let left = this.parseMul();
    while (this.peek()?.kind === "op" && "+-".includes(this.peek().value)) {
      const op = this.eat("op").value;
      const right = this.parseMul();
      left = op === "+" ? add(left, right) : sub(left, right);
    }
    return left;
  }
  parseMul() {
    let left = this.parseAtom();
    for (;;) {
      const t = this.peek();
      if (t?.kind === "op" && t.value === "*") {
        this.eat("op");
        left = scale(left, this.parseScalar());
      } else if (t?.kind === "k") {
        this.eat("k");
        left = scale(left, this.k);
      } else if (t?.kind === "num") {
        left = scale(left, this.eat("num").value); // implicit: gender 2
      } else break;
    }
    return left;
  }
  parseScalar() {
    const t = this.peek();
    if (!t) throw new ParseError("Expected scalar after *");
    if (t.kind === "k") {
      this.eat("k");
      return this.k;
    }
    if (t.kind === "num") return this.eat("num").value;
    if (t.kind === "op" && t.value === "-") {
      this.eat("op");
      if (this.peek()?.kind === "num") return -this.eat("num").value;
      if (this.peek()?.kind === "k") {
        this.eat("k");
        return -this.k;
      }
    }
    throw new ParseError("Expected number or k");
  }
  parseAtom() {
    const t = this.peek();
    if (!t) throw new ParseError("Expected value");
    if (t.kind === "op" && t.value === "-") {
      this.eat("op");
      return scale(this.parseAtom(), -1);
    }
    if (t.kind === "op" && t.value === "(") {
      this.eat("op");
      const v = this.parseAdd();
      this.eat("op", ")");
      return v;
    }
    if (t.kind === "word") {
      const name = this.eat("word").value;
      if (this.env.has(name)) return this.env.get(name);
      const w = this.emb.word(name);
      this.words.set(name, w.pieces);
      return w.vec;
    }
    throw new ParseError(`Unexpected token "${t.value}"`);
  }
}

function evalExpr(expr, ctx) {
  return new Parser(lex(expr.trim()), ctx.k, ctx.env, ctx.emb, ctx.words).parse();
}

function project(vec, axisVecs) {
  const out = {};
  for (const [key, d] of Object.entries(axisVecs)) {
    const dn = norm(d);
    out[key] = dn < 1e-12 ? 0 : dot(vec, d) / dn;
  }
  return out;
}

function definitionLines(definitions) {
  const out = [];
  definitions.split("\n").forEach((raw, idx) => {
    const line = raw.split("#")[0].trim();
    if (!line || /^print\s*\(/.test(line)) return;
    out.push({ line, idx, m: line.match(/^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$/) });
  });
  return out;
}

function refExpr(ref) {
  const bar = ref.indexOf("|");
  return bar >= 0
    ? { label: ref.slice(0, bar).trim(), expr: ref.slice(bar + 1).trim() }
    : { label: ref, expr: ref };
}

// Every literal word the plot will need, so async backends can embed them up front.
// Expressions that fail to lex are skipped here; computePlot reports their errors.
export function collectWords({ definitions = "", track = "", axes = {}, references = [] }) {
  const defs = definitionLines(definitions);
  const names = new Set(defs.filter((d) => d.m).map((d) => d.m[1]));
  const exprs = [
    ...defs.filter((d) => d.m).map((d) => d.m[2]),
    ...Object.values(axes),
    track,
    ...references.map((r) => r.trim()).filter((r) => r && !r.startsWith("#")).map((r) => refExpr(r).expr),
  ];
  const words = new Set();
  for (const e of exprs) {
    try {
      for (const t of lex(e || "")) if (t.kind === "word" && !names.has(t.value)) words.add(t.value);
    } catch {
      /* reported later */
    }
  }
  return [...words];
}

// backend: anything with word(text) -> { vec: Float32Array, pieces: string[] }.
// definitions: "name = expr" lines (# comments, print(...) lines ignored).
export function computePlot(emb, { definitions = "", k, track, axes, labels = {}, references = [] }) {
  const ctx = { k, env: new Map(), emb, words: new Map() };
  definitionLines(definitions).forEach(({ idx, m }) => {
    if (!m) throw new ParseError(`Definitions line ${idx + 1}: expected name = expr`);
    try {
      ctx.env.set(m[1], evalExpr(m[2], ctx));
    } catch (e) {
      throw new ParseError(`Definitions line ${idx + 1}: ${e.message}`);
    }
  });

  const axisVecs = {};
  for (const [key, expr] of Object.entries(axes)) {
    if (!expr || !expr.trim()) continue;
    try {
      axisVecs[key] = evalExpr(expr, ctx);
    } catch (e) {
      throw new ParseError(`Axis ${key.toUpperCase()}: ${e.message}`);
    }
  }
  if (!Object.keys(axisVecs).length) throw new ParseError("Define at least one axis");

  let trackVec;
  try {
    trackVec = evalExpr(track, ctx);
  } catch (e) {
    throw new ParseError(`Tracked expression: ${e.message}`);
  }

  const refs = [];
  const skipped = [];
  for (let ref of references) {
    ref = ref.trim();
    if (!ref || ref.startsWith("#")) continue;
    const { label, expr } = refExpr(ref);
    try {
      refs.push({ label, expr, coords: project(evalExpr(expr, ctx), axisVecs) });
    } catch (e) {
      // A bad reference only drops that point, so the rest of the plot still shows.
      if (!(e instanceof ParseError)) throw e;
      skipped.push(`${label}: ${e.message}`);
    }
  }

  return {
    track: { expr: track, norm: norm(trackVec), coords: project(trackVec, axisVecs) },
    axes: Object.fromEntries(
      Object.keys(axisVecs).map((key) => [key, { expr: axes[key], label: labels[key] || key }])
    ),
    references: refs,
    skipped,
    words: Object.fromEntries(ctx.words),
  };
}
