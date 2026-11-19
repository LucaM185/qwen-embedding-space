import { Tokenizer, WordTable, DenseWords, Embeddings, computePlot, collectWords } from "./lib/engine.js";
import { PlotView } from "./lib/view.js";

const EXAMPLES = {
  "Gender · royalty · age": {
    definitions: "age = old - young",
    track: "boy + age * k",
    k: 0,
    axes: { x: "man - woman", y: "(king + queen) - (man + woman)", z: "old - young" },
    labels: { x: "gender", y: "royalty", z: "age" },
    refs: "king\nqueen\nprince\nprincess\nman\nwoman\nboy\ngirl\ngrandfather\ngrandmother",
  },
  "Gender · familiarity · richness": {
    definitions: "gender = male - female",
    track: "mom + gender * k",
    k: 1,
    axes: { x: "friend - stranger", y: "male - female", z: "rich - poor" },
    labels: { x: "familiarity", y: "gender", z: "richness" },
    refs: "mom\ndad\nuncle\naunt\nwoman\nMan\nchild\nboy\ngirl\nrich",
  },
  "Royalty analogy": {
    definitions: "royal = king - man",
    track: "woman + royal * k",
    k: 1,
    axes: { x: "king - man", y: "woman - man", z: "old - young" },
    labels: { x: "royalty", y: "female", z: "age" },
    refs: "king\nqueen\nman\nwoman\nprince\nprincess\nboy\ngirl",
  },
  "Temperature · size": {
    definitions: "heat = hot - cold",
    track: "water + heat * k",
    k: 0,
    axes: { x: "hot - cold", y: "big - small", z: "fast - slow" },
    labels: { x: "temperature", y: "size", z: "speed" },
    refs: "fire\nice\nsun\nsnow\nelephant\nmouse\nrocket\nturtle\nwater",
  },
};
const DEFAULT = "Gender · royalty · age";
const STORAGE_KEY = "qwen-embedding-space:v1";
// Every embedding space the page can show.
const SPACES = {
  table: { title: "Qwen layer 0 · token table (any word)", loading: "Loading embeddings…" },
  l12: { title: "Qwen layer 12 (middle) · word list only", loading: "Loading layer 12…", layer: 12 },
  l24: { title: "Qwen layer 24 (final) · word list only", loading: "Loading layer 24…", layer: 24 },
  leaf: { title: "mdbr-leaf-ir (retrieval model, any text)", loading: "Loading model…" },
};
const MODES = {
  all: { label: "Compare all: Qwen layers 0 · 12 · 24 + RAG", spaces: ["table", "l12", "l24", "leaf"] },
  layers: { label: "Compare Qwen layers 0 · 12 · 24", spaces: ["table", "l12", "l24"] },
  "table-leaf": { label: "Compare Qwen layer 0 vs RAG", spaces: ["table", "leaf"] },
  table: { label: "Qwen layer 0 (token table)", spaces: ["table"] },
  l12: { label: "Qwen layer 12 (middle)", spaces: ["l12"] },
  l24: { label: "Qwen layer 24 (final)", spaces: ["l24"] },
  leaf: { label: "mdbr-leaf-ir (RAG)", spaces: ["leaf"] },
};
const LEGACY_MODES = { qwen: "table", compare: "table-leaf" };
const DEFAULT_MODE = "all";

const $ = (id) => document.getElementById(id);
const el = {
  defs: $("defs"),
  track: $("plot-track"),
  k: $("k"),
  kValue: $("k-value"),
  axes: { x: $("axis-x"), y: $("axis-y"), z: $("axis-z") },
  labels: { x: $("label-x"), y: $("label-y"), z: $("label-z") },
  refs: $("refs"),
  status: $("status"),
  clearTrail: $("clear-trail"),
  scale: $("plot-scale"),
  scaleValue: $("plot-scale-value"),
  preset: $("preset-select"),
  model: $("model-select"),
  share: $("share"),
  views: $("views"),
};

for (const [value, mode] of Object.entries(MODES)) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = mode.label;
  el.model.appendChild(opt);
}
el.model.value = DEFAULT_MODE;

const visualScale = () => parseFloat(el.scale.value);
const fmt = (v) => Number(v || 0).toFixed(3);

// --- Panels: one view per space ----------------------------------------------------
const panels = {};
const template = $("panel-template");
for (const [name, space] of Object.entries(SPACES)) {
  const node = template.content.firstElementChild.cloneNode(true);
  node.dataset.space = name;
  node.hidden = true;
  el.views.appendChild(node);
  const q = (sel) => node.querySelector(sel);
  q(".panel-title").textContent = space.title;
  q(".loading-label").textContent = space.loading;
  const panel = {
    node,
    view: new PlotView(q("canvas")),
    loading: q(".loading"),
    loadingBar: q(".loading-bar"),
    loadingText: q(".loading-text"),
    error: q(".panel-error"),
    coords: q(".coords-readout"),
    tokens: q(".tokens"),
    loadPromise: null,
  };
  q("canvas").addEventListener("pointerdown", () => (activePanel = panel));
  q("canvas").addEventListener("wheel", () => (activePanel = panel), { passive: true });
  panels[name] = panel;
}
let activePanel = panels.table;

function showProgress(panel, loaded, total) {
  panel.loadingBar.style.width = `${((loaded / total) * 100).toFixed(1)}%`;
  panel.loadingText.textContent = `${(loaded / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
}

// --- Backends ----------------------------------------------------------------------
async function fetchWithProgress(url, total, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  if (!res.body) return res.arrayBuffer();
  // Progress is measured against the decoded size from meta.json, since a CDN may
  // send the file compressed and Content-Length would then be the compressed size.
  const reader = res.body.getReader();
  const out = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (got + value.length > total) throw new Error("Embedding file larger than expected");
    out.set(value, got);
    got += value.length;
    onProgress(got, total);
  }
  return out.buffer.slice(0, got);
}

let layerShared = null; // promise of { meta, table } shared by all precomputed layers
function loadLayerShared() {
  const json = (url) =>
    fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return r.json();
    });
  layerShared ??= Promise.all([json("data/layers/meta.json"), json("data/layers/words.json")]).then(
    ([meta, words]) => ({ meta, table: new WordTable(words, meta.words) }),
    (err) => {
      layerShared = null; // allow a retry
      throw err;
    }
  );
  return layerShared;
}

async function loadLayer(panel, layer) {
  const { meta, table } = await loadLayerShared();
  const info = meta.layers[String(layer)];
  const total = meta.words * meta.dim * 4;
  const buf = await fetchWithProgress(`data/layers/${info.file}`, total, (a, b) => showProgress(panel, a, b));
  if (buf.byteLength !== total) throw new Error("Layer file size mismatch");
  return new DenseWords(buf, meta.dim, table);
}

async function loadBackend(name, panel) {
  if (name === "table") {
    const [meta, tok] = await Promise.all([
      fetch("data/meta.json").then((r) => r.json()),
      fetch("data/tokenizer.json").then((r) => r.json()),
    ]);
    const total = meta.rows * meta.row_bytes;
    const buf = await fetchWithProgress("data/embed.q4.bin", total, (a, b) => showProgress(panel, a, b));
    if (buf.byteLength !== total) throw new Error("Embedding file size mismatch");
    return new Embeddings(buf, meta, new Tokenizer(tok));
  }
  if (name === "leaf") {
    const { LeafEmbeddings } = await import("./lib/leaf.js");
    return LeafEmbeddings.load((a, b) => showProgress(panel, a, b));
  }
  return loadLayer(panel, SPACES[name].layer);
}

function ensureBackend(name) {
  const panel = panels[name];
  panel.loadPromise ??= loadBackend(name, panel).then(
    (backend) => {
      panel.loading.classList.add("hidden");
      return backend;
    },
    (err) => {
      panel.loadPromise = null;
      panel.loadingText.textContent = `Failed to load: ${err.message}`;
      throw err;
    }
  );
  return panel.loadPromise;
}

// --- State: examples, localStorage, share link ----------------------------------
function collect() {
  return {
    model: el.model.value,
    definitions: el.defs.value,
    track: el.track.value,
    k: parseFloat(el.k.value),
    axes: { x: el.axes.x.value, y: el.axes.y.value, z: el.axes.z.value },
    labels: { x: el.labels.x.value, y: el.labels.y.value, z: el.labels.z.value },
    refs: el.refs.value,
    scale: visualScale(),
  };
}

function apply(s) {
  if (!s) return;
  const mode = LEGACY_MODES[s.model] || s.model;
  if (mode && MODES[mode]) el.model.value = mode;
  if (s.definitions != null) el.defs.value = s.definitions;
  if (s.track != null) el.track.value = s.track;
  if (s.k != null) el.k.value = String(s.k);
  for (const key of ["x", "y", "z"]) {
    if (s.axes?.[key] != null) el.axes[key].value = s.axes[key];
    if (s.labels?.[key] != null) el.labels[key].value = s.labels[key];
  }
  if (s.refs != null) el.refs.value = s.refs;
  if (s.scale != null) el.scale.value = String(s.scale);
  el.kValue.textContent = Number(el.k.value).toFixed(2);
  el.scaleValue.textContent = `${visualScale()}×`;
}

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(collect()));
  } catch {
    /* storage unavailable */
  }
}

function fromHash() {
  if (!location.hash.startsWith("#s=")) return null;
  try {
    const b64 = location.hash.slice(3).replace(/-/g, "+").replace(/_/g, "/");
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return null;
  }
}

function shareLink() {
  const bytes = new TextEncoder().encode(JSON.stringify(collect()));
  const b64 = btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  return `${location.origin}${location.pathname}#s=${b64}`;
}

for (const name of Object.keys(EXAMPLES)) {
  const opt = document.createElement("option");
  opt.value = name;
  opt.textContent = name;
  el.preset.appendChild(opt);
}
const custom = document.createElement("option");
custom.value = "";
custom.textContent = "Custom";
el.preset.appendChild(custom);

function initialState() {
  const shared = fromHash();
  if (shared) {
    el.preset.value = "";
    return shared;
  }
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (stored) {
      el.preset.value = "";
      return stored;
    }
  } catch {
    /* ignore */
  }
  el.preset.value = DEFAULT;
  return EXAMPLES[DEFAULT];
}

// --- Compute ------------------------------------------------------------------
const activeModels = () => (MODES[el.model.value] || MODES[DEFAULT_MODE]).spaces;

// The spaces differ in scale by up to ~250x (layer 0 vs layer 24), so every view is
// zoomed to put its farthest reference point at the same on-screen distance.
function panelScale(plot) {
  const pts = plot.references.length ? plot.references.map((r) => r.coords) : [plot.track.coords];
  const m = Math.max(1e-6, ...pts.flatMap((c) => [c.x, c.y, c.z].map((v) => Math.abs(v || 0))));
  return (visualScale() * 0.5) / m;
}

function renderPanel(panel, plot, cfg, addToTrail) {
  const labels = panel.view.update(plot, { vis: panelScale(plot), k: cfg.k, addToTrail });
  const t = plot.track.coords;
  panel.coords.textContent = `track → ${labels.x}: ${fmt(t.x)} · ${labels.y}: ${fmt(t.y)} · ${labels.z}: ${fmt(t.z)}`;
  const multi = Object.entries(plot.words).filter(([, pieces]) => pieces.length > 1);
  panel.tokens.replaceChildren(
    ...(multi.length ? ["multi-token words (averaged): "] : []),
    ...multi.map(([w, pieces]) => {
      const s = document.createElement("span");
      s.textContent = `${w} → ${pieces.join("·")}`;
      return s;
    })
  );
}

let runSeq = 0;
async function run({ resetTrail = false } = {}) {
  const seq = ++runSeq;
  const s = collect();
  const cfg = {
    definitions: s.definitions,
    k: s.k,
    track: s.track,
    axes: s.axes,
    labels: s.labels,
    references: s.refs.split("\n"),
  };
  const words = collectWords(cfg);
  const t0 = performance.now();
  save();
  await Promise.all(
    activeModels().map(async (name) => {
      const panel = panels[name];
      if (resetTrail) panel.view.trail.length = 0;
      try {
        const backend = await ensureBackend(name);
        await backend.prepare(words);
        if (seq !== runSeq) return;
        const plot = computePlot(backend, cfg);
        renderPanel(panel, plot, cfg, true);
        const notListed = plot.skipped.filter((m) => /-word list$/.test(m)).map((m) => m.split(":")[0]);
        const other = plot.skipped.filter((m) => !/-word list$/.test(m));
        panel.error.textContent = [
          notListed.length ? `Not in the word list, hidden here: ${notListed.join(", ")}` : "",
          other.length ? `Skipped — ${other.join(" · ")}` : "",
        ].filter(Boolean).join(" · ");
      } catch (err) {
        if (seq !== runSeq) return;
        panel.error.textContent = err.message;
      }
    })
  );
  if (seq === runSeq) el.status.textContent = `${(performance.now() - t0).toFixed(0)} ms`;
}

let timer = null;
function schedule(delay, opts) {
  clearTimeout(timer);
  timer = setTimeout(() => run(opts), delay);
}

function onEdit() {
  el.preset.value = "";
  schedule(200, { resetTrail: true });
}

function applyMode() {
  const active = activeModels();
  for (const [name, panel] of Object.entries(panels)) panel.node.hidden = !active.includes(name);
  el.views.classList.toggle("compare", active.length > 1);
  if (!active.includes(Object.keys(panels).find((n) => panels[n] === activePanel))) {
    activePanel = panels[active[0]];
  }
}

el.k.addEventListener("input", () => {
  el.kValue.textContent = Number(el.k.value).toFixed(2);
  schedule(0);
});
for (const node of [el.defs, el.track, el.refs, ...Object.values(el.axes), ...Object.values(el.labels)]) {
  node.addEventListener("input", onEdit);
}
el.clearTrail.addEventListener("click", () => run({ resetTrail: true }));
el.scale.addEventListener("input", () => {
  el.scaleValue.textContent = `${visualScale()}×`;
  for (const name of activeModels()) {
    const p = panels[name];
    if (p.view.lastPlot) p.view.update(p.view.lastPlot, { vis: panelScale(p.view.lastPlot), k: 0, addToTrail: false });
  }
  save();
});
el.preset.addEventListener("change", () => {
  const ex = EXAMPLES[el.preset.value];
  if (!ex) return;
  apply({ ...ex, scale: visualScale() });
  run({ resetTrail: true });
});
el.model.addEventListener("change", () => {
  applyMode();
  run({ resetTrail: true });
});
el.share.addEventListener("click", async () => {
  const url = shareLink();
  history.replaceState(null, "", url);
  try {
    await navigator.clipboard.writeText(url);
    el.share.textContent = "Copied";
  } catch {
    el.share.textContent = "Link in address bar";
  }
  setTimeout(() => (el.share.textContent = "Copy link"), 1500);
});

// In compare modes all cameras move together: the view last touched leads.
function animate() {
  requestAnimationFrame(animate);
  const active = activeModels().map((n) => panels[n]);
  activePanel.view.frame();
  for (const p of active) {
    if (p === activePanel) continue;
    p.view.follow(activePanel.view);
    p.view.frame({ updateControls: false });
  }
}

apply(initialState());
applyMode();
animate();
run({ resetTrail: true });
