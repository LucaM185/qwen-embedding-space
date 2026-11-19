import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Tokenizer, Embeddings, computePlot } from "./lib/engine.js";

const EXAMPLES = {
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
const DEFAULT = "Gender · familiarity · richness";
const STORAGE_KEY = "qwen-embedding-space:v1";

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
  coords: $("coords-readout"),
  tokens: $("tokens"),
  clearTrail: $("clear-trail"),
  scale: $("plot-scale"),
  scaleValue: $("plot-scale-value"),
  preset: $("preset-select"),
  share: $("share"),
  canvas: $("plot-canvas"),
  loading: $("loading"),
  loadingBar: $("loading-bar"),
  loadingText: $("loading-text"),
};

let emb = null;
let lastPlot = null;
const trail = [];
const visualScale = () => parseFloat(el.scale.value);
const fmt = (v) => Number(v || 0).toFixed(3);

// --- State: examples, localStorage, share link ----------------------------------
function collect() {
  return {
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
  trail.length = 0;
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

// --- Three.js scene --------------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas: el.canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 2000);
camera.position.set(4.5, 3.5, 5.5);
const controls = new OrbitControls(camera, el.canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const light = new THREE.DirectionalLight(0xffffff, 0.85);
light.position.set(5, 8, 4);
scene.add(light);

const axisGroup = new THREE.Group();
const pointsGroup = new THREE.Group();
const trailLine = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0xf9a8d4 })
);
const trackMesh = new THREE.Mesh(
  new THREE.SphereGeometry(0.09, 24, 24),
  new THREE.MeshStandardMaterial({ color: 0x6ee7b7, emissive: 0x1a4035 })
);
trackMesh.visible = false;
scene.add(axisGroup, pointsGroup, trailLine, trackMesh);

function resizePlot() {
  const w = el.canvas.clientWidth;
  const h = el.canvas.clientHeight;
  if (w < 1 || h < 1) return;
  const px = renderer.getPixelRatio();
  if (el.canvas.width !== Math.floor(w * px) || el.canvas.height !== Math.floor(h * px)) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

function labelSprite(text, color) {
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  const font = "22px ui-monospace, monospace";
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 16;
  c.width = w;
  c.height = 36;
  ctx.font = font;
  ctx.fillStyle = "rgba(10,12,16,0.75)";
  ctx.fillRect(0, 0, w, 36);
  ctx.fillStyle = color;
  ctx.fillText(text, 8, 24);
  const sp = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true })
  );
  sp.scale.set(w / 80, 36 / 80, 1);
  return sp;
}

function clearGroup(g) {
  while (g.children.length) {
    const ch = g.children[0];
    g.remove(ch);
    ch.geometry?.dispose?.();
    for (const m of [ch.material].flat().filter(Boolean)) {
      m.map?.dispose?.();
      m.dispose?.();
    }
  }
}

const AXIS_COLORS = { x: ["#93c5fd", 0x93c5fd], y: ["#6ee7b7", 0x6ee7b7], z: ["#f9a8d4", 0xf9a8d4] };

function drawAxes(labels, size) {
  clearGroup(axisGroup);
  for (const key of ["x", "y", "z"]) {
    const end = new THREE.Vector3(key === "x" ? size : 0, key === "y" ? size : 0, key === "z" ? size : 0);
    axisGroup.add(
      new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), end]),
        new THREE.LineBasicMaterial({ color: AXIS_COLORS[key][1] })
      )
    );
    const sp = labelSprite(labels[key] || key, AXIS_COLORS[key][0]);
    sp.position.copy(end).multiplyScalar(1.08);
    axisGroup.add(sp);
  }
  const grid = new THREE.GridHelper(size * 2, 10, 0x2a3140, 0x1a1f28);
  grid.position.y = -size * 0.02;
  axisGroup.add(grid);
}

const toVec = (c, s) => new THREE.Vector3((c.x || 0) * s, (c.y || 0) * s, (c.z || 0) * s);

function plotRadius(points, s) {
  let m = 0.35;
  for (const p of points) m = Math.max(m, Math.abs((p.x || 0) * s), Math.abs((p.y || 0) * s), Math.abs((p.z || 0) * s));
  return Math.max(m * 1.35, 1);
}

function updatePlot(plot, { addToTrail = true } = {}) {
  lastPlot = plot;
  const vis = visualScale();
  const track = plot.track.coords;
  const refs = plot.references;
  const labels = {
    x: plot.axes.x?.label || "x",
    y: plot.axes.y?.label || "y",
    z: plot.axes.z?.label || "z",
  };
  drawAxes(labels, plotRadius([track, ...refs.map((r) => r.coords)], vis));

  clearGroup(pointsGroup);
  for (const r of refs) {
    const pos = toVec(r.coords, vis);
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.055, 16, 16),
      new THREE.MeshStandardMaterial({ color: 0x93c5fd, emissive: 0x142033 })
    );
    mesh.position.copy(pos);
    pointsGroup.add(mesh);
    const sp = labelSprite(r.label, "#93c5fd");
    sp.center.set(0, 0.5);
    sp.position.copy(pos).add(new THREE.Vector3(0.05, 0.02, 0));
    sp.scale.multiplyScalar(0.85);
    pointsGroup.add(sp);
  }

  trackMesh.position.copy(toVec(track, vis));
  trackMesh.visible = true;

  if (addToTrail) {
    const k = parseFloat(el.k.value);
    const last = trail[trail.length - 1];
    if (!last || last.k !== k || Math.hypot(last.x - track.x, last.y - track.y, last.z - track.z) > 1e-4) {
      trail.push({ k, x: track.x || 0, y: track.y || 0, z: track.z || 0 });
      if (trail.length > 400) trail.shift();
    }
  }
  trailLine.geometry.dispose();
  trailLine.geometry = new THREE.BufferGeometry();
  trailLine.geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(trail.flatMap((p) => [p.x * vis, p.y * vis, p.z * vis]), 3)
  );
  trailLine.visible = trail.length > 1;

  el.coords.textContent = `track → ${labels.x}: ${fmt(track.x)} · ${labels.y}: ${fmt(track.y)} · ${labels.z}: ${fmt(track.z)}`;
  el.tokens.replaceChildren(
    ...Object.entries(plot.words)
      .filter(([, pieces]) => pieces.length > 1)
      .map(([w, pieces]) => {
        const s = document.createElement("span");
        s.textContent = `${w} → ${pieces.join("·")}`;
        return s;
      })
  );
  if (el.tokens.childElementCount) el.tokens.prepend("multi-token words (averaged): ");
}

// --- Compute ------------------------------------------------------------------
function run({ resetTrail = false } = {}) {
  if (!emb) return;
  if (resetTrail) trail.length = 0;
  const s = collect();
  const t0 = performance.now();
  try {
    const plot = computePlot(emb, {
      definitions: s.definitions,
      k: s.k,
      track: s.track,
      axes: s.axes,
      labels: s.labels,
      references: s.refs.split("\n"),
    });
    updatePlot(plot);
    el.status.textContent = `${(performance.now() - t0).toFixed(0)} ms`;
    el.status.classList.remove("error");
  } catch (err) {
    el.status.textContent = err.message;
    el.status.classList.add("error");
  }
  save();
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

el.k.addEventListener("input", () => {
  el.kValue.textContent = Number(el.k.value).toFixed(2);
  schedule(0);
});
for (const node of [el.defs, el.track, el.refs, ...Object.values(el.axes), ...Object.values(el.labels)]) {
  node.addEventListener("input", onEdit);
}
el.clearTrail.addEventListener("click", () => {
  trail.length = 0;
  run();
});
el.scale.addEventListener("input", () => {
  el.scaleValue.textContent = `${visualScale()}×`;
  if (lastPlot) updatePlot(lastPlot, { addToTrail: false });
  save();
});
el.preset.addEventListener("change", () => {
  const ex = EXAMPLES[el.preset.value];
  if (!ex) return;
  apply({ ...ex, scale: visualScale() });
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

function animate() {
  requestAnimationFrame(animate);
  resizePlot();
  controls.update();
  renderer.render(scene, camera);
}

// --- Load data ------------------------------------------------------------------
async function fetchWithProgress(url, total) {
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
    el.loadingBar.style.width = `${((got / total) * 100).toFixed(1)}%`;
    el.loadingText.textContent = `${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
  }
  return out.buffer.slice(0, got);
}

async function main() {
  apply(initialState());
  animate();
  try {
    const [meta, tok] = await Promise.all([
      fetch("data/meta.json").then((r) => r.json()),
      fetch("data/tokenizer.json").then((r) => r.json()),
    ]);
    const buf = await fetchWithProgress("data/embed.q4.bin", meta.rows * meta.row_bytes);
    if (buf.byteLength !== meta.rows * meta.row_bytes) throw new Error("Embedding file size mismatch");
    emb = new Embeddings(buf, meta, new Tokenizer(tok));
    el.loading.classList.add("hidden");
    run({ resetTrail: true });
  } catch (err) {
    el.loadingText.textContent = `Failed to load: ${err.message}`;
    el.status.textContent = err.message;
    el.status.classList.add("error");
  }
}

main();
