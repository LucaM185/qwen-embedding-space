// One three.js 3D projection view (axes, reference points, tracked point, k trail).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const AXIS_COLORS = { x: ["#93c5fd", 0x93c5fd], y: ["#6ee7b7", 0x6ee7b7], z: ["#f9a8d4", 0xf9a8d4] };

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

const toVec = (c, s) => new THREE.Vector3((c.x || 0) * s, (c.y || 0) * s, (c.z || 0) * s);

function plotRadius(points, s) {
  let m = 0.35;
  for (const p of points) m = Math.max(m, Math.abs((p.x || 0) * s), Math.abs((p.y || 0) * s), Math.abs((p.z || 0) * s));
  return Math.max(m * 1.35, 1);
}

export class PlotView {
  constructor(canvas) {
    this.canvas = canvas;
    this.trail = [];
    this.lastPlot = null;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 2000);
    this.camera.position.set(4.5, 3.5, 5.5);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const light = new THREE.DirectionalLight(0xffffff, 0.85);
    light.position.set(5, 8, 4);
    this.scene.add(light);

    this.axisGroup = new THREE.Group();
    this.pointsGroup = new THREE.Group();
    this.trailLine = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xf9a8d4 })
    );
    this.trackMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.09, 24, 24),
      new THREE.MeshStandardMaterial({ color: 0x6ee7b7, emissive: 0x1a4035 })
    );
    this.trackMesh.visible = false;
    this.scene.add(this.axisGroup, this.pointsGroup, this.trailLine, this.trackMesh);
  }

  // Copy another view's camera so both orbit together.
  follow(other) {
    this.camera.position.copy(other.camera.position);
    this.controls.target.copy(other.controls.target);
    this.camera.quaternion.copy(other.camera.quaternion);
  }

  frame({ updateControls = true } = {}) {
    const { canvas, renderer, camera } = this;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w < 1 || h < 1) return;
    const px = renderer.getPixelRatio();
    if (canvas.width !== Math.floor(w * px) || canvas.height !== Math.floor(h * px)) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    if (updateControls) this.controls.update();
    renderer.render(this.scene, camera);
  }

  clear() {
    clearGroup(this.axisGroup);
    clearGroup(this.pointsGroup);
    this.trackMesh.visible = false;
    this.trailLine.visible = false;
    this.trail.length = 0;
    this.lastPlot = null;
  }

  drawAxes(labels, size) {
    clearGroup(this.axisGroup);
    for (const key of ["x", "y", "z"]) {
      const end = new THREE.Vector3(key === "x" ? size : 0, key === "y" ? size : 0, key === "z" ? size : 0);
      this.axisGroup.add(
        new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), end]),
          new THREE.LineBasicMaterial({ color: AXIS_COLORS[key][1] })
        )
      );
      const sp = labelSprite(labels[key] || key, AXIS_COLORS[key][0]);
      sp.position.copy(end).multiplyScalar(1.08);
      this.axisGroup.add(sp);
    }
    const grid = new THREE.GridHelper(size * 2, 10, 0x2a3140, 0x1a1f28);
    grid.position.y = -size * 0.02;
    this.axisGroup.add(grid);
  }

  update(plot, { vis, k, addToTrail = true }) {
    this.lastPlot = plot;
    const track = plot.track.coords;
    const refs = plot.references;
    const labels = {
      x: plot.axes.x?.label || "x",
      y: plot.axes.y?.label || "y",
      z: plot.axes.z?.label || "z",
    };
    this.drawAxes(labels, plotRadius([track, ...refs.map((r) => r.coords)], vis));

    clearGroup(this.pointsGroup);
    for (const r of refs) {
      const pos = toVec(r.coords, vis);
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.055, 16, 16),
        new THREE.MeshStandardMaterial({ color: 0x93c5fd, emissive: 0x142033 })
      );
      mesh.position.copy(pos);
      this.pointsGroup.add(mesh);
      const sp = labelSprite(r.label, "#93c5fd");
      sp.center.set(0, 0.5);
      sp.position.copy(pos).add(new THREE.Vector3(0.05, 0.02, 0));
      sp.scale.multiplyScalar(0.85);
      this.pointsGroup.add(sp);
    }

    this.trackMesh.position.copy(toVec(track, vis));
    this.trackMesh.visible = true;

    const trail = this.trail;
    if (addToTrail) {
      const last = trail[trail.length - 1];
      if (!last || last.k !== k || Math.hypot(last.x - track.x, last.y - track.y, last.z - track.z) > 1e-4) {
        trail.push({ k, x: track.x || 0, y: track.y || 0, z: track.z || 0 });
        if (trail.length > 400) trail.shift();
      }
    }
    this.trailLine.geometry.dispose();
    this.trailLine.geometry = new THREE.BufferGeometry();
    this.trailLine.geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(trail.flatMap((p) => [p.x * vis, p.y * vis, p.z * vis]), 3)
    );
    this.trailLine.visible = trail.length > 1;
    return labels;
  }
}
