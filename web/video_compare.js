// MAI Video Compare — browser-native synchronized viewer for the MAIVideoCompare
// node. The node writes small preview files; everything here is <video>
// elements: no tensors, no VAE, no VRAM. Ported from the flipbook's player.js
// (one live set of videos, buffered start-together, audio bound to one source).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const viewURL = (f) => api.apiURL(`/view?filename=${encodeURIComponent(f.filename)}` +
  `&type=${f.type}&subfolder=${encodeURIComponent(f.subfolder)}&t=${Date.now()}`);

function el(tag, style, parent, text) {
  const e = document.createElement(tag);
  Object.assign(e.style, style || {});
  if (text !== undefined) e.textContent = text;
  parent?.appendChild(e);
  return e;
}
function whole(v) {
  try { const b = v.buffered; return v.duration > 0 && b.length > 0 && b.end(b.length - 1) >= v.duration - 0.25; }
  catch (e) { return false; }
}

class CompareWidget {
  constructor(node) {
    this.node = node;
    this.items = [];          // manifest items
    this.vids = [];           // {item, video, pane, star}
    this.mode = "side";       // side | flip | wipe | grid
    this.pair = [0, 1];       // indexes into this.vids for pair modes
    this.audio = null;        // index whose audio is on (locked); hover overrides
    this.hover = null;
    this.showB = false;       // flip
    this.flicker = null;      // interval
    this.playing = false;
    this.build();
  }
  build() {
    const root = el("div", { fontFamily: "system-ui, sans-serif", fontSize: "12px", color: "#ddd",
      background: "#1b1b1b", padding: "6px", borderRadius: "6px", userSelect: "none" });
    this.bar = el("div", { display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center", marginBottom: "6px" }, root);
    this.stage = el("div", { position: "relative", width: "100%", background: "#000", minHeight: "120px" }, root);
    this.crow = el("div", { display: "flex", gap: "6px", alignItems: "center", marginTop: "6px" }, root);
    this.hint = el("div", { color: "#888", marginTop: "4px" }, root,
      "queue once to load previews. hover = hear, click = lock audio, ★ = winner (next queue passes it through), space play, ←/→ step, F flicker, 1-6 pick A then B");
    const modes = { side: "Side by side", flip: "Flip A/B", wipe: "Wipe", grid: "Grid" };
    this.modeBtns = {};
    for (const m in modes) this.modeBtns[m] = this.btn(this.bar, modes[m], () => this.setMode(m));
    this.playBtn = this.btn(this.crow, "Play", () => this.setPlay(!this.playing));
    this.scrub = el("input", { flex: "1" }, this.crow); this.scrub.type = "range"; this.scrub.min = 0; this.scrub.max = 1000; this.scrub.value = 0;
    this.scrub.oninput = () => this.seekAll(this.scrub.value / 1000 * this.dur());
    this.fnum = el("span", { minWidth: "80px", color: "#9c9" }, this.crow, "frame 0");
    this.wipe = el("input", { width: "120px", display: "none" }, this.crow); this.wipe.type = "range"; this.wipe.min = 0; this.wipe.max = 100; this.wipe.value = 50;
    this.wipe.oninput = () => this.stage.style.setProperty("--w", this.wipe.value + "%");
    root.tabIndex = 0;
    root.addEventListener("keydown", (e) => this.key(e));
    this.node.addDOMWidget("mai_compare_ui", "div", root, { serialize: false });
    const sz = this.node.size; this.node.setSize([Math.max(sz[0], 720), Math.max(sz[1], 520)]);
    this.root = root;
    this.setMode("side");
  }
  btn(parent, label, cb) {
    const b = el("button", { font: "12px system-ui", padding: "3px 8px", border: "1px solid #555",
      background: "#333", color: "#ddd", borderRadius: "4px", cursor: "pointer" }, parent, label);
    b.onclick = (e) => { e.stopPropagation(); cb(); };
    return b;
  }
  dur() { const v = this.vids[this.pair[0]]?.video; return v?.duration || 0; }
  fps() { return this.items[0]?.fps || 24; }

  // ---------- data from the node ----------
  onExecuted(msg) {
    const man = msg?.mai_compare?.[0];
    if (!man) return;
    this.teardown();
    this.items = man.items || [];
    this.winner = man.winner;
    this.pair = [0, Math.min(1, this.items.length - 1)];
    for (const it of this.items) {
      const v = document.createElement("video");
      v.src = viewURL(it); v.preload = "auto"; v.loop = true; v.muted = true; v.playsInline = true;
      Object.assign(v.style, { width: "100%", height: "100%", objectFit: "contain", display: "block", background: "#000" });
      this.vids.push({ item: it, video: v });
    }
    this.audio = this.vids.findIndex((x) => x.item.index === this.winner);
    if (this.audio < 0) this.audio = 0;
    this.layout();
    this.setPlay(true);
  }
  teardown() {
    this.setPlay(false);
    for (const x of this.vids) { try { x.video.pause(); x.video.removeAttribute("src"); x.video.load(); } catch (e) {} }
    this.vids = []; this.stage.innerHTML = "";
    if (this.flicker) { clearInterval(this.flicker); this.flicker = null; }
  }

  // ---------- layout ----------
  setMode(m) {
    this.mode = m;
    for (const k in this.modeBtns) this.modeBtns[k].style.background = k === m ? "#5a5" : "#333";
    this.wipe.style.display = m === "wipe" ? "" : "none";
    this.layout();
  }
  pane(x, idx, style) {
    const p = el("div", Object.assign({ position: "relative", overflow: "hidden", background: "#000" }, style), this.stage);
    p.appendChild(x.video);
    const tag = el("span", { position: "absolute", left: "6px", top: "4px", padding: "1px 6px", background: "rgba(0,0,0,.6)",
      borderRadius: "3px", fontSize: "11px", pointerEvents: "none" }, p,
      `${idx + 1} · ${x.item.label}` + (x.item.seed !== undefined && !String(x.item.label).includes("seed") ? ` · seed ${x.item.seed}` : ""));
    const star = el("button", { position: "absolute", right: "6px", top: "4px", font: "14px system-ui", background: "rgba(0,0,0,.6)",
      color: x.item.index === this.winner ? "#fc3" : "#888", border: "none", cursor: "pointer", borderRadius: "3px" }, p, "★");
    star.title = "winner: the next queue passes this source through";
    star.onclick = (e) => { e.stopPropagation(); this.setWinner(x.item.index); };
    x.pane = p; x.star = star; x.tag = tag;
    p.addEventListener("pointerenter", () => { this.hover = this.vids.indexOf(x); this.applyAudio(); });
    p.addEventListener("pointerleave", () => { this.hover = null; this.applyAudio(); });
    p.addEventListener("click", () => { this.audio = this.vids.indexOf(x); this.applyAudio(); });
    return p;
  }
  layout() {
    this.stage.innerHTML = "";
    for (const x of this.vids) x.pane = null;
    if (!this.vids.length) return;
    const n = this.vids.length;
    if (this.mode === "grid" || n === 1) {
      const cols = n <= 2 ? n : n <= 4 ? 2 : 3;
      Object.assign(this.stage.style, { display: "grid", gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: "3px" });
      this.vids.forEach((x, i) => this.pane(x, i, { aspectRatio: `${x.item.width}/${x.item.height}` }));
    } else {
      const [a, b] = this.pair.map((i) => this.vids[i]);
      Object.assign(this.stage.style, { display: this.mode === "side" ? "grid" : "block",
        gridTemplateColumns: this.mode === "side" ? "1fr 1fr" : "", gap: "3px",
        aspectRatio: this.mode === "side" ? `${2 * a.item.width}/${a.item.height}` : `${a.item.width}/${a.item.height}` });
      const pa = this.pane(a, this.pair[0], this.mode === "side" ? {} : { position: "absolute", inset: "0" });
      const pb = this.pane(b, this.pair[1], this.mode === "side" ? {} : { position: "absolute", inset: "0" });
      if (this.mode === "flip") { pb.style.display = this.showB ? "" : "none"; pa.style.display = this.showB ? "none" : ""; }
      if (this.mode === "wipe") {
        this.stage.style.setProperty("--w", this.wipe.value + "%");
        pb.style.clipPath = "inset(0 0 0 var(--w))";
        this.stage.onpointerdown = (ev) => {
          const move = (e) => { const r = this.stage.getBoundingClientRect();
            const pct = Math.max(0, Math.min(100, (e.clientX - r.left) / r.width * 100));
            this.stage.style.setProperty("--w", pct.toFixed(1) + "%"); this.wipe.value = Math.round(pct); };
          move(ev); this.stage.onpointermove = move; this.stage.onpointerup = () => { this.stage.onpointermove = null; };
        };
      } else this.stage.onpointerdown = null;
    }
    this.applyAudio();
  }
  setWinner(index) {
    this.winner = index;
    const w = this.node.widgets?.find((w) => w.name === "winner");
    if (w) { w.value = index; this.node.setDirtyCanvas?.(true, true); }
    for (const x of this.vids) if (x.star) x.star.style.color = x.item.index === index ? "#fc3" : "#888";
    this.hint.textContent = `winner: source ${index}. queue again to pass it through winner_video / finalize it`;
  }

  // ---------- playback ----------
  applyAudio() {
    const on = this.hover ?? this.audio;
    this.vids.forEach((x, i) => { x.video.muted = i !== on; });
  }
  setPlay(on) {
    this.playing = on;
    this.playBtn.textContent = on ? "Pause" : "Play";
    if (!on) { for (const x of this.vids) x.video.pause(); return; }
    if (!this.vids.length) return;
    if (this.started) { for (const x of this.vids) x.video.play().catch(() => {}); return; }
    this.started = true; this.playBtn.textContent = "Loading";
    const t0 = Date.now();
    const wait = setInterval(() => {
      if (!this.playing) { clearInterval(wait); return; }
      if (this.vids.every((x) => whole(x.video)) || Date.now() - t0 > 6000) {
        clearInterval(wait); this.playBtn.textContent = "Pause";
        for (const x of this.vids) { try { x.video.currentTime = 0; } catch (e) {} }
        for (const x of this.vids) x.video.play().catch(() => { x.video.muted = true; x.video.play().catch(() => {}); });
        this.tick();
      }
    }, 100);
  }
  tick() {
    if (!this.playing) return;
    const v = this.vids[this.pair[0]]?.video;
    if (v && v.duration) {
      this.scrub.value = Math.round(v.currentTime / v.duration * 1000);
      this.fnum.textContent = "frame " + Math.round(v.currentTime * this.fps());
      // once per loop, pull every other same-length source back onto the lead's clock
      if (v.currentTime < (this.lastT || 0) - 0.5)
        for (const x of this.vids) if (x.video !== v && Math.abs(x.video.duration - v.duration) < 0.08 && whole(x.video)
          && Math.abs(x.video.currentTime - v.currentTime) > 0.1) x.video.currentTime = v.currentTime;
      this.lastT = v.currentTime;
    }
    requestAnimationFrame(() => this.tick());
  }
  seekAll(t) {
    for (const x of this.vids) { try { x.video.currentTime = t; } catch (e) {} }
    this.fnum.textContent = "frame " + Math.round(t * this.fps());
  }
  step(df) {
    const v = this.vids[this.pair[0]]?.video; if (!v) return;
    this.setPlay(false);
    this.seekAll(Math.max(0, v.currentTime + df / this.fps()));
  }
  key(e) {
    if (e.key === " ") { e.preventDefault(); this.setPlay(!this.playing); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); this.step(e.shiftKey ? -12 : -1); }
    else if (e.key === "ArrowRight") { e.preventDefault(); this.step(e.shiftKey ? 12 : 1); }
    else if (e.key === "f" || e.key === "F") {
      if (this.flicker) { clearInterval(this.flicker); this.flicker = null; return; }
      if (this.mode !== "flip") this.setMode("flip");
      this.flicker = setInterval(() => { this.showB = !this.showB; this.layout(); }, 250);
    }
    else if (/^[1-6]$/.test(e.key)) {
      const i = parseInt(e.key) - 1; if (i >= this.vids.length) return;
      this.pair = [this.pair[1], i]; this.layout();
    }
    else if (e.key === "Enter") { const x = this.vids[this.pair[1]]; if (x) this.setWinner(x.item.index); }
  }
}

app.registerExtension({
  name: "MAINodes.VideoCompare",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MAIVideoCompare" && nodeData.name !== "MAISeedHunter") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      try { this.maiCompare = new CompareWidget(this); } catch (e) { console.error("MAIVideoCompare widget failed:", e); }
      return r;
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      this.maiCompare?.onExecuted(message);
    };
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () { this.maiCompare?.teardown(); onRemoved?.apply(this, arguments); };
  },
});
