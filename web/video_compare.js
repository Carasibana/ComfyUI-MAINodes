// MAI Video Compare — browser-native synchronized viewer for the MAIVideoCompare
// node. The node writes small preview files; everything here is <video>
// elements: no tensors, no VAE, no VRAM. Ported from the flipbook's player.js
// (one live set of videos, buffered start-together, audio bound to one source).
//
// The house review page is the reference implementation for the timeline, the
// gear settings and the precise export; this file is that player fitted to a
// node widget. State the page keeps in localStorage keyed by document title is
// kept here keyed by node id: the DOM widget is serialize:false, and adding a
// real serialized widget would shift every older workflow's widget order.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const viewURL = (f) => api.apiURL(`/view?filename=${encodeURIComponent(f.filename)}` +
  `&type=${f.type}&subfolder=${encodeURIComponent(f.subfolder)}&t=${Date.now()}`);

// The two muxers are vendored next to this file (web/vendor, MIT, see its
// README) and are UMD globals, not ES modules: inject them as classic scripts
// from this extension's own path. Precise export degrades to MediaRecorder if
// either the injection or WebCodecs is unavailable.
let MUXERS = null;
function loadMuxers() {
  if (MUXERS) return MUXERS;
  const base = new URL("./vendor/", import.meta.url);
  MUXERS = Promise.all(["mp4-muxer.min.js", "webm-muxer.min.js"].map((f) => new Promise((res) => {
    const s = document.createElement("script");
    s.src = new URL(f, base).href;
    s.onload = () => res(true);
    s.onerror = () => res(false);
    document.head.appendChild(s);
  })));
  return MUXERS;
}

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
    this.spans = [];          // regenerated-window frame spans from the manifest
    this.curves = null;       // {name: [per-frame floats]} from the manifest
    this.loopA = null;        // loop brackets, in seconds; always present
    this.loopB = null;
    this.wf = null;           // {L, R} peak buckets of the A-side audio
    this.waveGen = 0;
    this.lastT = 0;
    this.flashUntil = 0;
    this.flashColor = "#3fa66a";
    this.set = this.loadSet();
    this.build();
  }

  // ---------- per-node settings (localStorage keyed by node id) ----------
  setKey() { return "mai.compare.set." + (this.node?.id ?? "n"); }
  loadSet() {
    const d = { fc: false, blip: true, markblip: true, spanA: null, spanB: null, marks: "" };
    try { return Object.assign(d, JSON.parse(localStorage.getItem(this.setKey()) || "{}")); }
    catch (e) { return d; }
  }
  saveSet() { try { localStorage.setItem(this.setKey(), JSON.stringify(this.set)); } catch (e) {} }
  bands() {
    if (this.set.spanA != null && this.set.spanB != null) return [[this.set.spanA, this.set.spanB]];
    return this.spans || [];
  }
  markList() {
    return String(this.set.marks || "").split(",").map((x) => parseFloat(x)).filter((x) => isFinite(x));
  }

  build() {
    // The root fills the box the node gives the widget; the stage takes what
    // is left after the two bars and never grows past it (videos fit inside).
    const root = el("div", { fontFamily: "system-ui, sans-serif", fontSize: "12px", color: "#ddd",
      background: "#1b1b1b", padding: "6px", borderRadius: "6px", userSelect: "none",
      boxSizing: "border-box", width: "100%", height: "100%", minHeight: "360px",
      display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" });
    this.bar = el("div", { display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center", marginBottom: "6px", flex: "0 0 auto" }, root);
    this.setpanel = el("div", { display: "none", background: "#222", border: "1px solid #444",
      borderRadius: "6px", padding: "6px", marginBottom: "6px", flex: "0 0 auto", lineHeight: "1.8" }, root);
    this.stage = el("div", { position: "relative", width: "100%", background: "#000", flex: "1 1 auto", minHeight: "0", overflow: "hidden" }, root);
    this.tlrow = el("div", { display: "flex", gap: "6px", alignItems: "center", marginTop: "6px", flex: "0 0 auto" }, root);
    this.crow = el("div", { display: "flex", gap: "6px", alignItems: "center", marginTop: "6px", flex: "0 0 auto" }, root);
    this.hint = el("div", { color: "#888", marginTop: "4px", flex: "0 0 auto", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }, root,
      "queue once to load previews. hover = hear, click = lock audio, ★ = winner (next queue passes it through), space play, ←/→ step, F flicker, 1-6 pick A then B, i/o loop in/out");
    const modes = { side: "Side by side", flip: "Flip A/B", wipe: "Wipe", grid: "Grid" };
    this.modeBtns = {};
    for (const m in modes) this.modeBtns[m] = this.btn(this.bar, modes[m], () => this.setMode(m));
    this.gearBtn = this.btn(this.bar, "⚙ settings", () => this.toggleSettings());
    this.exportBtn = this.btn(this.bar, "⤓ export", () => this.openExport());
    this.playBtn = this.btn(this.crow, "Play", () => this.setPlay(!this.playing));
    this.scrub = el("input", { flex: "1" }, this.crow); this.scrub.type = "range"; this.scrub.min = 0; this.scrub.max = 1000; this.scrub.value = 0;
    this.scrub.oninput = () => this.seekAll(this.scrub.value / 1000 * this.dur());
    this.fnum = el("span", { minWidth: "80px", color: "#9c9" }, this.crow, "frame 0");
    this.wipe = el("input", { width: "120px", display: "none" }, this.crow); this.wipe.type = "range"; this.wipe.min = 0; this.wipe.max = 100; this.wipe.value = 50;
    this.wipe.oninput = () => this.stage.style.setProperty("--w", this.wipe.value + "%");
    this.buildOverlay();
    this.buildSettings();
    this.buildTimeline();
    this.buildExportDialog(root);
    root.tabIndex = 0;
    root.addEventListener("keydown", (e) => this.key(e));
    const w = this.node.addDOMWidget("mai_compare_ui", "div", root, { serialize: false });
    if (w) w.computeSize = (width) => [width, 470];      // minimum; the node's height gives the rest
    const sz = this.node.size; this.node.setSize([Math.max(sz[0], 760), Math.max(sz[1], 640)]);
    this.root = root;
    this.setMode("side");
    loadMuxers();
    this.raf();
  }
  btn(parent, label, cb) {
    const b = el("button", { font: "12px system-ui", padding: "3px 8px", border: "1px solid #555",
      background: "#333", color: "#ddd", borderRadius: "4px", cursor: "pointer" }, parent, label);
    b.onclick = (e) => { e.stopPropagation(); cb(); };
    return b;
  }
  input(parent, type, width, cb) {
    const i = el("input", { width, background: "#111", color: "#ddd", border: "1px solid #555",
      borderRadius: "3px", font: "12px system-ui", padding: "1px 4px" }, parent);
    i.type = type;
    if (type === "checkbox") Object.assign(i.style, { width: "auto", padding: "0" });
    i.oninput = () => cb(i);
    i.onchange = () => cb(i);
    i.onkeydown = (e) => e.stopPropagation();   // typing must not reach the player keys
    return i;
  }
  label(parent, text) {
    return el("label", { marginRight: "10px", color: "#bbb" }, parent, text);
  }
  dur() { const v = this.vids[this.pair[0]]?.video; return v?.duration || 0; }
  fps() { return this.items[0]?.fps || 24; }
  lead() { return this.vids[this.pair[0]]?.video || null; }
  other() { return this.vids[this.pair[1]]?.video || null; }
  heardIndex() { const on = this.hover ?? this.audio; return on == null ? 0 : on; }

  // ---------- data from the node ----------
  onExecuted(msg) {
    const man = msg?.mai_compare?.[0];
    if (!man) return;
    this.teardown();
    this.items = man.items || [];
    this.winner = man.winner;
    this.spans = Array.isArray(man.spans) ? man.spans : [];
    this.curves = man.curves || null;
    this.pair = [0, Math.min(1, this.items.length - 1)];
    for (const it of this.items) {
      const v = document.createElement("video");
      v.src = viewURL(it); v.preload = "auto"; v.loop = true; v.muted = true; v.playsInline = true;
      Object.assign(v.style, { width: "100%", height: "100%", objectFit: "contain", display: "block", background: "#000" });
      this.vids.push({ item: it, video: v });
    }
    this.audio = this.vids.findIndex((x) => x.item.index === this.winner);
    if (this.audio < 0) this.audio = 0;
    this.loopA = null; this.loopB = null;
    this.layout();
    this.setPlay(true);
    this.loadWave();
  }
  teardown() {
    this.setPlay(false);
    for (const x of this.vids) { try { x.video.pause(); x.video.removeAttribute("src"); x.video.load(); } catch (e) {} }
    this.vids = []; this.stage.innerHTML = "";
    this.wf = null; this.waveGen++;
    if (this.flicker) { clearInterval(this.flicker); this.flicker = null; }
  }
  destroy() { this.dead = true; this.teardown(); }

  // ---------- layout ----------
  setMode(m) {
    this.mode = m;
    for (const k in this.modeBtns) this.modeBtns[k].style.background = k === m ? "#5a5" : "#333";
    this.wipe.style.display = m === "wipe" ? "" : "none";
    this.layout();
  }
  pane(x, idx, style) {
    const p = el("div", Object.assign({ position: "relative", overflow: "hidden", background: "#000", height: "100%" }, style), this.stage);
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
    if (!this.vids.length) { this.stage.appendChild(this.ovl); return; }
    const n = this.vids.length;
    this.stage.style.aspectRatio = "";
    if (this.mode === "grid" || n === 1) {
      const cols = n <= 2 ? n : n <= 4 ? 2 : 3, rows = Math.ceil(n / cols);
      Object.assign(this.stage.style, { display: "grid", gridTemplateColumns: `repeat(${cols}, 1fr)`,
        gridTemplateRows: `repeat(${rows}, 1fr)`, gap: "3px" });
      this.vids.forEach((x, i) => this.pane(x, i, { minHeight: "0", minWidth: "0" }));
    } else {
      const [a, b] = this.pair.map((i) => this.vids[i]);
      Object.assign(this.stage.style, { display: this.mode === "side" ? "grid" : "block",
        gridTemplateColumns: this.mode === "side" ? "1fr 1fr" : "", gridTemplateRows: "1fr", gap: "3px" });
      const pa = this.pane(a, this.pair[0], this.mode === "side" ? { minHeight: "0", minWidth: "0" } : { position: "absolute", inset: "0" });
      const pb = this.pane(b, this.pair[1], this.mode === "side" ? { minHeight: "0", minWidth: "0" } : { position: "absolute", inset: "0" });
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
    this.stage.appendChild(this.ovl);       // absolute, out of flow: survives the rebuild
    this.applyAudio();
  }
  setWinner(index) {
    this.winner = index;
    const w = this.node.widgets?.find((w) => w.name === "winner");
    if (w) { w.value = index; this.node.setDirtyCanvas?.(true, true); }
    for (const x of this.vids) if (x.star) x.star.style.color = x.item.index === index ? "#fc3" : "#888";
    this.hint.textContent = `winner: source ${index}. queue again to pass it through winner_video / finalize it`;
  }

  // ---------- overlay: frame counter + region ball ----------
  buildOverlay() {
    this.ovl = el("div", { position: "absolute", top: "6px", right: "8px", zIndex: "6", display: "flex",
      gap: "8px", alignItems: "center", font: "13px ui-monospace, monospace", color: "#fff",
      textShadow: "0 1px 4px #000", pointerEvents: "none" });
    this.blipEl = el("span", { width: "14px", height: "14px", borderRadius: "50%", opacity: "0",
      transition: "opacity .12s", boxShadow: "0 0 8px 2px rgba(0,0,0,.5)" }, this.ovl);
    this.fcEl = el("span", { display: "none" }, this.ovl, "frame 0");
    this.stage.appendChild(this.ovl);
  }
  flash(c) { this.flashColor = c; this.flashUntil = performance.now() + 350; }
  tickCross() {
    const v = this.lead(); if (!v) return;
    const t = v.currentTime || 0, fps = this.fps();
    if (t < this.lastT - 0.2) { this.lastT = t; return; }     // wrapped
    for (const sp of this.bands()) {
      const a = sp[0] / fps, b = sp[1] / fps;
      if (this.set.blip) {
        if (this.lastT < a && t >= a) this.flash("#3fa66a");   // region enter
        if (this.lastT < b && t >= b) this.flash("#d7732a");   // region exit
      }
    }
    if (this.set.markblip) for (const m of this.markList()) {
      const tm = m / fps; if (this.lastT < tm && t >= tm) { this.flash("#3f93d6"); break; }
    }
    this.lastT = t;
  }
  drawOvl() {
    const v = this.lead(), fps = this.fps();
    const t = v ? (v.currentTime || 0) : 0;
    this.fcEl.style.display = this.set.fc ? "" : "none";
    if (this.set.fc) this.fcEl.textContent = "frame " + Math.round(t * fps);
    const inR = this.set.blip && this.bands().some((sp) => t >= sp[0] / fps && t <= sp[1] / fps);
    if (performance.now() < this.flashUntil) { this.blipEl.style.background = this.flashColor; this.blipEl.style.opacity = "1"; }
    else if (inR) { this.blipEl.style.background = "#3fa66a"; this.blipEl.style.opacity = "0.55"; }
    else this.blipEl.style.opacity = "0";
  }

  // ---------- gear settings ----------
  buildSettings() {
    const p = this.setpanel;
    const l1 = this.label(p, "region frames ");
    this.spanAIn = this.input(l1, "number", "70px", (i) => {
      const sp = this.bands()[0] || [0, 0];
      this.set.spanA = i.value === "" ? null : +i.value;
      if (this.set.spanA != null && this.set.spanB == null) this.set.spanB = sp[1];
      this.saveSet();
    });
    el("span", {}, l1, " to ");
    this.spanBIn = this.input(l1, "number", "70px", (i) => {
      const sp = this.bands()[0] || [0, 0];
      this.set.spanB = i.value === "" ? null : +i.value;
      if (this.set.spanB != null && this.set.spanA == null) this.set.spanA = sp[0];
      this.saveSet();
    });
    this.btn(p, "reset to node", () => {
      this.set.spanA = this.set.spanB = null; this.saveSet();
      const sp = this.bands()[0] || [0, 0];
      this.spanAIn.value = sp[0]; this.spanBIn.value = sp[1];
    });
    const l2 = this.label(p, "");
    this.fcIn = this.input(l2, "checkbox", "", (i) => { this.set.fc = i.checked; this.saveSet(); });
    el("span", {}, l2, " frame counter");
    const l3 = this.label(p, "");
    this.blipIn = this.input(l3, "checkbox", "", (i) => { this.set.blip = i.checked; this.saveSet(); });
    el("span", {}, l3, " region enter/exit blips");
    el("div", { height: "2px" }, p);
    const l4 = this.label(p, "marks (frames, comma-sep) ");
    this.marksIn = this.input(l4, "text", "220px", (i) => { this.set.marks = i.value; this.saveSet(); });
    this.marksIn.placeholder = "e.g. 24, 60, 118";
    const l5 = this.label(p, "");
    this.markBlipIn = this.input(l5, "checkbox", "", (i) => { this.set.markblip = i.checked; this.saveSet(); });
    el("span", {}, l5, " mark blips");
  }
  toggleSettings() {
    const open = this.setpanel.style.display === "none";
    this.setpanel.style.display = open ? "" : "none";
    this.gearBtn.style.background = open ? "#5a5" : "#333";
    if (!open) return;
    const sp = this.bands()[0] || [0, 0];
    this.spanAIn.value = this.set.spanA != null ? this.set.spanA : sp[0];
    this.spanBIn.value = this.set.spanB != null ? this.set.spanB : sp[1];
    this.fcIn.checked = this.set.fc; this.blipIn.checked = this.set.blip;
    this.markBlipIn.checked = this.set.markblip; this.marksIn.value = this.set.marks;
  }

  // ---------- pro timeline ----------
  buildTimeline() {
    const box = el("div", { flex: "1 1 auto", minWidth: "200px", height: "84px", background: "#1f232a",
      border: "1px solid #444", borderRadius: "6px", overflow: "hidden" }, this.tlrow);
    this.tl = el("canvas", { width: "100%", height: "100%", display: "block", cursor: "crosshair" }, box);
    this.tnum = el("span", { minWidth: "165px", color: "#9c9", fontVariantNumeric: "tabular-nums" }, this.tlrow, "0.00 s · frame 0");
    this.btn(this.tlrow, "⌊ in", () => this.markIn());
    this.btn(this.tlrow, "out ⌋", () => this.markOut());
    this.btn(this.tlrow, "↺ full", () => { this.loopA = 0; this.loopB = this.dur() || null; });
    this.tlDrag = null;
    this.tl.addEventListener("pointerdown", (e) => {
      const x = e.offsetX, y = e.offsetY, H = this.tl.clientHeight;
      const tab = y < 14 || y > H - 14;                      // top/bottom grab tabs
      const near = (t, tol) => t != null && Math.abs(this.tlX(t) - x) < tol;
      this.tlDrag = near(this.loopA, tab ? 14 : 5) ? "in" : near(this.loopB, tab ? 14 : 5) ? "out" : "seek";
      this.tl.setPointerCapture(e.pointerId);
      if (this.tlDrag === "seek") this.tlMove(e);
      e.stopPropagation();
    });
    this.tl.addEventListener("pointermove", (e) => this.tlMove(e));
    this.tl.addEventListener("pointerup", () => { this.tlDrag = null; });
    this.tl.addEventListener("dblclick", () => { this.loopA = 0; this.loopB = this.dur() || null; });
    window.addEventListener("resize", () => this.tlResize());
    setTimeout(() => this.tlResize(), 50);
  }
  markIn() { const v = this.lead(); if (!v) return; this.loopA = v.currentTime; if (this.loopB != null && this.loopB <= this.loopA) this.loopB = null; }
  markOut() { const v = this.lead(); if (!v) return; this.loopB = v.currentTime; if (this.loopA != null && this.loopA >= this.loopB) this.loopA = null; }
  tlX(t) { const d = this.dur(); return d ? t / d * this.tl.clientWidth : 0; }
  tlT(x) { const d = this.dur(); return d ? Math.max(0, Math.min(d, x / this.tl.clientWidth * d)) : 0; }
  tlResize() {
    const r = window.devicePixelRatio || 1;
    this.tl.width = Math.max(1, Math.round(this.tl.clientWidth * r));
    this.tl.height = Math.max(1, Math.round(this.tl.clientHeight * r));
    this.tl.getContext("2d").setTransform(r, 0, 0, r, 0, 0);
  }
  tlMove(e) {
    if (!this.tlDrag) return;
    const t = this.tlT(e.offsetX), v = this.lead();
    if (this.tlDrag === "seek") this.seekAll(t);
    else if (this.tlDrag === "in") {
      this.loopA = this.loopB != null ? Math.min(t, this.loopB - 0.05) : t;
      if (v && v.currentTime < this.loopA) this.seekAll(this.loopA);       // stranded playhead
    } else {
      this.loopB = this.loopA != null ? Math.max(t, this.loopA + 0.05) : t;
      if (v && v.currentTime > this.loopB) this.seekAll(Math.max(this.loopA || 0, this.loopB - 0.05));
    }
  }
  loopTick() {
    const v = this.lead(); if (!v) return;
    if (this.loopA == null || this.loopB == null) return;
    const t = v.currentTime;
    if (t > this.loopB + 0.02 || t < this.loopA - 0.35) this.seekAll(this.loopA);
  }
  async loadWave() {
    this.wf = null;
    const gen = ++this.waveGen, x = this.vids[this.pair[0]];
    if (!x) return;
    try {
      const ab = await (await fetch(x.video.currentSrc || x.video.src)).arrayBuffer();
      const AC = window.AudioContext || window.webkitAudioContext;
      const ac = new AC();
      const buf = await ac.decodeAudioData(ab);
      ac.close();
      if (gen !== this.waveGen) return;
      const N = 1000, c0 = buf.getChannelData(0), c1 = buf.numberOfChannels > 1 ? buf.getChannelData(1) : c0;
      const bucket = (c) => {
        const out = [], step = Math.max(1, Math.floor(c.length / N));
        for (let i = 0; i < N; i++) {
          let m = 0; const st = i * step, e = Math.min(c.length, st + step);
          for (let j = st; j < e; j += 2) { const v = Math.abs(c[j]); if (v > m) m = v; }
          out.push(m);
        }
        return out;
      };
      this.wf = { L: bucket(c0), R: bucket(c1) };
    } catch (e) { this.wf = null; }       // a silent preview simply has no wave
  }
  drawTL() {
    const cx = this.tl.getContext("2d"), W = this.tl.clientWidth, H = this.tl.clientHeight;
    const d = this.dur(), fps = this.fps(), mid = H / 2;
    cx.clearRect(0, 0, W, H);
    cx.fillStyle = "#14171c"; cx.fillRect(0, 0, W, H);
    if (d) {
      if (this.loopA == null) this.loopA = 0;
      if (this.loopB == null || this.loopB > d + 0.001) this.loopB = d;
      if (this.loopA > d) this.loopA = 0;
      cx.fillStyle = "rgba(63,147,214,.16)";
      for (const sp of this.bands()) cx.fillRect(this.tlX(sp[0] / fps), 0, Math.max(2, this.tlX((sp[1] - sp[0]) / fps)), H);
      for (const m of this.markList()) {
        const x = this.tlX(m / fps);
        cx.fillStyle = "#3f93d6"; cx.beginPath(); cx.moveTo(x - 4, 0); cx.lineTo(x + 4, 0); cx.lineTo(x, 6); cx.closePath(); cx.fill();
        cx.fillStyle = "rgba(63,147,214,.16)";
      }
    }
    if (this.wf) {
      const n = this.wf.L.length;
      cx.fillStyle = "rgba(154,163,173,.85)";
      for (let i = 0; i < n; i++) {
        const x = i / n * W, w = Math.max(1, W / n);
        const l = this.wf.L[i] * (mid - 3), r = this.wf.R[i] * (mid - 3);
        cx.fillRect(x, mid - 2 - l, w, l);    // L above the centre line
        cx.fillRect(x, mid + 2, w, r);        // R below it
      }
      cx.fillStyle = "rgba(255,255,255,.25)"; cx.fillRect(0, mid - 1, W, 2);
    } else { cx.fillStyle = "rgba(255,255,255,.15)"; cx.fillRect(0, mid - 1, W, 2); }
    if (this.curves) {
      const names = Object.keys(this.curves), cols = ["#3f93d6", "#c2456b", "#3fa66a", "#f2c94c"];
      names.forEach((nm, k) => {
        const c = this.curves[nm];
        if (!Array.isArray(c) || c.length < 2) return;
        const hi = Math.max(...c, 1e-6);
        cx.strokeStyle = cols[k % cols.length]; cx.lineWidth = 1.2; cx.beginPath();
        c.forEach((v, i) => { const x = i / (c.length - 1) * W, y = H - 4 - (H - 8) * v / hi; i ? cx.lineTo(x, y) : cx.moveTo(x, y); });
        cx.stroke();
      });
    }
    if (this.loopA != null && this.loopB != null) {
      const xa = this.tlX(this.loopA), xb = this.tlX(this.loopB);
      cx.fillStyle = "rgba(242,201,76,.16)"; cx.fillRect(xa, 0, xb - xa, H);
      cx.strokeStyle = "#f2c94c"; cx.lineWidth = 2; cx.fillStyle = "#f2c94c";
      const brk = (x, dir) => {
        cx.beginPath(); cx.moveTo(x + 7 * dir, 1); cx.lineTo(x, 1); cx.lineTo(x, H - 1); cx.lineTo(x + 7 * dir, H - 1); cx.stroke();
        cx.fillRect(dir > 0 ? x : x - 12, 0, 12, 10); cx.fillRect(dir > 0 ? x : x - 12, H - 10, 12, 10);   // grab tabs
      };
      brk(xa, 1); brk(xb, -1);
    }
    const v = this.lead();
    if (d && v) {
      const x = this.tlX(v.currentTime);
      cx.fillStyle = "#fff"; cx.fillRect(x - 0.5, 0, 1.5, H);
      cx.beginPath(); cx.moveTo(x - 5, 0); cx.lineTo(x + 5, 0); cx.lineTo(x, 7); cx.closePath();
      cx.fillStyle = "#f2c94c"; cx.fill();
    }
    const t = v ? (v.currentTime || 0) : 0;
    this.tnum.textContent = t.toFixed(2) + " s · frame " + Math.round(t * fps) +
      ((this.loopA != null && this.loopB != null) ? " · loop " + Math.round(this.loopA * fps) + "-" + Math.round(this.loopB * fps) : "");
  }
  raf() {
    if (this.dead) return;
    try {
      const want = Math.max(1, Math.round(this.tl.clientWidth * (window.devicePixelRatio || 1)));
      if (this.tl.clientWidth && this.tl.width !== want) this.tlResize();
      this.tickCross(); this.drawTL(); this.drawOvl(); this.loopTick();
    } catch (e) {}
    requestAnimationFrame(() => this.raf());
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
      if (v.currentTime < (this.lastSync || 0) - 0.5)
        for (const x of this.vids) if (x.video !== v && Math.abs(x.video.duration - v.duration) < 0.08 && whole(x.video)
          && Math.abs(x.video.currentTime - v.currentTime) > 0.1) x.video.currentTime = v.currentTime;
      this.lastSync = v.currentTime;
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
    else if (e.key === "i") { this.markIn(); }
    else if (e.key === "o") { this.markOut(); }
    else if (/^[1-6]$/.test(e.key)) {
      const i = parseInt(e.key) - 1; if (i >= this.vids.length) return;
      this.pair = [this.pair[1], i]; this.layout(); this.loadWave();
    }
    else if (e.key === "Enter") { const x = this.vids[this.pair[1]]; if (x) this.setWinner(x.item.index); }
  }

  // ---------- export ----------
  buildExportDialog(root) {
    this.xdlg = el("div", { display: "none", position: "absolute", inset: "0", background: "rgba(0,0,0,.65)",
      zIndex: "20", padding: "18px", overflow: "auto" }, root);
    const box = el("div", { maxWidth: "440px", margin: "0 auto", background: "#1f232a", border: "1px solid #444",
      borderRadius: "8px", padding: "12px", lineHeight: "1.9" }, this.xdlg);
    el("div", { fontWeight: "600", marginBottom: "4px" }, box, "Export the current view");
    this.xinfo = el("div", { color: "#9aa3ad", marginBottom: "6px" }, box, "");
    const l1 = this.label(box, "");
    this.xaudio = this.input(l1, "checkbox", "", () => {});
    this.xaudio.checked = true;
    el("span", {}, l1, " include audio (the side you hear)");
    el("div", { height: "2px" }, box);
    const l2 = this.label(box, "");
    this.xprecise = this.input(l2, "checkbox", "", () => {});
    this.xprecise.checked = true;
    el("span", {}, l2, " precise export (WebCodecs: frame-exact, faster than real time)");
    el("div", { height: "2px" }, box);
    const l3 = this.label(box, "codec ");
    this.xcodec = el("select", { background: "#111", color: "#ddd", border: "1px solid #555", borderRadius: "3px" }, l3);
    for (const [v, t] of [["vp9", "best quality (VP9, .webm)"], ["av1", "smallest (AV1, .webm, slow)"],
      ["mp4", "compatibility (H.264 .mp4, the browser encoder is call-grade)"]]) {
      const o = document.createElement("option"); o.value = v; o.textContent = t; this.xcodec.append(o);
    }
    const l4 = this.label(box, "bitrate ");
    this.xrate = el("select", { background: "#111", color: "#ddd", border: "1px solid #555", borderRadius: "3px" }, l4);
    for (const r of [24, 16, 40, 60, 100]) { const o = document.createElement("option"); o.value = r; o.textContent = r + " Mbps"; this.xrate.append(o); }
    const row = el("div", { display: "flex", gap: "8px", alignItems: "center", marginTop: "8px" }, box);
    this.btn(row, "record & save", () => this.runExport());
    this.btn(row, "cancel", () => { this.xdlg.style.display = "none"; });
    this.xstat = el("span", { color: "#9aa3ad" }, row, "");
  }
  pickMime() {
    return ["video/mp4;codecs=avc1.640028", "video/mp4;codecs=avc1.42E01E", "video/mp4;codecs=avc1", "video/mp4",
      "video/webm;codecs=vp9", "video/webm"].find((m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m));
  }
  async openExport() {
    const d = this.dur();
    if (!d) { this.hint.textContent = "queue once and let a pair load before exporting"; return; }
    await loadMuxers();
    const a = this.loopA != null ? this.loopA : 0, b = this.loopB != null ? this.loopB : d;
    const A = this.vids[this.pair[0]]?.item, B = this.vids[this.pair[1]]?.item;
    this.xinfo.textContent = `${A?.label || "A"} vs ${B?.label || "B"} · ${this.mode === "wipe" ? "wipe (current slider)" : "side by side"} · ` +
      `${a.toFixed(2)}s to ${b.toFixed(2)}s (${(b - a).toFixed(2)}s) + 2s end card`;
    const cap = ("VideoEncoder" in window) && (typeof Mp4Muxer !== "undefined" || typeof WebMMuxer !== "undefined");
    this.xprecise.checked = cap; this.xprecise.disabled = !cap;
    this.xstat.textContent = cap ? "" : "WebCodecs or the muxers are unavailable: real-time capture only";
    this.xdlg.style.display = "";
  }
  exportName(t0, t1, ext) {
    const A = this.vids[this.pair[0]]?.item, B = this.vids[this.pair[1]]?.item, fps = this.fps();
    const nm = (x) => String(x?.label || "src").replace(/[^A-Za-z0-9_.-]+/g, "_").slice(0, 40);
    return `${nm(A)}_vs_${nm(B)}_f${Math.round(t0 * fps)}-${Math.round(t1 * fps)}.${ext}`;
  }
  endCardLines() {
    const A = this.vids[this.pair[0]]?.item, B = this.vids[this.pair[1]]?.item;
    const f = (x) => x ? [x.label, x.seed !== undefined ? `seed ${x.seed}` : "",
      `${x.frames}f @ ${Number(x.fps).toFixed(2)} fps`, `${x.width}x${x.height}`].filter(Boolean).join("  ·  ") : "";
    return [`A  ${f(A)}`, `B  ${f(B)}`];
  }
  drawKit() {
    // one canvas plus painters, shared by both export paths
    const va = this.lead(), vb = this.other() || this.lead();
    const W = va.videoWidth || 864, H = va.videoHeight || 480;
    const wipe = this.mode === "wipe";
    const cw = wipe ? W : 2 * W + 4, ch = H;
    const cv = document.createElement("canvas"); cv.width = cw; cv.height = ch;
    const cx = cv.getContext("2d");
    const A = this.vids[this.pair[0]]?.item, B = this.vids[this.pair[1]]?.item;
    const top = `A ${A?.label || ""}   vs   B ${B?.label || ""}`;
    const fps = this.fps(), self = this;
    const drawAB = () => {
      cx.fillStyle = "#000"; cx.fillRect(0, 0, cw, ch);
      if (wipe) {
        cx.drawImage(va, 0, 0, W, H);
        const w = (self.wipe.value / 100) * W;
        cx.save(); cx.beginPath(); cx.rect(w, 0, W - w, H); cx.clip(); cx.drawImage(vb, 0, 0, W, H); cx.restore();
        cx.fillStyle = "#f2c94c"; cx.fillRect(w - 1, 0, 2, H);
      } else {
        cx.drawImage(va, 0, 0, W, H); cx.drawImage(vb, W + 4, 0, W, H);
        cx.font = "20px system-ui, sans-serif"; cx.textBaseline = "top";
        cx.fillStyle = "rgba(0,0,0,.55)"; cx.fillRect(0, ch - 34, 90, 34); cx.fillRect(W + 4, ch - 34, 90, 34);
        cx.fillStyle = "#fff"; cx.fillText("A", 12, ch - 28); cx.fillText("B", W + 16, ch - 28);
      }
      cx.fillStyle = "rgba(0,0,0,.55)"; cx.fillRect(0, 0, cw, 40);
      cx.fillStyle = "#fff"; cx.font = "20px system-ui, sans-serif"; cx.textBaseline = "top"; cx.fillText(top, 12, 10);
      // bake the frame counter and the region ball in when they are on
      const tt = va.currentTime || 0; let bx = cw - 16;
      if (self.set.fc) {
        cx.font = "20px ui-monospace, monospace";
        const txt = "frame " + Math.round(tt * fps), w = cx.measureText(txt).width;
        cx.fillStyle = "rgba(0,0,0,.55)"; cx.fillRect(cw - w - 28, 46, w + 20, 30);
        cx.fillStyle = "#fff"; cx.fillText(txt, cw - w - 18, 52); bx = cw - w - 44;
      }
      const inR = self.set.blip && self.bands().some((sp) => tt >= sp[0] / fps && tt <= sp[1] / fps);
      let col = null, op = 0;
      if (performance.now() < self.flashUntil) { col = self.flashColor; op = 1; }
      else if (inR) { col = "#3fa66a"; op = 0.55; }
      if (col) { cx.globalAlpha = op; cx.fillStyle = col; cx.beginPath(); cx.arc(bx - 8, 61, 8, 0, 7); cx.fill(); cx.globalAlpha = 1; }
    };
    const drawEnd = () => {
      cx.fillStyle = "#000"; cx.fillRect(0, 0, cw, ch);
      cx.fillStyle = "#fff"; cx.textBaseline = "top";
      cx.font = "26px system-ui, sans-serif"; cx.fillText(top, 40, 40);
      cx.font = "20px system-ui, sans-serif";
      let y = 96;
      for (const line of self.endCardLines()) { cx.fillText(line, 40, y); y += 30; if (y > ch - 40) break; }
      const bd = self.bands()[0];
      if (bd && y <= ch - 40) cx.fillText(`region  f${bd[0]}-f${bd[1]}`, 40, y);
    };
    return { va, vb, cv, cx, cw, ch, drawAB, drawEnd };
  }
  async runExport() {
    const d = this.dur(); if (!d) return;
    const t0 = this.loopA != null ? this.loopA : 0, t1 = this.loopB != null ? this.loopB : d;
    const kit = this.drawKit();
    const { va, vb } = kit;
    const wasA = va.loop, wasB = vb.loop, wasPlaying = this.playing;
    this.xdlg.style.display = "none";
    this.exportBtn.disabled = true;
    va.loop = false; vb.loop = false;
    this.setPlay(false);
    try {
      if (this.xprecise.checked && ("VideoEncoder" in window) && (typeof Mp4Muxer !== "undefined" || typeof WebMMuxer !== "undefined"))
        await this.exportPrecise(kit, t0, t1);
      else
        await this.exportRealtime(kit, t0, t1);
    } catch (err) {
      console.error("MAIVideoCompare export failed", err);
      this.hint.textContent = "export failed: " + err;
    }
    va.loop = wasA; vb.loop = wasB;
    this.exportBtn.disabled = false; this.exportBtn.textContent = "⤓ export";
    if (wasPlaying) this.setPlay(true);
  }
  saveBlob(blob, name) {
    const u = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = u; a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(u), 10000);
  }
  async exportRealtime(kit, t0, t1) {
    const { va, vb, cv, drawAB, drawEnd } = kit;
    const mime = this.pickMime();
    if (!mime) throw new Error("no MediaRecorder codec in this browser");
    const stream = cv.captureStream(this.fps());
    if (this.xaudio.checked) {
      const hv = this.vids[this.heardIndex()]?.video || va;
      try { const at = hv.captureStream().getAudioTracks()[0]; if (at) stream.addTrack(at); } catch (e) {}
    }
    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: (parseInt(this.xrate.value) || 24) * 1e6 });
    const chunks = []; rec.ondataavailable = (e) => chunks.push(e.data);
    const done = new Promise((r) => { rec.onstop = r; });
    this.exportBtn.textContent = "recording…";
    this.seekAll(t0);
    await new Promise((r) => setTimeout(r, 300));
    rec.start(250); va.play(); vb.play();
    let phase = "main", endAt = 0, raf;
    await new Promise((res) => {
      const step = () => {
        if (phase === "main") {
          drawAB();
          if (va.currentTime >= t1 - 0.02 || va.ended) { phase = "end"; endAt = performance.now() + 2000; va.pause(); vb.pause(); }
        } else { drawEnd(); if (performance.now() > endAt) { cancelAnimationFrame(raf); return res(); } }
        raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    });
    rec.stop(); await done;
    const ext = mime.startsWith("video/mp4") ? "mp4" : "webm";
    this.saveBlob(new Blob(chunks, { type: mime }), this.exportName(t0, t1, ext));
  }
  async exportPrecise(kit, t0, t1) {
    const { va, vb, cv, cw, ch, drawAB, drawEnd } = kit;
    const FPS = this.fps();
    const RATE = (parseInt(this.xrate.value) || 24) * 1e6;
    const want = this.xcodec.value;
    const ok = async (c) => (await VideoEncoder.isConfigSupported({ codec: c, width: cw, height: ch, bitrate: RATE, framerate: FPS })).supported;
    let vcodec = null, mp4ok = false;
    if (want === "mp4" && typeof Mp4Muxer !== "undefined" && await ok("avc1.640028")) { vcodec = "avc1.640028"; mp4ok = true; }
    else if (want === "av1" && typeof WebMMuxer !== "undefined" && await ok("av01.0.08M.08")) { vcodec = "av01.0.08M.08"; }
    if (!vcodec && typeof WebMMuxer !== "undefined" && await ok("vp09.00.10.08")) vcodec = "vp09.00.10.08";
    if (!vcodec && typeof Mp4Muxer !== "undefined" && await ok("avc1.640028")) { vcodec = "avc1.640028"; mp4ok = true; }
    if (!vcodec) throw new Error("no supported encoder+muxer");
    let abuf = null;
    if (this.xaudio.checked && ("AudioEncoder" in window)) {
      try {
        const hv = this.vids[this.heardIndex()]?.video || va;
        const ab = await (await fetch(hv.currentSrc || hv.src)).arrayBuffer();
        const AC = window.AudioContext || window.webkitAudioContext;
        const ac = new AC();
        abuf = await ac.decodeAudioData(ab);
        ac.close();
        if (!(await AudioEncoder.isConfigSupported({ codec: "opus", sampleRate: abuf.sampleRate,
          numberOfChannels: Math.min(2, abuf.numberOfChannels), bitrate: 192000 })).supported) abuf = null;
      } catch (e) { abuf = null; }
    }
    const sr = abuf ? abuf.sampleRate : 48000, nch = abuf ? Math.min(2, abuf.numberOfChannels) : 2;
    let muxer, target;
    if (mp4ok) {
      target = new Mp4Muxer.ArrayBufferTarget();
      muxer = new Mp4Muxer.Muxer({ target, fastStart: "in-memory", video: { codec: "avc", width: cw, height: ch },
        audio: abuf ? { codec: "opus", sampleRate: sr, numberOfChannels: nch } : undefined });
    } else {
      target = new WebMMuxer.ArrayBufferTarget();
      muxer = new WebMMuxer.Muxer({ target, video: { codec: vcodec.startsWith("av01") ? "V_AV1" : "V_VP9", width: cw, height: ch },
        audio: abuf ? { codec: "A_OPUS", sampleRate: sr, numberOfChannels: nch } : undefined });
    }
    const venc = new VideoEncoder({ output: (c, m) => muxer.addVideoChunk(c, m), error: (e) => { throw e; } });
    const vcfg = { codec: vcodec, width: cw, height: ch, bitrate: RATE, framerate: FPS, latencyMode: "quality" };
    if (mp4ok) vcfg.avc = { format: "avc" };
    venc.configure(vcfg);
    const seek = (v, t) => new Promise((r) => {
      let done = false;
      const f = () => { if (done) return; done = true; v.removeEventListener("seeked", f); r(); };
      v.addEventListener("seeked", f);
      v.currentTime = Math.min(t, (v.duration || t + 1) - 0.0001);
      setTimeout(f, 800);
    });
    va.pause(); vb.pause();
    const n = Math.max(1, Math.round((t1 - t0) * FPS)), us = (f) => Math.round(f * 1e6 / FPS);
    for (let i = 0; i < n; i++) {
      const t = t0 + i / FPS;
      await seek(va, t); if (vb !== va) await seek(vb, t);
      drawAB();
      const vf = new VideoFrame(cv, { timestamp: us(i), duration: us(1) });
      venc.encode(vf, { keyFrame: i % (FPS * 2) === 0 }); vf.close();
      if (i % 6 === 0) { this.exportBtn.textContent = "export " + Math.round(100 * i / n) + "%"; await new Promise((r) => setTimeout(r, 0)); }
      while (venc.encodeQueueSize > 4) await new Promise((r) => setTimeout(r, 2));
    }
    drawEnd();
    for (let i = 0; i < 2 * FPS; i++) {
      const vf = new VideoFrame(cv, { timestamp: us(n + i), duration: us(1) });
      venc.encode(vf, { keyFrame: i === 0 }); vf.close();
    }
    this.exportBtn.textContent = "finishing…";
    await venc.flush(); venc.close();
    if (abuf) {
      const aenc = new AudioEncoder({ output: (c, m) => muxer.addAudioChunk(c, m), error: (e) => { throw e; } });
      aenc.configure({ codec: "opus", sampleRate: sr, numberOfChannels: nch, bitrate: 192000 });
      const s0 = Math.floor(t0 * sr), s1 = Math.min(abuf.length, Math.ceil(t1 * sr));
      const CH = [];
      for (let c = 0; c < nch; c++) CH.push(abuf.getChannelData(Math.min(c, abuf.numberOfChannels - 1)));
      const STEP = Math.round(sr / 50);
      for (let sp = s0; sp < s1; sp += STEP) {
        const len = Math.min(STEP, s1 - sp);
        const data = new Float32Array(len * nch);
        for (let c = 0; c < nch; c++) data.set(CH[c].subarray(sp, sp + len), c * len);
        const ad = new AudioData({ format: "f32-planar", sampleRate: sr, numberOfFrames: len, numberOfChannels: nch,
          timestamp: Math.round((sp - s0) / sr * 1e6), data });
        aenc.encode(ad); ad.close();
        if (aenc.encodeQueueSize > 8) await new Promise((r) => setTimeout(r, 2));
      }
      await aenc.flush(); aenc.close();
    }
    muxer.finalize();
    const type = mp4ok ? "video/mp4" : "video/webm", ext = mp4ok ? "mp4" : "webm";
    this.saveBlob(new Blob([target.buffer], { type }), this.exportName(t0, t1, ext));
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
    nodeType.prototype.onRemoved = function () { this.maiCompare?.destroy(); onRemoved?.apply(this, arguments); };
  },
});
