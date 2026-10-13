"use strict";
// Weld Annotator -- browser side.
// Left  = human annotation (exactly the rgb_masks of train / test), read-only.
// Right = "My edit": starts from the SAM-refined masks (or the last save) and is what gets saved.
// Label maps are Uint16Array, one object id per pixel, 0 = background.

const $ = (s) => document.querySelector(s);
const UNDO_LIMIT = 40;
const TINY = 20;                     // objects smaller than this are listed separately
const EDIT_TOOLS = ["brush", "erase", "box", "reclass", "human", "sam"];
const BRUSH_TOOLS = ["brush", "erase", "human", "sam"];

const S = {
  tree: [], key: null, W: 0, H: 0, N: 0, curExp: 0, onlyLeft: false,
  classes: {}, classOrder: [], photoRGB: null,
  human: null, sam: null, res: null,           // {labels, inst: Map}
  extra: null,                                 // {labels, inst: Map} objects only in the JSON
  conflicts: [], conflictMap: null,
  view: { scale: 1, tx: 0, ty: 0 }, mode: "overlay",
  compare: false, showExtra: false, paintOver: true, opacity: 0.6,
  tool: "select", brush: 6, cls: "droplet", from: "selected",
  sel: { side: null, id: 0 }, picked: new Set(), hoverConflict: 0, tab: "result",
  undo: [], redo: [], dirty: false, cursor: null,
};

// ------------------------------------------------------------------ helpers
async function unpack(b64) {
  const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint16Array(await new Response(stream).arrayBuffer());
}
async function pack(arr) {
  const stream = new Blob([arr.buffer]).stream().pipeThrough(new CompressionStream("deflate"));
  const buf = new Uint8Array(await new Response(stream).arrayBuffer());
  let s = "";
  for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
  return btoa(s);
}
const instMap = (list) => new Map(list.map((i) => [i.id, { humanId: null, samId: null, deleted: false, ...i }]));
const nextId = (map) => Math.max(0, ...map.keys()) + 1;
const cname = (cls) => (cls === "background" ? "background" : S.classes[cls] ? S.classes[cls].short : cls);
const ccol = (cls) => (S.classes[cls] ? S.classes[cls].color : [200, 200, 200]);
const swatch = (cls) => `<i class="sw" style="background:rgb(${cls === "background" ? "0,0,0" : ccol(cls)})"></i>`;
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function status(msg) { $("#status").textContent = msg; }
function currentReviewed() {
  if (!S.key || !S.tree.length) return false;
  const p = locate(S.key), im = S.tree[p.e] && S.tree[p.e].images[p.i];
  return !!(im && im.key === S.key && im.saved);
}
function setDirty(d) {
  S.dirty = d;
  const el = $("#saveState"), reviewed = currentReviewed();
  el.textContent = d ? "● Unsaved" : reviewed ? "✓ Reviewed" : "○ Not reviewed";
  el.className = "savestate " + (d ? "dirty" : reviewed ? "saved" : "dirty");
}
function countIds(labels) {
  const c = new Map();
  for (let i = 0; i < labels.length; i++) { const v = labels[i]; if (v) c.set(v, (c.get(v) || 0) + 1); }
  return c;
}

// ------------------------------------------------------------------ loading
async function loadTree() {
  const r = await fetch("/api/tree").then((x) => x.json());
  S.tree = r.tree;
  S.outRoot = r.outRoot;
  const base = (p) => p.split(/[\\/]/).slice(-1)[0] + "\\";
  $("#saveTo").textContent = `Saving to ${base(r.workRoot)} · complete folders move to ${base(r.outRoot)}`;
  $("#saveTo").title = `Each save goes to ${r.workRoot}. As soon as every image of an experiment folder is reviewed, ` +
    `that folder is moved to ${r.outRoot}. Nothing is written into train, test or final.`;
  const sel = $("#expSelect");
  sel.onchange = () => {
    const target = +sel.value;
    sel.value = pos().e;                                   // stay until the gate lets us go
    goTo(target, firstVisibleImage(target));
  };
  const want = new URLSearchParams(location.search).get("key");
  const start = want ? locate(want) : { e: 0, i: 0 };
  fillImages(start.e, start.i);
}
// ---- what the folder / image lists show ------------------------------------
const doneCount = (e) => S.tree[e].images.filter((x) => x.saved).length;
const folderStarted = (e) => { const d = doneCount(e); return d > 0 && d < S.tree[e].images.length; };
function folderVisible(e) { return !S.onlyLeft || e === S.curExp || folderStarted(e); }
function imageVisible(e, i) { return !S.onlyLeft || !S.tree[e].images[i].saved || (e === S.curExp && i === +$("#imgSelect").value); }
function firstVisibleImage(e) {
  const i = S.tree[e].images.findIndex((im) => !S.onlyLeft || !im.saved);
  return i < 0 ? 0 : i;
}
function rebuildLists() {
  const cur = S.curExp, curImg = +$("#imgSelect").value;
  $("#expSelect").innerHTML = S.tree.map((e, k) => (folderVisible(k) ? `<option value="${k}">${expLabel(e)}</option>` : "")).join("");
  $("#expSelect").value = cur;
  $("#imgSelect").innerHTML = S.tree[cur].images.map((im, k) =>
    (!S.onlyLeft || !im.saved || k === curImg ? `<option value="${k}">${im.saved ? "✓" : "○"} ${shortFrame(im.frame)}</option>` : "")).join("");
  $("#expSelect").title = `${S.tree[cur].split} / ${S.tree[cur].experiment} — ✓ complete, ◐ started, ○ not started`;
  $("#imgSelect").value = curImg;
}
const shortExp = (name) => name.replace(/^\d{8}_ER70S-6_/, "");
const shortFrame = (frame) => "f" + frame.replace(/^.*_f/, "").replace(/\.png$/, "");
function expLabel(e) {
  const done = e.images.filter((x) => x.saved).length;
  return `${e.published ? "✓" : done ? "◐" : "○"} ${e.split} · ${shortExp(e.experiment)}  ${done}/${e.images.length}`;
}
function fillImages(expIdx, imgIdx) {
  S.curExp = expIdx;
  $("#imgSelect").innerHTML = `<option value="${imgIdx}"></option>`;   // so rebuildLists keeps imgIdx
  $("#imgSelect").value = imgIdx;
  rebuildLists();
  openImage(S.tree[expIdx].images[imgIdx].key);
}
// the folder whose images are shown -- not what the folder dropdown momentarily says
function pos() { return { e: S.curExp, i: +$("#imgSelect").value }; }
function locate(key) {
  for (let e = 0; e < S.tree.length; e++) {
    const i = S.tree[e].images.findIndex((im) => im.key === key);
    if (i >= 0) return { e, i };
  }
  return { e: 0, i: 0 };
}
function updateProgress() {
  const { e, i } = pos(), exp = S.tree[e], imgs = exp.images;
  const done = imgs.filter((x) => x.saved).length;
  $("#pfill").style.width = `${(100 * done) / imgs.length}%`;
  $("#ptext").textContent = `${done}/${imgs.length}${exp.published ? " ✓" : ""}`;
  $("#progress").title = `Image ${i + 1} of ${imgs.length}. ${done} of ${imgs.length} images of this folder reviewed` +
    (exp.published ? " — folder complete and moved to the output." : `, ${imgs.length - done} left.`);
  rebuildLists();
}
async function openImage(key) {
  if (S.dirty && key !== S.key && !confirm("You have unsaved changes on this image. Discard them?")) {
    const p = locate(S.key); $("#imgSelect").value = p.i; return;
  }
  status("Loading…");
  try { await loadImage(key); }
  catch (err) {
    const msg = `Could not show this image: ${err.message}. If the tool was restarted or updated, reload the page (F5); make sure only one copy of annotate.py is running.`;
    status(msg); showCanvasMessage(msg);
  }
}
function showCanvasMessage(msg) {
  for (const side of ["left", "right"]) {
    const c = cv[side], r = c.getBoundingClientRect(), ctx = c.getContext("2d");
    c.width = Math.round(r.width); c.height = Math.round(r.height);
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#ffcf66"; ctx.font = "14px Segoe UI, sans-serif";
    const words = msg.split(" "); let line = "", y = 40;
    for (const w of words) {
      if (ctx.measureText(line + w).width > c.width - 40) { ctx.fillText(line, 20, y); line = ""; y += 20; }
      line += w + " ";
    }
    ctx.fillText(line, 20, y);
  }
}
async function loadImage(key) {
  const d = await fetch("/api/image?key=" + encodeURIComponent(key)).then((x) => x.json());
  if (d.error) throw new Error(d.error);
  if (!d.human || !d.human.extraLabels) throw new Error("the server sent data in an older format");
  S.key = key; S.W = d.width; S.H = d.height; S.N = d.width * d.height;
  S.classes = {}; S.classOrder = [];
  d.classes.forEach((c) => { S.classes[c.name] = c; if (c.name !== "background") S.classOrder.push(c.name); });
  S.human = { labels: await unpack(d.human.labels), inst: instMap(d.human.instances) };
  S.extra = { labels: await unpack(d.human.extraLabels), inst: instMap(d.human.extra) };
  S.sam = { labels: await unpack(d.sam.labels), inst: instMap(d.sam.instances) };
  S.res = { labels: await unpack(d.result.labels), inst: instMap(d.result.instances) };
  S.conflicts = d.conflicts; S.conflictMap = await unpack(d.conflictMap);
  S.conflicts.forEach((c) => { c.owner = c.default; });
  S.undo = []; S.redo = []; S.sel = { side: null, id: 0 }; S.picked = new Set(); S.hoverConflict = 0;
  $("#humanSub").textContent = `rgb_masks · ${d.frame}`;
  $("#editSub").textContent = d.result.source === "saved edit"
    ? `your last save · ${d.savedPath}` : `started from SAM-refined masks · ${d.sam.source}`;
  $("#extraCount").textContent = S.extra.inst.size ? `(${S.extra.inst.size})` : "(none)";
  buildClassPick(); buildLegend();
  await loadPhoto(key);
  setDirty(false); updateProgress();
  fit(); renderAll(); refreshSide();
  status(`${d.split} / ${d.experiment} / ${d.frame} · ${d.width}×${d.height} · Save writes to ${d.savedPath}`);
}
// ---- image adjustments (display only) ---------------------------------------
const ADJ_DEFAULT = { bright: 0, contrast: 0, local: 0 };
S.adj = (() => {
  try { return { ...ADJ_DEFAULT, ...JSON.parse(localStorage.getItem("weldImageAdj") || "{}") }; }
  catch (e) { return { ...ADJ_DEFAULT }; }
})();
// running-sum box blur of a W x H float image, radius r (horizontal then vertical)
function boxBlur(src, W, H, r) {
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length), n = 2 * r + 1;
  for (let y = 0; y < H; y++) {
    const row = y * W; let acc = 0;
    for (let k = -r; k <= r; k++) acc += src[row + Math.min(W - 1, Math.max(0, k))];
    for (let x = 0; x < W; x++) {
      tmp[row + x] = acc / n;
      acc += src[row + Math.min(W - 1, x + r + 1)] - src[row + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < W; x++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += tmp[Math.min(H - 1, Math.max(0, k)) * W + x];
    for (let y = 0; y < H; y++) {
      out[y * W + x] = acc / n;
      acc += tmp[Math.min(H - 1, y + r + 1) * W + x] - tmp[Math.max(0, y - r) * W + x];
    }
  }
  return out;
}
// brightness / contrast / local contrast applied to the original photo
function processPhoto() {
  const src = S.photoOrig; if (!src) return;
  const { bright, contrast, local } = S.adj;
  if (!bright && !contrast && !local) { S.photoRGB = src; return; }
  const W = S.W, H = S.H, N = W * H, out = new Uint8ClampedArray(src.length);
  let detail = null;
  if (local) {
    const lum = new Float32Array(N);
    for (let i = 0; i < N; i++) lum[i] = 0.299 * src[i * 4] + 0.587 * src[i * 4 + 1] + 0.114 * src[i * 4 + 2];
    const r = Math.max(4, Math.round(Math.min(W, H) / 40));
    const blur = boxBlur(boxBlur(lum, W, H, r), W, H, r);     // two passes ≈ Gaussian
    const amount = (local / 100) * 3;
    detail = new Float32Array(N);
    for (let i = 0; i < N; i++) detail[i] = (lum[i] - blur[i]) * amount;
  }
  const c = contrast >= 0 ? 1 + (contrast / 100) * 2 : 1 + contrast / 100;   // 0 .. 3x
  const b = bright * 1.28;
  for (let i = 0; i < N; i++) {
    const d = detail ? detail[i] : 0, o = i * 4;
    out[o] = (src[o] + d - 128) * c + 128 + b;
    out[o + 1] = (src[o + 1] + d - 128) * c + 128 + b;
    out[o + 2] = (src[o + 2] + d - 128) * c + 128 + b;
    out[o + 3] = 255;
  }
  S.photoRGB = out;
}
function syncAdjControls() {
  for (const [k, id] of [["bright", "adjBright"], ["contrast", "adjContrast"], ["local", "adjLocal"]]) {
    $("#" + id).value = S.adj[k]; $("#" + id + "Val").textContent = S.adj[k];
  }
}
let adjFrame = 0;
function setAdj(key, value, now = false) {
  S.adj[key] = value; syncAdjControls();
  try { localStorage.setItem("weldImageAdj", JSON.stringify(S.adj)); } catch (e) { /* ignore */ }
  cancelAnimationFrame(adjFrame);
  if (now) { processPhoto(); renderAll(); return; }         // reset / slider released
  adjFrame = requestAnimationFrame(() => { processPhoto(); renderAll(); });   // while dragging
}
function loadPhoto(key) {
  return new Promise((resolve) => {
    const im = new Image();
    im.onload = () => {
      const c = document.createElement("canvas"); c.width = S.W; c.height = S.H;
      const ctx = c.getContext("2d"); ctx.drawImage(im, 0, 0, S.W, S.H);
      S.photoOrig = ctx.getImageData(0, 0, S.W, S.H).data; processPhoto(); resolve();
    };
    im.onerror = () => { S.photoOrig = new Uint8ClampedArray(S.N * 4); processPhoto(); resolve(); };
    im.src = "/api/photo?key=" + encodeURIComponent(key);
  });
}
function step(delta) {
  const seq = [];
  S.tree.forEach((exp, e) => { if (folderVisible(e)) exp.images.forEach((_, i) => { if (imageVisible(e, i)) seq.push([e, i]); }); });
  const { e, i } = pos();
  const at = seq.findIndex(([a, b]) => a === e && b === i);
  const next = seq[at + delta];
  if (!next) { status(S.onlyLeft ? "No more images left to review in started folders." : "No more images."); return; }
  goTo(next[0], next[1]);
}
// Every move to another image goes through here. Leaving a folder that still
// has unreviewed images is not allowed: the gate dialog lists them instead.
function goTo(e, i) {
  const cur = pos().e;
  if (e !== cur && doneCount(cur) > 0) {            // only a folder you have started holds you
    const left = S.tree[cur].images.map((im, ii) => [im, ii]).filter(([im]) => !im.saved);
    if (left.length) { openGate(cur, left, [e, i]); return; }
  }
  if (S.dirty && !confirm("You have unsaved changes on this image. Discard them?")) return;
  S.dirty = false;
  if (S.curExp !== e) { fillImages(e, i); return; }
  $("#imgSelect").value = i; openImage(S.tree[e].images[i].key);
}
let gateTarget = null;
function openGate(cur, left, target) {
  gateTarget = target;
  const exp = S.tree[cur];
  $("#gateTitle").textContent = `${left.length} image${left.length > 1 ? "s" : ""} in this folder not reviewed yet`;
  $("#gateText").innerHTML = `Before moving to another folder, review every image of <b>${esc(exp.experiment)}</b> ` +
    `(Save or Save &amp; next on each), or save the remaining ones as they are. Click an image to open it:`;
  $("#gateList").innerHTML = left.map(([im, ii]) => `<button data-ii="${ii}">○ ${shortFrame(im.frame)}</button>`).join("");
  $("#gateSaveAll").textContent = `Save all ${left.length} as they are`;
  $("#gate").hidden = false;
}
function closeGate() { $("#gate").hidden = true; }
$("#gateList").addEventListener("click", (ev) => {
  const b = ev.target.closest("button"); if (!b) return;
  closeGate(); goTo(pos().e, +b.dataset.ii);
});
$("#gateFirst").onclick = () => { closeGate(); const first = $("#gateList button"); if (first) goTo(pos().e, +first.dataset.ii); };
$("#gateStay").onclick = closeGate;
$("#gateSaveAll").onclick = async () => {
  closeGate();
  if (await saveAll(false) && gateTarget) { const [e, i] = gateTarget; goTo(e, i); }
};
// Mark every remaining image of the current folder reviewed, as it is.
async function saveAll(ask = true) {
  const { e } = pos(), exp = S.tree[e];
  const left = exp.images.filter((im) => !im.saved).length;
  if (!left && !S.dirty) { status("Every image of this folder is already reviewed."); return true; }
  if (ask && !confirm(`Mark all ${left} not-yet-reviewed images of ${exp.experiment} as reviewed?\n\n` +
      "Each is saved exactly as it currently is (your last save, or the SAM result). " +
      "Unsaved changes on the open image are saved too.")) return false;
  if ((S.dirty || !currentReviewed()) && !(await save())) return false;
  document.querySelectorAll(".savebtn").forEach((b) => { b.disabled = true; });
  status(`Saving the remaining images of ${exp.experiment}…`);
  let ok = false;
  try {
    const r = await fetch("/api/save_folder", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ split: exp.split, experiment: exp.experiment }),
    }).then((x) => x.json());
    if (r.error) throw new Error(r.error);
    const t = await fetch("/api/tree").then((x) => x.json());
    S.tree = t.tree;
    setDirty(false); updateProgress();
    status(r.published ? `All ${r.total} images of ${r.experiment} are reviewed — the folder is in ${r.published}`
                       : `Saved ${r.saved} more images (${r.reviewed} of ${r.total} reviewed).`);
    ok = true;
  } catch (err) { status("Save all failed: " + err.message); alert("Save all failed: " + err.message); }
  document.querySelectorAll(".savebtn").forEach((b) => { b.disabled = false; });
  return ok;
}

// ------------------------------------------------------------------ rendering
const off = { left: document.createElement("canvas"), right: document.createElement("canvas") };
const img = { left: null, right: null };
const cv = { left: $("#cvLeft"), right: $("#cvRight") };

function colorTable(inst) {
  const n = Math.max(0, ...inst.keys()) + 1, t = new Uint8Array(n * 3);
  inst.forEach((v, id) => { if (!v.deleted) t.set(ccol(v.cls), id * 3); });
  return t;
}
function highlight(side) {
  const set = new Set(), s = S.sel;
  if (S.picked.size > 1) {
    S.picked.forEach((id) => {
      if (side === "right") set.add(id);
      else { const r = S.res.inst.get(id); if (r && r.humanId) set.add(r.humanId); }
    });
    return set;
  }
  if (!s.id) return set;
  if (side === "left") {
    if (s.side === "human") set.add(s.id);
    else if (s.side === "result") { const r = S.res.inst.get(s.id); if (r && r.humanId) set.add(r.humanId); }
  } else {
    if (s.side === "result") set.add(s.id);
    else if (s.side === "human") S.res.inst.forEach((v, id) => { if (!v.deleted && v.humanId === s.id) set.add(id); });
  }
  return set;
}
function renderRegion(side, x0 = 0, y0 = 0, x1 = S.W - 1, y1 = S.H - 1) {
  const W = S.W, H = S.H;
  x0 = Math.max(0, x0 - 1); y0 = Math.max(0, y0 - 1); x1 = Math.min(W - 1, x1 + 1); y1 = Math.min(H - 1, y1 + 1);
  const left = side === "left";
  const lab = left ? S.human.labels : S.res.labels;
  const col = colorTable(left ? S.human.inst : S.res.inst), hi = highlight(side);
  const cmp = !left && S.compare ? S.human.labels : null;
  const ex = left ? S.extra.labels : null;
  const exSel = left && S.sel.side === "extra" ? S.sel.id : 0;
  const cmap = left && S.hoverConflict ? S.conflictMap : null;
  if (!img[side] || img[side].width !== W) { off[side].width = W; off[side].height = H; img[side] = new ImageData(W, H); }
  const d = img[side].data, p = S.photoRGB, mode = S.mode, N = S.N;
  const dim = 1, a = S.opacity, na = 1 - a;
  const edge = (L, i, x, id) =>
    (x > 0 && L[i - 1] !== id) || (x < W - 1 && L[i + 1] !== id) || (i >= W && L[i - W] !== id) || (i < N - W && L[i + W] !== id);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = y * W + x, o = i * 4, id = lab[i];
      let r, g, b;
      if (mode === "mask") { r = col[id * 3]; g = col[id * 3 + 1]; b = col[id * 3 + 2]; }
      else {
        r = p[o] * dim; g = p[o + 1] * dim; b = p[o + 2] * dim;
        if (id && mode === "overlay") { r = r * na + col[id * 3] * a; g = g * na + col[id * 3 + 1] * a; b = b * na + col[id * 3 + 2] * a; }
      }
      if (cmap && cmap[i] === S.hoverConflict) { r = 255; g = 0; b = 255; }
      if (ex) {
        const e = ex[i];
        if (e && e === exSel) { r = r * 0.4 + 153; g = g * 0.4; b = b * 0.4 + 153; }
        if (e && (S.showExtra || e === exSel) && edge(ex, i, x, e) && (x + y) % 3 !== 0) { r = 255; g = 120; b = 255; }
      }
      if (cmp && cmp[i] && edge(cmp, i, x, cmp[i])) { r = 255; g = 255; b = 255; }
      if (id && (mode === "outline" || mode === "overlay") && edge(lab, i, x, id)) {
        if (hi.has(id)) { r = 255; g = 255; b = 255; } else { r = col[id * 3]; g = col[id * 3 + 1]; b = col[id * 3 + 2]; }
      } else if (id && hi.has(id) && edge(lab, i, x, id)) { r = 255; g = 255; b = 255; }
      d[o] = r; d[o + 1] = g; d[o + 2] = b; d[o + 3] = 255;
    }
  }
  off[side].getContext("2d").putImageData(img[side], 0, 0, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
  draw(side);
}
function renderAll() { if (!S.res) return; renderRegion("left"); renderRegion("right"); }

let boxDrag = null;
function draw(side) {
  const c = cv[side], rect = c.getBoundingClientRect();
  if (c.width !== Math.round(rect.width) || c.height !== Math.round(rect.height)) {
    c.width = Math.round(rect.width); c.height = Math.round(rect.height);
  }
  const ctx = c.getContext("2d"), v = S.view;
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, c.width, c.height);
  if (!S.res) return;
  ctx.imageSmoothingEnabled = false;
  ctx.setTransform(v.scale, 0, 0, v.scale, v.tx, v.ty);
  ctx.drawImage(off[side], 0, 0);
  ctx.lineWidth = 1.5 / v.scale;
  if (boxDrag && side === "right") {
    const b = boxDrag;
    ctx.strokeStyle = "#fff"; ctx.setLineDash([5 / v.scale, 4 / v.scale]);
    ctx.strokeRect(Math.min(b.x0, b.x1), Math.min(b.y0, b.y1), Math.abs(b.x1 - b.x0) + 1, Math.abs(b.y1 - b.y0) + 1);
    ctx.setLineDash([]);
  }
  const cur = S.cursor;
  if (cur) {
    const cx = cur.x + 0.5, cy = cur.y + 0.5;
    if (side === "right" && BRUSH_TOOLS.includes(S.tool)) {
      ctx.strokeStyle = S.tool === "erase" ? "#ff8f92" : "#fff";
      ctx.beginPath(); ctx.arc(cx, cy, S.brush, 0, Math.PI * 2); ctx.stroke();
    } else if (cur.side !== side) {           // mirror the cursor on the other image
      const r = 6 / v.scale;
      ctx.strokeStyle = "#fff";
      ctx.beginPath(); ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy); ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r); ctx.stroke();
    }
  }
}
function drawBoth() { draw("left"); draw("right"); }
function fit() {
  const r = cv.left.getBoundingClientRect();
  const s = Math.min(r.width / S.W, r.height / S.H) * 0.98;
  S.view = { scale: s, tx: (r.width - S.W * s) / 2, ty: (r.height - S.H * s) / 2 };
}
function zoomTo(labels, id) {
  let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
  for (let i = 0; i < labels.length; i++) if (labels[i] === id) {
    const x = i % S.W, y = (i - x) / S.W;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 < 0) return;
  const r = cv.left.getBoundingClientRect(), pad = 30;
  const s = Math.min(40, Math.min(r.width / (x1 - x0 + 1 + 2 * pad), r.height / (y1 - y0 + 1 + 2 * pad)));
  S.view = { scale: s, tx: r.width / 2 - ((x0 + x1) / 2) * s, ty: r.height / 2 - ((y0 + y1) / 2) * s };
  drawBoth();
}
function buildLegend() {
  $("#legend").innerHTML = S.classOrder.map((c) => `<span>${swatch(c)}${cname(c)}</span>`).join("") +
    `<span class="muted">· Colours are the RGB-mask colours · white = selected object${S.compare ? " / human outline" : ""}</span>`;
}

// ------------------------------------------------------------------ edit primitives
function snapshot() { return { labels: S.res.labels.slice(), inst: [...S.res.inst.values()].map((v) => ({ ...v })) }; }
function pushUndo() { S.undo.push(snapshot()); if (S.undo.length > UNDO_LIMIT) S.undo.shift(); S.redo = []; setDirty(true); }
function restore(snap) { S.res.labels = snap.labels; S.res.inst = new Map(snap.inst.map((v) => [v.id, v])); }
function undo() { if (!S.undo.length) return; S.redo.push(snapshot()); restore(S.undo.pop()); setDirty(true); renderAll(); refreshSide(); }
function redo() { if (!S.redo.length) return; S.undo.push(snapshot()); restore(S.redo.pop()); setDirty(true); renderAll(); refreshSide(); }

function newInstance(cls, extra = {}) {
  const id = nextId(S.res.inst);
  const k = [...S.res.inst.values()].filter((v) => v.cls === cls && / new \d+$/.test(v.name)).length + 1;
  S.res.inst.set(id, { id, name: `${cname(cls)} new ${k}`, cls, humanId: null, samId: null, deleted: false, ...extra });
  return id;
}
function humanToResult(hid) {
  for (const [id, v] of S.res.inst) if (!v.deleted && v.humanId === hid) return id;
  const h = S.human.inst.get(hid);
  return newInstance(h.cls, { name: h.name, humanId: hid });
}
function samToResult(sid) {
  for (const [id, v] of S.res.inst) if (!v.deleted && v.samId === sid) return id;
  for (const [id, v] of S.res.inst) if (v.deleted && v.samId === sid) { v.deleted = false; return id; }
  const s = S.sam.inst.get(sid);
  return newInstance(s.cls, { name: s.name, humanId: s.humanId, samId: sid });
}
function applyPixels(indices, fn) {
  const L = S.res.labels, W = S.W;
  let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
  for (const i of indices) {
    const v = fn(i);
    if (v === undefined || v === L[i]) continue;
    L[i] = v;
    const x = i % W, y = (i - x) / W;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  return x1 < 0 ? null : [x0, y0, x1, y1];
}
function* disk(cx, cy, r) {
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
    if (dx * dx + dy * dy > r * r) continue;
    const x = cx + dx, y = cy + dy;
    if (x >= 0 && y >= 0 && x < S.W && y < S.H) yield y * S.W + x;
  }
}
function* where(labels, id) { for (let i = 0; i < labels.length; i++) if (labels[i] === id) yield i; }
function* union(a, b) { const seen = new Set(); for (const i of a) { seen.add(i); yield i; } for (const i of b) if (!seen.has(i)) yield i; }
function cleanupEmpty() {
  const present = new Set(S.res.labels);
  S.res.inst.forEach((v, id) => { if (!present.has(id) && !v.deleted && !v.keepEmpty) v.deleted = true; });
}
function repaint(bb) { if (bb) { renderRegion("right", ...bb); if (S.compare || S.sel.id) renderRegion("left", ...bb); } }

// brush-type tools
let stroke = null;
function strokeTarget() {
  if (S.tool !== "brush") return 0;
  const sel = S.sel.side === "result" ? S.res.inst.get(S.sel.id) : null;
  if (sel && !sel.deleted && sel.cls === S.cls) return sel.id;
  const id = newInstance(S.cls, { keepEmpty: true });
  S.sel = { side: "result", id }; S.picked = new Set([id]);
  return id;
}
function noteTouch(id) {
  if (!id || id === stroke.target) return;
  const v = S.res.inst.get(id);
  if (v && !v.deleted && v.cls === S.res.inst.get(stroke.target).cls) stroke.touched.add(id);
}
// Merge objects of the same class into one. The object matched to a human
// object (else the largest) keeps its name. Returns the id that remains.
function mergeObjects(ids) {
  const live = [...new Set(ids)].map((id) => S.res.inst.get(id)).filter((v) => v && !v.deleted);
  if (live.length < 2) return live.length ? live[0].id : 0;
  const cnt = countIds(S.res.labels);
  live.sort((a, b) => (b.humanId ? 1 : 0) - (a.humanId ? 1 : 0) || (cnt.get(b.id) || 0) - (cnt.get(a.id) || 0));
  const keep = live[0], others = new Set(live.slice(1).map((v) => v.id));
  const L = S.res.labels;
  for (let i = 0; i < L.length; i++) if (others.has(L[i])) L[i] = keep.id;
  others.forEach((id) => { S.res.inst.get(id).deleted = true; });
  status(`Merged ${live.slice(1).map((v) => v.name).join(", ")} into ${keep.name}`);
  return keep.id;
}
function paintAt(x, y) {
  const L = S.res.labels, over = S.paintOver;
  let fn;
  if (S.tool === "brush") fn = (i) => {
    if (!(over || L[i] === 0)) return undefined;
    noteTouch(L[i]);
    const W = S.W, x = i % W;
    if (x > 0) noteTouch(L[i - 1]); if (x < W - 1) noteTouch(L[i + 1]);
    if (i >= W) noteTouch(L[i - W]); if (i < L.length - W) noteTouch(L[i + W]);
    return stroke.target;
  };
  else if (S.tool === "erase") fn = () => 0;
  else if (S.tool === "human") fn = (i) => { const h = S.human.labels[i]; return h ? humanToResult(h) : 0; };
  else if (S.tool === "sam") fn = (i) => { const s = S.sam.labels[i]; return s ? samToResult(s) : 0; };
  repaint(applyPixels(disk(x, y, S.brush), fn));
}
function paintLine(a, b) {
  const dist = Math.hypot(b.x - a.x, b.y - a.y), n = Math.max(1, Math.ceil(dist / Math.max(1, S.brush / 2)));
  for (let k = 1; k <= n; k++) paintAt(Math.round(a.x + (b.x - a.x) * k / n), Math.round(a.y + (b.y - a.y) * k / n));
}

// box tools
function boxRect(b) {
  return { x0: Math.max(0, Math.min(b.x0, b.x1)), x1: Math.min(S.W - 1, Math.max(b.x0, b.x1)),
           y0: Math.max(0, Math.min(b.y0, b.y1)), y1: Math.min(S.H - 1, Math.max(b.y0, b.y1)) };
}
function joinClassTarget(indices, excludeId) {
  const L = S.res.labels, W = S.W, touch = new Map();
  for (const i of indices) for (const j of [i - 1, i + 1, i - W, i + W]) {
    const id = L[j]; if (!id || id === excludeId) continue;
    const v = S.res.inst.get(id);
    if (v && !v.deleted && v.cls === S.cls) touch.set(id, (touch.get(id) || 0) + 1);
  }
  return touch.size ? [...touch.entries()].sort((a, b) => b[1] - a[1])[0][0] : newInstance(S.cls);
}
function boxFill(b) {
  const { x0, x1, y0, y1 } = boxRect(b), L = S.res.labels, W = S.W, empty = [];
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (L[y * W + x] === 0) empty.push(y * W + x);
  if (!empty.length) { status("Fill gap: there are no unlabelled (background) pixels inside that box."); return; }
  pushUndo();
  const target = joinClassTarget(empty, 0);
  repaint(applyPixels(empty, () => target));
  status(`Fill gap: ${empty.length} unlabelled pixels → ${S.res.inst.get(target).name}`);
  refreshSide();
}
function boxReclass(b) {
  const { x0, x1, y0, y1 } = boxRect(b), L = S.res.labels, W = S.W, hits = [];
  const selId = S.sel.side === "result" ? S.sel.id : 0;
  if (S.from === "selected" && !selId) { status("Change class: select an object in My edit first, or pick a class under 'Change pixels of'."); return; }
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const i = y * W + x, id = L[i];
    if (S.from === "background") { if (!id) hits.push(i); continue; }
    if (!id) continue;
    const v = S.res.inst.get(id); if (!v || v.deleted) continue;
    if (S.from === "selected" ? id === selId : v.cls === S.from) hits.push(i);
  }
  if (!hits.length) { status("Change class: no matching pixels inside that box."); return; }
  const clsAt = (i) => (L[i] ? S.res.inst.get(L[i]).cls : "background");
  if (hits.every((i) => clsAt(i) === S.cls)) { status(`Change class: those pixels are already ${cname(S.cls)}.`); return; }
  pushUndo();
  const target = S.cls === "background" ? 0 : joinClassTarget(hits, S.from === "selected" ? selId : 0);
  repaint(applyPixels(hits, () => target));
  cleanupEmpty();
  status(`Change class: ${hits.length} pixels → ${target ? S.res.inst.get(target).name : "background"}`);
  refreshSide();
}

// object actions
function act(kind, id) {
  const v = S.res.inst.get(id);
  pushUndo();
  if (kind === "delete") { applyPixels(where(S.res.labels, id), () => 0); v.deleted = true; }
  else if (kind === "restore") {
    v.deleted = false;
    if (v.samId) applyPixels(where(S.sam.labels, v.samId), () => id);
    else if (v.humanId) applyPixels(where(S.human.labels, v.humanId), () => id);
  } else if (kind === "useHuman") {
    applyPixels([...union(where(S.res.labels, id), where(S.human.labels, v.humanId))],
      (i) => { const h = S.human.labels[i]; return h ? humanToResult(h) : 0; });
  } else if (kind === "useSam") {
    applyPixels([...union(where(S.res.labels, id), where(S.sam.labels, v.samId))],
      (i) => { const s = S.sam.labels[i]; return s ? samToResult(s) : 0; });
  }
  cleanupEmpty(); renderAll(); refreshSide();
}
function copyHuman(hid) {
  pushUndo();
  applyPixels([...where(S.human.labels, hid)], () => humanToResult(hid));
  S.sel = { side: "result", id: humanToResult(hid) }; S.picked = new Set([S.sel.id]);
  cleanupEmpty(); renderAll(); refreshSide();
}
function copyExtra(eid) {
  const e = S.extra.inst.get(eid);
  pushUndo();
  const id = newInstance(e.cls, { name: e.name.replace(" (only in JSON)", "") });
  applyPixels([...where(S.extra.labels, eid)], () => id);
  S.sel = { side: "result", id }; S.picked = new Set([id]);
  cleanupEmpty(); renderAll(); refreshSide();
}
function setClass(id, cls) { pushUndo(); S.res.inst.get(id).cls = cls; renderAll(); refreshSide(); }
function setConflict(cid, owner) {
  const c = S.conflicts.find((x) => x.id === cid); c.owner = owner;
  const L = S.human.labels, M = S.conflictMap;
  for (let i = 0; i < L.length; i++) if (M[i] === cid) L[i] = owner;
  renderAll(); refreshSide();
}
function resetTo(which) {
  const what = which === "sam" ? "the SAM-refined mask" : "the human mask";
  if (!confirm(`Replace My edit with ${what}? (You can undo this.)`)) return;
  pushUndo();
  if (which === "sam") {
    S.res.labels = S.sam.labels.slice();
    S.res.inst = new Map([...S.sam.inst.values()].map((v) => [v.id, { ...v, deleted: false }]));
  } else {
    S.res.labels = S.human.labels.slice();
    S.res.inst = new Map([...S.human.inst.values()].map((v) => [v.id, { ...v, humanId: v.id, samId: null, deleted: false }]));
  }
  S.sel = { side: null, id: 0 }; S.picked = new Set(); renderAll(); refreshSide();
}
let saving = false;
async function save() {
  if (saving || !S.res) return false;
  saving = true;
  cleanupEmpty();
  // capture everything now: the user may move on while the request is running
  const key = S.key, W = S.W, H = S.H, labels = S.res.labels.slice();
  const present = new Set(labels);
  const instances = [...S.res.inst.values()].filter((v) => !v.deleted && present.has(v.id))
    .map(({ id, name, cls, humanId, samId }) => ({ id, name, cls, humanId, samId }));
  document.querySelectorAll(".savebtn").forEach((b) => { b.disabled = true; });
  status("Saving…");
  let ok = false;
  try {
    const r = await fetch("/api/save", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, width: W, height: H, labels: await pack(labels), instances }),
    }).then((x) => x.json());
    if (r.error) throw new Error(r.error);
    ok = true;
    if (S.key === key) setDirty(false);
    const p = locate(key); S.tree[p.e].images[p.i].saved = true;
    if (r.published) S.tree[p.e].published = true;
    if (S.key === key) setDirty(false);
    updateProgress();
    status(r.published
      ? `Saved. All ${r.total} images of ${r.experiment} are reviewed — the folder was moved to ${r.published}`
      : `Saved (${r.reviewed} of ${r.total} images of this folder reviewed). The folder moves to the output once all ${r.total} are reviewed.`);
  } catch (err) { status("Save failed: " + err.message); alert("Save failed: " + err.message); }
  document.querySelectorAll(".savebtn").forEach((b) => { b.disabled = false; });
  saving = false;
  return ok;
}
async function saveAndNext() { if (await save()) step(1); }
function nextUnreviewed() {
  const { e, i } = pos();
  const order = [];
  for (let k = 0; k < S.tree.length; k++) {
    const ee = (e + k) % S.tree.length;
    S.tree[ee].images.forEach((im, ii) => { if (!(ee === e && ii <= i && k === 0)) order.push([ee, ii]); });
  }
  // this folder first (after the current image, then from its start), then the next folders
  const here = order.filter(([ee]) => ee === e), rest = order.filter(([ee]) => ee !== e);
  for (let ii = 0; ii <= i; ii++) here.push([e, ii]);
  const hit = [...here, ...rest.filter(([ee]) => !S.onlyLeft || folderStarted(ee))].find(([ee, ii]) => !S.tree[ee].images[ii].saved);
  if (!hit) { status("Every image in every folder is reviewed."); return; }
  goTo(hit[0], hit[1]);
}

// ------------------------------------------------------------------ right side: card + lists
let sideTimer = null;
function refreshSide() { clearTimeout(sideTimer); sideTimer = setTimeout(() => { buildCard(); buildList(); }, 20); }
function classSelect(id, cur) {
  return `<select data-cls="${id}">${S.classOrder.map((c) => `<option value="${c}" ${c === cur ? "selected" : ""}>${cname(c)}</option>`).join("")}</select>`;
}
function mergeRow(v) {
  const cnt = countIds(S.res.labels);
  const same = [...S.res.inst.values()].filter((o) => !o.deleted && o.id !== v.id && o.cls === v.cls && cnt.get(o.id));
  if (!same.length) return "";
  return `<div class="row2">Merge with <select id="mergeWith">${same.map((o) =>
    `<option value="${o.id}">${esc(o.name)} (${cnt.get(o.id)} px)</option>`).join("")}</select>
    <button data-merge="1" title="Join that object into this one -- it becomes one object">Merge</button></div>`;
}
// ---- multi-selection (Shift+click) -----------------------------------------
function resultIdsFor(side, id) {
  if (side === "result") return [id];
  if (side === "human") return [...S.res.inst.values()].filter((v) => !v.deleted && v.humanId === id).map((v) => v.id);
  return [];
}
function select(side, id, shift) {
  if (!shift) {
    S.sel = id ? { side, id } : { side: null, id: 0 };
    S.picked = new Set(id ? resultIdsFor(side, id) : []);
  } else {
    const ids = resultIdsFor(side, id);
    if (!ids.length) {
      status(side === "human" ? "That human object is not in My edit yet — open its card and use “Copy it into My edit” first."
                              : "Only objects of My edit can be merged.");
      return;
    }
    const allIn = ids.every((x) => S.picked.has(x));
    ids.forEach((x) => (allIn ? S.picked.delete(x) : S.picked.add(x)));
    const first = [...S.picked][0];
    S.sel = first ? { side: "result", id: first } : { side: null, id: 0 };
  }
  if (S.sel.id) S.tab = S.picked.size > 1 || S.sel.side === "result" ? "result" : "human";
  renderAll(); refreshSide();
}
function pickedLive() { return [...S.picked].map((id) => S.res.inst.get(id)).filter((v) => v && !v.deleted); }
function mergePicked() {
  const list = pickedLive();
  if (list.length < 2) return;
  const cls = $("#mergeClass") ? $("#mergeClass").value : list[0].cls;
  pushUndo();
  list.forEach((v) => { v.cls = cls; });
  const keep = mergeObjects(list.map((v) => v.id));
  S.sel = { side: "result", id: keep }; S.picked = new Set([keep]);
  cleanupEmpty(); renderAll(); refreshSide();
}
function multiCard() {
  const list = pickedLive(), cnt = countIds(S.res.labels);
  const classes = [...new Set(list.map((v) => v.cls))];
  const mixed = classes.length > 1;
  return `<div class="title">${list.length} objects selected</div>
    <div class="picked">${list.map((v) => `<div>${swatch(v.cls)} ${esc(v.name)} <span class="muted">${(cnt.get(v.id) || 0).toLocaleString()} px</span></div>`).join("")}</div>
    ${mixed ? `<div class="note">They have different classes — the merged object gets the class you choose.</div>` : ""}
    <div class="row2">Merged class <select id="mergeClass">${S.classOrder.map((c) => `<option value="${c}" ${c === classes[0] ? "selected" : ""}>${cname(c)}</option>`).join("")}</select></div>
    <div class="actions">
      <button class="primary" data-multi="merge">⧉ Merge these ${list.length} into one object</button>
      <button class="danger" data-multi="delete">✕ Delete all ${list.length}</button>
      <button data-multi="clear">Clear selection</button>
    </div>
    <div class="sub">Shift+click objects (in the images or the list) to add or remove them.</div>`;
}
function buildCard() {
  const C = $("#selCard"), s = S.sel;
  if (S.res && pickedLive().length > 1) { C.innerHTML = multiCard(); return; }
  if (!S.res || !s.id) {
    C.innerHTML = `<div class="title">No object selected</div>
      <div class="sub">Use <b>Inspect (V)</b> and click an object in either image, or pick one from the list below. <b>Shift+click</b> more objects to merge them.</div>
      <div class="actions"><button data-reset="sam">Reset My edit to the SAM mask</button><button data-reset="human">Reset My edit to the human mask</button></div>`;
    return;
  }
  if (s.side === "result") {
    const v = S.res.inst.get(s.id); if (!v) { S.sel = { side: null, id: 0 }; return buildCard(); }
    const px = [...where(S.res.labels, v.id)].length, h = v.humanId ? S.human.inst.get(v.humanId) : null;
    C.innerHTML = `<div class="title">${swatch(v.cls)}${esc(v.name)}</div>
      <div class="sub">In <b>My edit</b> · ${px.toLocaleString()} px · ${h ? `matches human “${esc(h.name)}”` : "no matching human object"}</div>
      ${v.deleted ? `<div class="note">This object is deleted.</div><div class="actions"><button data-act="restore">Restore it</button></div>` : `
      <div class="row2">Class ${classSelect(v.id, v.cls)}</div>
      ${mergeRow(v)}
      <div class="actions">
        ${h ? `<button data-act="useHuman" title="Replace this object's pixels with the human object">⇦ Replace with the human version</button>` : ""}
        ${v.samId ? `<button data-act="useSam" title="Replace this object's pixels with the original SAM object">↺ Replace with the SAM version</button>` : ""}
        <button data-zoom="result">⌕ Zoom to it</button>
        <button class="danger" data-act="delete">✕ Delete this object</button>
      </div>`}`;
  } else if (s.side === "human") {
    const v = S.human.inst.get(s.id), px = [...where(S.human.labels, v.id)].length;
    const inEdit = [...S.res.inst.values()].filter((r) => !r.deleted && r.humanId === v.id);
    C.innerHTML = `<div class="title">${swatch(v.cls)}${esc(v.name)}</div>
      <div class="sub">In the <b>human</b> mask · ${cname(v.cls)} · ${px.toLocaleString()} px<br>
      In My edit: ${inEdit.length ? inEdit.map((r) => `“${esc(r.name)}”`).join(", ") : "<b>missing</b>"}</div>
      <div class="actions"><button data-copy="${v.id}">⇦ Copy it into My edit</button><button data-zoom="human">⌕ Zoom to it</button></div>`;
  } else if (s.side === "extra") {
    const v = S.extra.inst.get(s.id);
    C.innerHTML = `<div class="title">${swatch(v.cls)}${esc(v.name)}</div>
      <div class="sub">This object is in the COCO JSON but <b>not in the human RGB mask</b> (${v.pixels.toLocaleString()} px, shown in pink on the left).</div>
      <div class="actions"><button data-extra="${v.id}">⇦ Copy it into My edit</button><button data-zoom="extra">⌕ Zoom to it</button></div>`;
  }
}
function buildList() {
  const L = $("#list");
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === S.tab));
  $("#tabs [data-tab=result]").textContent = `My edit (${[...S.res.inst.values()].filter((v) => !v.deleted).length})`;
  $("#tabs [data-tab=human]").textContent = `Human (${S.human.inst.size + S.extra.inst.size})`;
  $("#tabs [data-tab=conflicts]").textContent = `Overlaps (${S.conflicts.length})`;
  const item = (side, v, px, flag = "") =>
    `<div class="item ${(S.sel.side === side && S.sel.id === v.id) || (side === "result" && S.picked.size > 1 && S.picked.has(v.id)) ? "on" : ""} ${v.deleted ? "gone" : ""}" data-side="${side}" data-id="${v.id}">
      ${swatch(v.cls)}<span class="nm">${esc(v.name)}</span>${flag}<span class="px">${px.toLocaleString()} px</span></div>`;
  if (S.tab === "result") {
    const cnt = countIds(S.res.labels), all = [...S.res.inst.values()];
    const live = all.filter((v) => !v.deleted && cnt.get(v.id)).sort((a, b) => cnt.get(b.id) - cnt.get(a.id));
    const big = live.filter((v) => cnt.get(v.id) >= TINY), tiny = live.filter((v) => cnt.get(v.id) < TINY);
    const dead = all.filter((v) => v.deleted);
    const row = (v) => item("result", v, cnt.get(v.id) || 0, v.humanId ? "" : `<span class="flag">new</span>`);
    L.innerHTML = (live.length ? big.map(row).join("") : `<div class="empty">My edit is empty.</div>`) +
      (tiny.length ? `<div class="sect">Tiny pieces (under ${TINY} px)</div>${tiny.map(row).join("")}` : "") +
      (dead.length ? `<div class="sect">Deleted</div>${dead.map((v) => item("result", v, 0)).join("")}` : "");
  } else if (S.tab === "human") {
    const cnt = countIds(S.human.labels);
    const missing = (v) => ![...S.res.inst.values()].some((r) => !r.deleted && r.humanId === v.id);
    L.innerHTML = [...S.human.inst.values()].sort((a, b) => (cnt.get(b.id) || 0) - (cnt.get(a.id) || 0))
      .map((v) => item("human", v, cnt.get(v.id) || 0, missing(v) ? `<span class="flag">not in my edit</span>` : "")).join("") +
      (S.extra.inst.size ? `<div class="sect">Only in the JSON (not in the RGB mask)</div>` +
        [...S.extra.inst.values()].map((v) => item("extra", v, v.pixels)).join("") : "");
  } else {
    L.innerHTML = S.conflicts.length
      ? `<div class="empty">Two human objects of the same class overlap in the JSON. Pick which one owns the shared pixels (hover to see them in pink). This changes the human side and what “Copy from human” copies.</div>` +
        S.conflicts.map((c) => {
          const a = S.human.inst.get(c.a), b = S.human.inst.get(c.b);
          return `<div class="conf" data-conf="${c.id}">${swatch(a.cls)} <b>${esc(a.name)}</b> vs <b>${esc(b.name)}</b> · ${c.pixels} px
            <div class="btns"><button data-own="${c.id}:${c.a}" class="${c.owner === c.a ? "on" : ""}">${esc(a.name)}</button>
            <button data-own="${c.id}:${c.b}" class="${c.owner === c.b ? "on" : ""}">${esc(b.name)}</button></div></div>`;
        }).join("")
      : `<div class="empty">No overlapping human objects in this image.</div>`;
  }
}
$("#selCard").addEventListener("click", (ev) => {
  const t = ev.target.closest("button"); if (!t) return;
  if (t.dataset.multi) {
    const list = pickedLive();
    if (t.dataset.multi === "merge") mergePicked();
    else if (t.dataset.multi === "delete") {
      pushUndo();
      list.forEach((v) => { applyPixels(where(S.res.labels, v.id), () => 0); v.deleted = true; });
      S.picked = new Set(); S.sel = { side: null, id: 0 };
      cleanupEmpty(); renderAll(); refreshSide();
    } else { S.picked = new Set(); S.sel = { side: null, id: 0 }; renderAll(); refreshSide(); }
    return;
  }
  if (t.dataset.merge) {
    pushUndo();
    const keep = mergeObjects([S.sel.id, +$("#mergeWith").value]);
    S.sel = { side: "result", id: keep }; S.picked = new Set([keep]);
    cleanupEmpty(); renderAll(); refreshSide();
    return;
  }
  if (t.dataset.act) act(t.dataset.act, S.sel.id);
  else if (t.dataset.copy) copyHuman(+t.dataset.copy);
  else if (t.dataset.extra) copyExtra(+t.dataset.extra);
  else if (t.dataset.reset) resetTo(t.dataset.reset);
  else if (t.dataset.zoom) zoomTo({ result: S.res.labels, human: S.human.labels, extra: S.extra.labels }[t.dataset.zoom], S.sel.id);
});
$("#selCard").addEventListener("change", (ev) => { if (ev.target.dataset.cls) setClass(+ev.target.dataset.cls, ev.target.value); });
$("#list").addEventListener("click", (ev) => {
  const own = ev.target.closest("[data-own]");
  if (own) { const [cid, o] = own.dataset.own.split(":").map(Number); setConflict(cid, o); return; }
  const it = ev.target.closest(".item"); if (!it) return;
  select(it.dataset.side, +it.dataset.id, ev.shiftKey);
});
$("#list").addEventListener("mouseover", (ev) => {
  const c = ev.target.closest("[data-conf]"), id = c ? +c.dataset.conf : 0;
  if (id !== S.hoverConflict) { S.hoverConflict = id; renderRegion("left"); }
});
document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => { S.tab = b.dataset.tab; buildList(); });

// ------------------------------------------------------------------ tools & options
const HINTS = {
  select: () => "<b>Inspect</b> — click an object in either image to select it; its card (top right) lets you change its class, replace it or delete it. <b>Shift+click</b> more objects to select several — then the card has a <b>Merge</b> button.",
  pan: () => "<b>Pan</b> — drag to move the view. The mouse wheel zooms with every tool; right-drag or holding Space also pans.",
  brush: () => `<b>Brush</b> — drag on <b>My edit</b> to paint <b>${cname(S.cls)}</b>. If a ${cname(S.cls)} object is selected, the paint extends it; otherwise it becomes a new object. Painting that touches other ${cname(S.cls)} objects merges them into one.`,
  erase: () => "<b>Eraser</b> — drag on <b>My edit</b> to remove labels (the pixels become background).",
  box: () => `<b>Fill gap in box</b> — drag a box on <b>My edit</b>. Only unlabelled (black) pixels inside it become <b>${cname(S.cls)}</b>; they join the ${cname(S.cls)} object they touch.`,
  reclass: () => `<b>Change class in box</b> — drag a box on <b>My edit</b>. Pixels of <b>${S.from === "selected" ? "the selected object" : cname(S.from)}</b> inside it become <b>${cname(S.cls)}</b>. Everything else stays.`,
  human: () => "<b>Copy from human</b> — drag on <b>My edit</b>: under the brush, the human labels are copied in.",
  sam: () => "<b>Copy from SAM</b> — drag on <b>My edit</b>: under the brush, the original SAM-refined labels are copied back.",
};
function setTool(t) {
  S.tool = t;
  document.querySelectorAll("#tools button").forEach((b) => b.classList.toggle("on", b.dataset.tool === t));
  const needsClass = ["brush", "box", "reclass"].includes(t);
  $("#classRow").hidden = !needsClass;
  $("#classLabel").textContent = t === "reclass" ? "Change them to" : t === "box" ? "Fill with" : "Paint";
  $("#fromRow").hidden = t !== "reclass";
  $("#sizeRow").hidden = !BRUSH_TOOLS.includes(t);
  $("#overRow").hidden = t !== "brush";
  $("#optBlock").hidden = !(needsClass || BRUSH_TOOLS.includes(t));
  if (S.classOrder.length) buildClassPick();
  $("#toolHint").innerHTML = HINTS[t]();
  cv.right.style.cursor = cv.left.style.cursor = t === "pan" ? "grab" : EDIT_TOOLS.includes(t) ? "crosshair" : "default";
  cv.left.style.cursor = t === "pan" ? "grab" : "default";
  drawBoth();
}
document.querySelectorAll("#tools button").forEach((b) => b.onclick = () => setTool(b.dataset.tool));
function buildClassPick() {
  // background is a choice only for "Change class in box" (the eraser covers painting it)
  const list = S.tool === "reclass" ? [...S.classOrder, "background"] : S.classOrder;
  if (!list.includes(S.cls)) S.cls = S.classOrder.includes(S.lastCls) ? S.lastCls : S.classOrder[0];
  $("#classPick").innerHTML = list.map((c) =>
    `<button data-c="${c}" class="${c === S.cls ? "on" : ""}">${swatch(c)}${cname(c)}</button>`).join("");
  $("#fromSelect").innerHTML = `<option value="selected">the selected object</option>` +
    S.classOrder.map((c) => `<option value="${c}">${cname(c)}</option>`).join("") +
    `<option value="background">background (unlabelled)</option>`;
  $("#fromSelect").value = S.from;
}
$("#classPick").addEventListener("click", (ev) => {
  const b = ev.target.closest("button"); if (!b) return;
  S.cls = b.dataset.c;
  if (S.cls !== "background") S.lastCls = S.cls;
  document.querySelectorAll("#classPick button").forEach((x) => x.classList.toggle("on", x === b));
  $("#toolHint").innerHTML = HINTS[S.tool]();
});
$("#fromSelect").onchange = (e) => { S.from = e.target.value; $("#toolHint").innerHTML = HINTS[S.tool](); };
$("#brushSize").oninput = (e) => { S.brush = +e.target.value; $("#brushVal").textContent = S.brush; drawBoth(); };
$("#paintOver").onchange = (e) => { S.paintOver = e.target.checked; };
function setMode(m) {
  S.mode = m;
  $("#opacityRow").hidden = m !== "overlay";
  document.querySelectorAll("#viewSeg button").forEach((b) => b.classList.toggle("on", b.dataset.view === m));
  renderAll();
}
document.querySelectorAll("#viewSeg button").forEach((b) => b.onclick = () => setMode(b.dataset.view));
$("#compare").onchange = (e) => { S.compare = e.target.checked; buildLegend(); renderRegion("right"); };
$("#showExtra").onchange = (e) => { S.showExtra = e.target.checked; renderRegion("left"); };
for (const [id, key] of [["adjBright", "bright"], ["adjContrast", "contrast"], ["adjLocal", "local"]]) {
  $("#" + id).oninput = (e) => setAdj(key, +e.target.value);
  $("#" + id).onchange = (e) => setAdj(key, +e.target.value, true);
}
$("#adjReset").onclick = () => { S.adj = { ...ADJ_DEFAULT }; setAdj("bright", 0, true); };
syncAdjControls();
$("#opacity").oninput = (e) => { S.opacity = +e.target.value / 100; renderAll(); };
$("#fitBtn").onclick = () => { fit(); drawBoth(); };
$("#undoBtn").onclick = undo;
$("#redoBtn").onclick = redo;
$("#saveBtn").onclick = save;
$("#saveNextBtn").onclick = saveAndNext;
$("#nextOpenBtn").onclick = nextUnreviewed;
$("#saveAllBtn").onclick = () => saveAll(true);
$("#onlyLeft").onchange = (ev) => {
  S.onlyLeft = ev.target.checked;
  if (!S.onlyLeft) { rebuildLists(); status("Showing every folder and every image."); return; }
  const cur = S.curExp;
  if (folderStarted(cur)) {                                  // stay here, jump to its first open image if needed
    rebuildLists();
    if (S.tree[cur].images[pos().i].saved) goTo(cur, firstVisibleImage(cur));
    status("Showing only the images left to review in folders you have started.");
    return;
  }
  const e = S.tree.findIndex((_, k) => folderStarted(k));
  if (e < 0) {
    ev.target.checked = S.onlyLeft = false; rebuildLists();
    status("No started folder has images left — every folder you started is complete."); return;
  }
  goTo(e, firstVisibleImage(e));
  status("Showing only the images left to review in folders you have started.");
};
$("#prevBtn").onclick = () => step(-1);
$("#nextBtn").onclick = () => step(1);
$("#imgSelect").onchange = (e) => openImage(S.tree[pos().e].images[+e.target.value].key);
// while the gate dialog is open, no shortcut may move away behind it
window.addEventListener("keydown", (ev) => {
  if ($("#gate").hidden) return;
  if (ev.key === "Escape") closeGate();
  ev.stopImmediatePropagation();
}, true);
$("#helpBtn").onclick = () => { $("#help").hidden = false; };
$("#helpClose").onclick = () => { $("#help").hidden = true; try { localStorage.setItem("weldHelpSeen", "1"); } catch (e) { /* ignore */ } };

// ------------------------------------------------------------------ pointer
function toImage(side, ev) {
  const r = cv[side].getBoundingClientRect(), v = S.view;
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  return { side, mx, my, x: Math.floor((mx - v.tx) / v.scale), y: Math.floor((my - v.ty) / v.scale) };
}
let spaceDown = false, panDrag = null;
for (const side of ["left", "right"]) {
  const c = cv[side];
  c.addEventListener("contextmenu", (e) => e.preventDefault());
  c.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const p = toImage(side, ev), v = S.view, f = Math.pow(1.0015, -ev.deltaY);
    const s = Math.min(60, Math.max(0.2, v.scale * f)), k = s / v.scale;
    v.tx = p.mx - (p.mx - v.tx) * k; v.ty = p.my - (p.my - v.ty) * k; v.scale = s; drawBoth();
  }, { passive: false });
  c.addEventListener("pointerdown", (ev) => {
    if (!S.res) return;
    c.setPointerCapture(ev.pointerId);
    const p = toImage(side, ev);
    if (ev.button === 1 || ev.button === 2 || spaceDown || S.tool === "pan") { panDrag = { mx: p.mx, my: p.my }; return; }
    if (ev.button !== 0) return;
    if (S.tool === "select") {
      const inside = p.x >= 0 && p.y >= 0 && p.x < S.W && p.y < S.H, i = p.y * S.W + p.x;
      let sel = { side: null, id: 0 };
      if (inside && side === "left") {
        const ex = S.extra.labels[i];
        if (ex && S.showExtra && !S.human.labels[i]) sel = { side: "extra", id: ex };
        else if (S.human.labels[i]) sel = { side: "human", id: S.human.labels[i] };
      } else if (inside && S.res.labels[i]) sel = { side: "result", id: S.res.labels[i] };
      if (ev.shiftKey && sel.id) { select(sel.side, sel.id, true); return; }
      select(sel.side, sel.id, false); return;
    }
    if (side === "left") { status("The left image is the read-only reference — edit tools work on My edit (right)."); return; }
    if (S.tool === "box" || S.tool === "reclass") { boxDrag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y }; drawBoth(); return; }
    pushUndo();
    stroke = { target: strokeTarget(), last: p, touched: new Set() };
    paintAt(p.x, p.y);
  });
  c.addEventListener("pointermove", (ev) => {
    const p = toImage(side, ev);
    S.cursor = { side, x: p.x, y: p.y };
    if (panDrag) { S.view.tx += p.mx - panDrag.mx; S.view.ty += p.my - panDrag.my; panDrag.mx = p.mx; panDrag.my = p.my; drawBoth(); }
    else if (boxDrag) { boxDrag.x1 = p.x; boxDrag.y1 = p.y; drawBoth(); }
    else if (stroke) { paintLine(stroke.last, p); stroke.last = p; drawBoth(); }
    else drawBoth();
    hoverStatus(p);
  });
  const end = () => {
    if (boxDrag) {
      const b = boxDrag; boxDrag = null;
      if (S.tool === "reclass") boxReclass(b); else boxFill(b);
      drawBoth();
    }
    if (stroke) {
      const t = S.res.inst.get(stroke.target); if (t) delete t.keepEmpty;
      if (S.tool === "brush" && stroke.touched.size) {
        // painting that touches other objects of the same class joins them
        const keep = mergeObjects([stroke.target, ...stroke.touched]);
        S.sel = { side: "result", id: keep }; S.picked = new Set([keep]);
        renderAll();
      }
      stroke = null; cleanupEmpty(); refreshSide();
    }
    panDrag = null;
  };
  c.addEventListener("pointerup", end);
  c.addEventListener("pointercancel", end);
  c.addEventListener("pointerleave", () => { S.cursor = null; drawBoth(); });
}
function hoverStatus(p) {
  if (!S.res || p.x < 0 || p.y < 0 || p.x >= S.W || p.y >= S.H) return;
  const i = p.y * S.W + p.x, h = S.human.inst.get(S.human.labels[i]), r = S.res.inst.get(S.res.labels[i]);
  const d = (o) => (o ? `${o.name}` : "background");
  status(`x ${p.x}, y ${p.y}   ·   human: ${d(h)}   ·   my edit: ${d(r)}`);
}

// ------------------------------------------------------------------ keyboard
document.addEventListener("change", (ev) => { if (ev.target.tagName === "SELECT") ev.target.blur(); });
window.addEventListener("keydown", (ev) => {
  if (!$("#help").hidden) { if (ev.key === "Escape" || ev.key === "Enter") $("#helpClose").click(); return; }
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName) && !ev.ctrlKey) return;
  const k = ev.key.toLowerCase();
  if (ev.ctrlKey && k === "s") { ev.preventDefault(); save(); return; }
  if (ev.ctrlKey && ev.key === "Enter") { ev.preventDefault(); saveAndNext(); return; }
  if (ev.ctrlKey && k === "z" && !ev.shiftKey) { ev.preventDefault(); undo(); return; }
  if (ev.ctrlKey && (k === "y" || (k === "z" && ev.shiftKey))) { ev.preventDefault(); redo(); return; }
  if (ev.ctrlKey) return;
  if (k === " ") { spaceDown = true; ev.preventDefault(); return; }
  const tools = { v: "select", b: "brush", e: "erase", g: "box", r: "reclass", h: "human", s: "sam" };
  if (tools[k]) return setTool(tools[k]);
  if (k === "f") { fit(); drawBoth(); }
  else if (k === "o") { $("#compare").checked = !S.compare; $("#compare").dispatchEvent(new Event("change")); }
  else if (k === "1") setMode("photo"); else if (k === "2") setMode("outline");
  else if (k === "3") setMode("overlay"); else if (k === "4") setMode("mask");
  else if (k === "[" || k === "]") {
    S.brush = Math.min(40, Math.max(1, S.brush + (k === "]" ? 1 : -1)));
    $("#brushSize").value = S.brush; $("#brushVal").textContent = S.brush; drawBoth();
  }
  else if (ev.key === "ArrowLeft") step(-1); else if (ev.key === "ArrowRight") step(1);
  else if (ev.key === "Escape") { S.sel = { side: null, id: 0 }; S.picked = new Set(); renderAll(); refreshSide(); }
});
window.addEventListener("keyup", (ev) => { if (ev.key === " ") spaceDown = false; });
window.addEventListener("resize", () => drawBoth());
window.addEventListener("beforeunload", (ev) => { if (S.dirty) { ev.preventDefault(); ev.returnValue = ""; } });

// ------------------------------------------------------------------ start
setTool("select"); setMode("overlay");
try { if (!localStorage.getItem("weldHelpSeen") && !/selftest|nohelp/.test(location.search)) $("#help").hidden = false; } catch (e) { /* ignore */ }
loadTree().catch((e) => status("Could not load: " + e));
if (location.search.includes("selftest")) {
  const t = document.createElement("script"); t.src = "/static/selftest.js"; document.body.appendChild(t);
}
