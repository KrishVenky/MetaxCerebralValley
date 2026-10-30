"use strict";
// Drawing Lazarus demo UI. Static, dependency-free: SVG over the analysed tile image.
// The connectivity/isolation logic below is a JS port of lazarus/simulate.py and must stay consistent
// with it (closed valves are reachable but block passage; junctions pass through; the source always
// expands). It is connectivity analysis only, not hydraulics: flow direction is unknown.

const NS = "http://www.w3.org/2000/svg";
const KINDS = { valve: "Valve", instrument: "Instrument", equipment: "Equipment", fitting: "Fitting", marker: "Marker", unknown: "Unknown" };
const FINDING = {
  isolated: "Isolated symbol", degree: "Unexpected connections", tag_code: "Tag not in legend",
  tag_missing: "Tag missing", tag_prefix: "Tag prefix mismatch", low_conf: "Low confidence", dangling_end: "Dangling pipe end",
  open_end: "Open pipe end",
};
// Open ends are informational (a pipe that leaves the drawn area or just stops); everything else needs a person.
const INFO = new Set(["open_end"]);
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const el = (tag, attrs = {}, parent) => {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
};

/* ---------- simulation (port of lazarus/simulate.py) ---------- */
function buildNet(graph) {
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  const adj = new Map(graph.nodes.map((n) => [n.id, new Set()]));
  for (const [a, b] of graph.edges) if (adj.has(a) && adj.has(b) && a !== b) { adj.get(a).add(b); adj.get(b).add(a); }
  return { nodes, adj };
}

function reach(net, source, blocked) {
  const seen = new Set([source]), queue = [source];
  while (queue.length) {
    const u = queue.shift();
    if (blocked.has(u) && u !== source) continue;
    for (const v of net.adj.get(u)) if (!seen.has(v)) { seen.add(v); queue.push(v); }
  }
  return seen;
}

function components(net) {
  const seen = new Set(), out = [];
  for (const s of net.adj.keys()) if (!seen.has(s)) { const c = reach(net, s, new Set()); c.forEach((i) => seen.add(i)); out.push(c); }
  return out;
}

function effectiveDegree(net, node) {
  const seen = new Set([node]), out = new Set(), stack = [node];
  while (stack.length) for (const v of net.adj.get(stack.pop())) {
    if (seen.has(v)) continue;
    seen.add(v);
    if (net.nodes.get(v).kind === "junction") stack.push(v); else out.add(v);
  }
  return out.size;
}

// Equipment in the biggest connected network, major equipment by name first, then the biggest hub (junction-transparent degree), then lowest id.
function suggestSources(net, limit = 3) {
  const size = new Map();
  for (const c of components(net)) c.forEach((i) => size.set(i, c.size));
  const real = [...net.nodes.values()].filter((n) => n.kind !== "junction").map((n) => n.id);
  const equip = real.filter((i) => net.nodes.get(i).kind === "equipment");
  const pool = equip.length ? equip : real;
  const major = (i) => (/vessel|tank|pump|exchanger|compressor|column|reactor|drum/i.test(net.nodes.get(i).name || "") ? 0 : 1);
  const key = (i) => [-size.get(i), major(i), -effectiveDegree(net, i), i];
  const cmp = (a, b) => { const x = key(a), y = key(b); return x[0] - y[0] || x[1] - y[1] || x[2] - y[2] || x[3] - y[3]; };
  return pool.sort(cmp).slice(0, limit);
}

function isolate(net, closed, source) {
  const shut = new Set([...closed].filter((i) => net.nodes.get(i)?.kind === "valve"));
  const base = reach(net, source, new Set()), now = reach(net, source, shut);
  const all = [...net.nodes.keys()];
  return {
    now, base, shut,
    lost: new Set(all.filter((i) => base.has(i) && !now.has(i))),
    baseline: new Set(all.filter((i) => !base.has(i))),
  };
}

const symbols = (net, ids) => [...ids].filter((i) => net.nodes.get(i).kind !== "junction").sort((a, b) => a - b);

function rankValves(net, source) {
  const rows = [];
  for (const n of net.nodes.values()) if (n.kind === "valve") {
    rows.push({ id: n.id, lost: symbols(net, isolate(net, [n.id], source).lost).length });
  }
  return rows.sort((a, b) => b.lost - a.lost || a.id - b.id);
}

/* ---------- state ---------- */
let DATA, D, net, closed, source, autoSource, view, hotFinding = null, findFilter = null;
let work, log = [], picking = null, sigLinked = new Set();   // work: editable copy of the graph; log: review steps (undo, persistence)
let W = 1792, H = 1140, U = 1;   // sheet size in pixels; U scales marker sizes so they read the same on any sheet
const nodeEls = new Map(), edgeEls = [], juncEls = new Map(), findEls = [];
let gHover = null;
const svg = $("svg"), wrap = svg.parentElement;
let dragged = false;

const label = (n) => n.tag || `#${n.id}`;
const center = (n) => [(n.box[0] + n.box[2]) / 2, (n.box[1] + n.box[3]) / 2];

async function init() {
  DATA = await (await fetch("data/index.json")).json();
  const sel = $("drawing");
  const opt = (d, i) => `<option value="${i}">${esc(d.name)} — ${d.graph.nodes.filter((n) => n.kind !== "junction").length} symbols</option>`;
  sel.innerHTML = ["benchmark", "real scan"].map((g) => {
    const items = DATA.drawings.map((d, i) => [d, i]).filter(([d]) => d.group === g);
    return items.length ? `<optgroup label="${g === "benchmark" ? "Benchmark sheets (held out)" : "Real scans (no ground truth)"}">${items.map(([d, i]) => opt(d, i)).join("")}</optgroup>` : "";
  }).join("");
  const want = DATA.drawings.findIndex((d) => d.name === location.hash.slice(1));
  sel.value = want >= 0 ? want : 0;
  sel.onchange = () => load(+sel.value);
  ["edges", "junc", "find", "sig", "open", "region"].forEach((k) => ($("t-" + k).onchange = (e) => wrap.classList.toggle("hide-" + k, !e.target.checked)));
  $("reset-view").onclick = () => flyTo({ x: 0, y: 0, w: W, h: H });
  wrap.classList.add("hide-open");
  $("reopen").onclick = () => { closed.clear(); update(); };
  $("undo").onclick = undo;
  $("export").onclick = exportCorrections;
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") { if (picking) stopPicking(); else { closed.clear(); update(); } } });
  renderLegend();
  load(+sel.value);
}

function load(i) {
  D = DATA.drawings[i];
  [W, H] = D.size;
  U = W / 1792;
  wrap.style.aspectRatio = `${W} / ${H}`;
  svg.style.setProperty("--u", U);
  location.hash = D.name;
  sigLinked = new Set((D.graph.signal_edges || []).flat());
  log = loadLog();
  picking = null;
  replay();
  closed = new Set();
  autoSource = suggestSources(net)[0];
  source = autoSource;
  hotFinding = findFilter = null;
  view = { x: 0, y: 0, w: W, h: H };
  buildScene();
  update();
}

/* ---------- editable graph + review log ---------- */
const logKey = () => "lazarus-review:" + D.name;
function loadLog() {
  try { const v = JSON.parse(localStorage.getItem(logKey()) || "[]"); return Array.isArray(v) ? v : []; } catch { return []; }
}
function saveLog() { try { localStorage.setItem(logKey(), JSON.stringify(log)); } catch { /* storage unavailable: review still works for this visit */ } }

function pristine() {
  return {
    nodes: D.graph.nodes.map((n) => ({ ...n })),
    edges: D.graph.edges.map(([a, b], i) => ({ a, b, path: (D.paths && D.paths[i]) || null })),
  };
}
function applyOp(o) {
  const n = work.nodes.find((x) => x.id === o.id);
  const same = (e) => (e.a === o.a && e.b === o.b) || (e.a === o.b && e.b === o.a);
  if (o.op === "link") { if (!work.edges.some(same)) work.edges.push({ a: o.a, b: o.b, path: null, added: true }); return; }
  if (o.op === "unlink") { work.edges = work.edges.filter((e) => !same(e)); return; }
  if (!n) return;
  if (o.op === "kind") { n.kind = o.kind; n.name = o.name; if (o.legend_id != null) n.legend_id = o.legend_id; n.corrected = true; }
  else if (o.op === "tag") { n.tag = o.tag; n.corrected = true; }
  else if (o.op === "delete") {
    work.nodes = work.nodes.filter((x) => x.id !== o.id);
    work.edges = work.edges.filter((e) => e.a !== o.id && e.b !== o.id);
  }
}
function rebuildNet() { net = buildNet({ nodes: work.nodes, edges: work.edges.map((e) => [e.a, e.b]) }); }
function replay() {
  work = pristine();
  for (const step of log) step.ops.forEach(applyOp);
  rebuildNet();
}
const reviewOf = (i) => log.find((s) => s.f === i);

/* A finding is open until a person acts on it, or until the verifier's own rule stops holding after an edit. */
function autoCleared(f) {
  if (f.node < 0) return false;
  const n = net.nodes.get(f.node);
  if (!n) return true;
  const deg = net.adj.get(n.id).size;
  if (f.kind === "isolated") return deg > 0 || sigLinked.has(n.id);
  if (f.kind === "degree") return n.kind !== "valve" || deg === 2;
  if (f.kind === "tag_missing") return n.kind !== "instrument" || !!n.tag;
  return false;
}
const isOpen = (i) => !reviewOf(i) && !autoCleared(D.findings[i]);
const humanOpen = () => D.findings.map((f, i) => i).filter((i) => !INFO.has(D.findings[i].kind) && isOpen(i));

function commit(i, action, ops = []) {
  log.push({ f: i, action, ops });
  ops.forEach(applyOp);
  rebuildNet();
  saveLog();
  stopPicking(true);
  hotFinding = null;
  buildScene();
  update();
  const next = humanOpen()[0];          // move on to the next open finding, like a review queue
  if (next !== undefined) focusFinding(next); else closeReview();
}
function undo() {
  if (!log.length) return;
  const last = log.pop();
  saveLog();
  replay();
  buildScene();
  update();
  if (D.findings[last.f]) focusFinding(last.f);
}
function exportCorrections() {
  const body = { drawing: D.name, reviewed: log.map((s) => ({ finding: s.f, kind: D.findings[s.f]?.kind, action: s.action, ops: s.ops })) };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2)], { type: "application/json" }));
  a.download = `${D.name}.corrections.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ---------- scene ---------- */
function buildScene() {
  svg.replaceChildren();
  nodeEls.clear(); juncEls.clear(); edgeEls.length = 0; findEls.length = 0;
  setView(view);
  el("image", { href: D.image, x: 0, y: 0, width: W, height: H }, svg);
  const gE = el("g", {}, svg), gS = el("g", {}, svg), gH = el("g", {}, svg), gN = el("g", {}, svg), gF = el("g", {}, svg);
  gHover = gH;

  for (const r of [D.legend_box, ...D.excluded].filter(Boolean)) {
    const isLegend = D.legend_box && r === D.legend_box;
    const g = el("g", { class: "region" }, gE);
    el("rect", { x: r[0], y: r[1], width: r[2] - r[0], height: r[3] - r[1], class: "region-box" }, g);
    if (isLegend) {
      const t = el("text", { x: r[0] + 14 * U, y: r[1] + 34 * U, class: "region-t" }, g);
      t.textContent = `legend read by Muse Spark: ${D.legend.length} entries, excluded from tracing`;
    }
  }
  for (const { a, b, path, added } of work.edges) {
    if (!net.nodes.has(a) || !net.nodes.has(b)) continue;
    const pts = path || [center(net.nodes.get(a)), center(net.nodes.get(b))];
    edgeEls.push({ a, b, added, e: el("polyline", { points: pts.map((p) => p.join(",")).join(" "), class: "edge" }, gE) });
  }
  (D.graph.signal_verified || []).forEach(([a, b], k) => {   // links a deterministic pixel check confirmed, drawn along the line it found
    if (!net.nodes.has(a) || !net.nodes.has(b)) return;
    const pts = (D.graph.signal_paths && D.graph.signal_paths[k]) || [center(net.nodes.get(a)), center(net.nodes.get(b))];
    el("polyline", { points: pts.map((p) => p.join(",")).join(" "), class: "sig" }, gS);
  });
  for (const n of work.nodes) {
    if (n.kind === "junction") {
      const [cx, cy] = center(n);
      juncEls.set(n.id, el("circle", { cx, cy, r: 7 * U, class: "junc" }, gE));
      continue;
    }
    const [x1, y1, x2, y2] = n.box, pad = 4 * U;
    const g = el("g", { class: "node " + n.kind, style: `--k: var(--${n.kind in KINDS ? n.kind : "unknown"})` }, gN);
    el("rect", { class: "box", x: x1 - pad, y: y1 - pad, width: x2 - x1 + 2 * pad, height: y2 - y1 + 2 * pad }, g);
    el("path", { class: "x", d: `M${x1} ${y1}L${x2} ${y2}M${x2} ${y1}L${x1} ${y2}` }, g);
    const t = el("text", { class: "badge-t", x: (x1 + x2) / 2, y: y1 - 10 * U }, g);
    nodeEls.set(n.id, { g, t });
    if (n.source === "reinspect") g.classList.add("recovered");
    g.addEventListener("click", (ev) => { if (!dragged) onNode(n, ev.shiftKey); });
    g.addEventListener("mousemove", (ev) => showTip(n, ev));
    g.addEventListener("mouseenter", () => showInferred(n.id));
    g.addEventListener("mouseleave", () => { $("tip").hidden = true; gHover.replaceChildren(); });
  }
  D.findings.forEach((f, i) => {
    if (!isOpen(i)) { findEls.push(null); return; }             // reviewed or cleared: no flag on the drawing
    const n = net.nodes.get(f.node);
    const [x, y] = n && f.node >= 0 && n.kind !== "junction" ? [n.box[2], n.box[1]] : f.pos;
    const info = INFO.has(f.kind);
    const g = el("g", { class: "finding" + (info ? " info" : "") }, gF);
    el("circle", { class: "ring", cx: x, cy: y, r: (info ? 10 : 13) * U }, g);
    el("circle", { class: "dot", cx: x, cy: y, r: (info ? 7 : 10) * U }, g);
    if (!info) { const num = el("text", { class: "num", x, y }, g); num.textContent = i + 1; }
    g.addEventListener("click", () => { if (!dragged) focusFinding(i); });
    g.addEventListener("mousemove", (ev) => showTip(null, ev, f));
    g.addEventListener("mouseleave", () => ($("tip").hidden = true));
    findEls.push(g);
  });
}

function renderLegend() {
  const kinds = Object.entries(KINDS).filter(([k]) => k !== "unknown").map(([k, v]) => `<span><i style="background:var(--${k})"></i>${v}</span>`);
  $("legend").innerHTML = kinds.join("") +
    `<span><i style="background:var(--reach)"></i>reachable</span><span><i style="background:var(--lost)"></i>cut off</span>` +
    `<span><i style="background:var(--base)"></i>not connected</span><span><i style="background:var(--amber)"></i>needs human</span>`;
}

/* ---------- update ---------- */
function update() {
  const iso = isolate(net, closed, source);
  const state = (i) => (iso.now.has(i) ? "reach" : iso.lost.has(i) ? "lost" : "base");

  for (const { a, b, added, e } of edgeEls) e.setAttribute("class", "edge " + (iso.now.has(a) && iso.now.has(b) ? "reach" : iso.lost.has(a) || iso.lost.has(b) ? "lost" : "base") + (added ? " added" : ""));
  for (const [i, c] of juncEls) c.setAttribute("class", "junc " + state(i));
  for (const [i, { g, t }] of nodeEls) {
    const n = net.nodes.get(i), st = state(i), isClosed = iso.shut.has(i);
    g.setAttribute("class", `node ${n.kind} ${st}${isClosed ? " closed" : ""}${i === source ? " source" : ""}${i === hotFinding ? " flash" : ""}${n.source === "reinspect" ? " recovered" : ""}${n.corrected ? " corrected" : ""}${picking && picking.from === i ? " pick" : ""}`);
    t.textContent = i === source ? "SOURCE" : isClosed ? "CLOSED" : "";
    t.setAttribute("class", "badge-t " + (i === source ? "src" : "cl"));
  }
  findEls.forEach((g, i) => g && g.classList.toggle("hot", i === hotFinding));
  renderPanel(iso);
  renderReview();
}

function row(n, extra = "") {
  return `<div class="row" tabindex="0" data-id="${n.id}"><span class="id">#${n.id}</span><span class="tag">${esc(n.tag || "")}</span><span class="nm">${esc(n.name.replace(/^[^,]*,\s*/, ""))}</span>${extra}</div>`;
}

function groups(ids) {
  const by = {};
  for (const i of symbols(net, ids)) (by[net.nodes.get(i).kind] ||= []).push(net.nodes.get(i));
  return Object.keys(KINDS).filter((k) => by[k]).map((k) =>
    `<div class="grp" style="--k:var(--${k})"><i></i>${KINDS[k]}s (${by[k].length})</div>${by[k].map((n) => row(n)).join("")}`).join("");
}

function renderPanel(iso) {
  const s = net.nodes.get(source), reachN = symbols(net, iso.now).length, lostN = symbols(net, iso.lost).length, baseN = symbols(net, iso.baseline).length;
  $("source-card").innerHTML = `<div class="card-h"><h2>Source</h2><span class="sub">${source === autoSource ? "auto-picked: largest network, biggest equipment hub" : "chosen by you"}</span></div>
    <div class="t">${esc(s.name.split(",")[0])} ${esc(s.tag || "")} <span class="m">#${s.id}</span></div>
    <div class="m">${esc(s.name)}</div>` +
    (source === autoSource ? "" : `<button type="button" id="auto">Use suggested source</button>`);
  if ($("auto")) $("auto").onclick = () => { source = autoSource; update(); };

  $("stats").innerHTML = `<div class="stat r"><div class="n">${reachN}</div><div class="l">symbols reachable</div></div>
    <div class="stat l2"><div class="n">${lostN}</div><div class="l">cut off by valves</div></div>
    <div class="stat b"><div class="n">${baseN}</div><div class="l">not connected</div></div>`;

  $("closed").innerHTML = closed.size
    ? [...closed].sort((a, b) => a - b).map((i) => `<span class="chip" data-v="${i}">${esc(label(net.nodes.get(i)))}<span class="x">✕</span></span>`).join("")
    : `<span class="empty">All valves open. Click a valve on the drawing to close it.</span>`;
  $("lost-count").textContent = lostN;
  $("lost").innerHTML = lostN ? groups(iso.lost) : `<div class="empty">Nothing cut off yet.</div>`;
  $("base-count").textContent = baseN;
  $("base").innerHTML = baseN ? groups(iso.baseline) : `<div class="empty">Every symbol is connected to the source.</div>`;

  const crit = rankValves(net, source).filter((r) => r.lost > 0).slice(0, 8);
  $("critical").innerHTML = crit.length ? crit.map((r) => {
    const n = net.nodes.get(r.id);
    return `<div class="row" tabindex="0" data-crit="${r.id}"><span class="id">#${n.id}</span><span class="tag">${esc(n.tag || "valve")}</span><span class="nm">${closed.has(r.id) ? "closed" : "click to close"}</span><span class="r">${r.lost}</span></div>`;
  }).join("") : `<div class="empty">No single valve isolates anything from this source.</div>`;

  renderConfidence();
  document.querySelectorAll("[data-id]").forEach((r) => (r.onclick = () => focusNode(+r.dataset.id)));
  document.querySelectorAll("[data-crit]").forEach((r) => (r.onclick = () => toggleValve(+r.dataset.crit)));
  document.querySelectorAll(".chip").forEach((c) => (c.onclick = () => toggleValve(+c.dataset.v)));
  document.querySelectorAll(".row").forEach((r) => (r.onkeydown = (e) => e.key === "Enter" && r.click()));
}

function renderConfidence() {
  const counts = {};
  const open = D.findings.map((f, i) => i).filter(isOpen);
  open.forEach((i) => (counts[D.findings[i].kind] = (counts[D.findings[i].kind] || 0) + 1));
  const human = open.filter((i) => !INFO.has(D.findings[i].kind)).map((i) => D.findings[i]), openEnds = open.length - human.length;
  const total = open.length, max = Math.max(1, ...Object.values(counts));
  const real = work.nodes.filter((n) => n.kind !== "junction");
  const flagged = new Set(human.filter((f) => f.node >= 0).map((f) => f.node));
  const clean = real.filter((n) => !flagged.has(n.id)).length;
  const mean = real.reduce((a, n) => a + (n.confidence || 0), 0) / Math.max(1, real.length);
  $("clean").textContent = `${clean}/${real.length} symbols clean`;
  $("conf").innerHTML = Object.keys(FINDING).filter((k) => counts[k]).map((k) =>
    `<div class="bar${findFilter === k ? " on" : ""}" data-k="${k}"><span>${FINDING[k]}</span><div class="tr"><div class="fl" style="width:${(counts[k] / max) * 100}%"></div></div><span class="c">${counts[k]}</span></div>`).join("") +
    (total ? "" : `<div class="empty">Verifier found nothing to flag.</div>`) +
    `<div class="kv"><span><b>${real.length}</b> symbols</span><span><b>${work.edges.length}</b> connections</span><span>mean detection conf. <b>${(mean * 100).toFixed(0)}%</b></span></div>`;
  $("conf").querySelectorAll(".bar").forEach((b) => (b.onclick = () => { findFilter = findFilter === b.dataset.k ? null : b.dataset.k; renderConfidence(); }));

  const shown = D.findings.map((f, i) => [f, i]).filter(([f, i]) => isOpen(i) && (findFilter ? f.kind === findFilter : !INFO.has(f.kind)));
  $("find-count").textContent = findFilter ? `${shown.length} of ${total}` : `${human.length} left + ${openEnds} open ends`;
  $("findings").innerHTML = shown.length ? shown.map(([f, i]) =>
    `<div class="f${i === hotFinding ? " on" : ""}" tabindex="0" data-f="${i}"><div class="n">${i + 1}</div><div><div class="k">${FINDING[f.kind] || f.kind}</div><div class="msg">${esc(f.message)}</div></div></div>`).join("")
    : `<div class="empty">${log.length ? "Every flag has been reviewed." : "Nothing needs a human on this drawing"}${openEnds ? "; open pipe ends are in the bars above" : ""}.</div>`;
  const done = D.findings.map((f, i) => [f, i]).filter(([f, i]) => !isOpen(i) && !INFO.has(f.kind));
  $("done-wrap").hidden = !done.length;
  $("done-count").textContent = done.length;
  $("done").innerHTML = done.map(([f, i]) => {
    const r = reviewOf(i);
    return `<div class="done-row"><b>${i + 1}</b> ${FINDING[f.kind] || f.kind} — ${r ? esc(r.action) : "cleared: the verifier rule no longer holds after your edit"}</div>`;
  }).join("");
  $("undo").disabled = $("export").disabled = !log.length;
  renderSummary(human.length, openEnds);
  $("findings").querySelectorAll(".f").forEach((r) => { r.onclick = () => focusFinding(+r.dataset.f); r.onkeydown = (e) => e.key === "Enter" && r.click(); });
}

function renderSummary(human, openEnds) {
  const real = work.nodes.filter((n) => n.kind !== "junction"), rec = real.filter((n) => n.source === "reinspect").length;
  const sig = D.graph.signal_edges || [], ver = D.graph.signal_verified || [];
  $("summary").innerHTML = [
    [real.length, "symbols found"], [work.edges.length, "pipe connections"],
    ...(sig.length ? [[sig.length, `dashed-line links (${ver.length} pixel-verified)`]] : []),
    [rec, "recovered by re-inspection"], [human, "need a human"], [openEnds, "open pipe ends"],
  ].map(([n, l]) => `<span><b>${n}</b> ${l}</span>`).join("");
  $("note-real").hidden = !D.taxonomy;
  $("legend-card").hidden = !D.legend.length;
  $("legend-n").textContent = D.legend.length;
  $("legend-list").innerHTML = D.legend.map((e) => `<div class="lg"><b>${esc(e.item)}</b><span>${esc(e.meaning)}</span></div>`).join("");
}

/* ---------- interaction ---------- */
function toggleValve(i) { closed.has(i) ? closed.delete(i) : closed.add(i); update(); }
function onNode(n, shift) {
  if (picking) {
    if (n.id !== picking.from) commit(picking.f, `connected #${picking.from} to #${n.id}`, [{ op: "link", a: picking.from, b: n.id }]);
    return;
  }
  if (n.kind === "valve" && !shift) toggleValve(n.id);
  else if (n.id !== source) { source = n.id; update(); }
}

function showTip(n, ev, f) {
  const tip = $("tip"), r = wrap.getBoundingClientRect();
  if (n) {
    const iso = isolate(net, closed, source), st = n.id === source ? "source" : iso.now.has(n.id) ? "reachable" : iso.lost.has(n.id) ? "cut off by closed valve" : "not connected to source";
    const fl = D.findings.filter((x, j) => x.node === n.id && isOpen(j)).map((x) => `<div class="w">⚠ ${esc(FINDING[x.kind] || x.kind)}</div>`).join("");
    const sg = (D.graph.signal_edges || []).filter((e) => e.includes(n.id)), sv = (D.graph.signal_verified || []).filter((e) => e.includes(n.id));
    const sgl = sg.length ? `<div class="m">dashed-line links: ${sg.length} (${sv.length} pixel-verified)</div>` : "";
    tip.innerHTML = `<b>${esc(n.tag || n.name.split(",")[0])}</b> <span class="m">#${n.id} · ${n.kind}</span><div class="m">${esc(n.name)}</div>
      <div>confidence ${(n.confidence * 100).toFixed(0)}% · ${esc(n.source || "")} · ${iso.shut.has(n.id) ? "CLOSED" : n.kind === "valve" ? "open" : ""}</div><div>${st}</div>${sgl}${n.source === "reinspect" ? `<div class="g">&#10003; found by re-inspection after the first pass missed it</div>` : ""}${fl}
      <div class="m">${n.kind === "valve" ? "click: open/close · shift+click: set source" : "click: set as source"}</div>`;
  } else tip.innerHTML = `<b>${esc(FINDING[f.kind] || f.kind)}</b><div class="m">${esc(f.message)}</div><div class="m">click to jump</div>`;
  tip.hidden = false;
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  tip.style.left = Math.min(x + 14, r.width - 290) + "px";
  tip.style.top = Math.min(y + 14, r.height - tip.offsetHeight - 6) + "px";
}

function focusNode(id) {
  const n = net.nodes.get(id), [cx, cy] = center(n), w = Math.min(view.w, W / 2);
  hotFinding = null;
  flyTo({ x: cx - w / 2, y: cy - (w * H) / W / 2, w, h: (w * H) / W });
  const g = nodeEls.get(id)?.g;
  if (g) { g.classList.add("flash"); setTimeout(() => g.classList.remove("flash"), 1400); }
}

function focusFinding(i) {
  const f = D.findings[i], n = net.nodes.get(f.node);
  const [x, y] = n && f.node >= 0 && n.kind !== "junction" ? center(n) : f.pos;
  hotFinding = i;
  const w = W / 2.2;
  flyTo({ x: x - w / 2, y: y - (w * H) / W / 2, w, h: (w * H) / W });
  update();
}

/* ---------- review card ---------- */
let reviewKey = "";
function stopPicking(silent) { picking = null; reviewKey = ""; if (!silent) update(); }
function closeReview() { hotFinding = null; reviewKey = ""; update(); }
function showInferred(id) {
  gHover.replaceChildren();
  for (const [a, b] of D.graph.signal_edges || []) {
    if ((a !== id && b !== id) || !net.nodes.has(a) || !net.nodes.has(b)) continue;
    el("polyline", { points: [center(net.nodes.get(a)), center(net.nodes.get(b))].map((p) => p.join(",")).join(" "), class: "sig hover" }, gHover);
  }
}

function renderReview() {
  const card = $("review-card"), i = hotFinding;
  if (i === null || !D.findings[i] || !isOpen(i)) { card.hidden = true; reviewKey = ""; return; }
  const f = D.findings[i], n = f.node >= 0 ? net.nodes.get(f.node) : null, isSym = !!n && n.kind !== "junction";
  const key = [i, picking ? picking.from : "", n && n.kind, n && n.tag, n ? [...net.adj.get(n.id)].join() : ""].join("|");
  card.hidden = false;
  if (key === reviewKey) return;
  reviewKey = key;
  let h = `<div class="card-h"><h2>Review flag ${i + 1}</h2><button type="button" class="link" id="rv-close">close</button></div>
    <div class="t">${esc(FINDING[f.kind] || f.kind)}</div><div class="m">${esc(f.message)}</div>`;
  if (isSym) h += `<div class="m">symbol #${n.id} · ${esc(n.tag || "no tag")} · ${esc(n.name)}</div>`;
  h += `<div class="acts">`;
  if (picking) {
    h += `<div class="pickmsg">Click the symbol on the drawing that #${picking.from} connects to. Esc cancels.</div><button type="button" id="rv-cancel">Cancel</button>`;
  } else {
    h += `<button type="button" class="good" id="rv-ok">${INFO.has(f.kind) ? "Dismiss: the pipe really ends here" : "Looks right, confirm"}</button>`;
    if (isSym) {
      h += `<div class="ln"><span>Type</span><select id="rv-kind">${Object.entries(KINDS).filter(([k]) => k !== "unknown").map(([k, v]) => `<option value="${k}"${k === n.kind ? " selected" : ""}>${v}</option>`).join("")}</select>
        ${D.taxonomy ? "" : `<select id="rv-name"></select>`}<button type="button" id="rv-type">Apply</button></div>
        <div class="ln"><span>Tag</span><input id="rv-tag" value="${esc(n.tag || "")}" placeholder="e.g. TI 255"><button type="button" id="rv-tagb">Apply</button></div>
        <div class="ln"><span>Pipe</span><button type="button" id="rv-connect">Connect to…</button>
          ${[...net.adj.get(n.id)].map((v) => { const m = net.nodes.get(v); return `<button type="button" data-unlink="${v}">Disconnect ${m.kind === "junction" ? "junction" : "#" + v}</button>`; }).join("")}</div>
        <button type="button" class="danger" id="rv-del">Not a symbol: remove it</button>`;
    }
  }
  card.innerHTML = h + `</div>`;
  const on = (id, fn) => { const e = card.querySelector("#" + id); if (e) e.onclick = fn; };
  on("rv-close", closeReview);
  on("rv-cancel", () => stopPicking());
  on("rv-ok", () => commit(i, INFO.has(f.kind) ? "dismissed" : "confirmed as drawn"));
  if (isSym && !picking) {
    const kindSel = card.querySelector("#rv-kind"), nameSel = card.querySelector("#rv-name");
    const fillNames = () => {
      if (!nameSel) return;
      const opts = Object.entries(DATA.legend).filter(([, v]) => v.kind === kindSel.value);
      nameSel.innerHTML = opts.map(([id, v]) => `<option value="${id}"${+id === n.legend_id ? " selected" : ""}>${esc(v.name.replace(/^[^,]*,\s*/, ""))}</option>`).join("") || `<option value="">(no legend entry)</option>`;
    };
    kindSel.onchange = fillNames;
    fillNames();
    on("rv-type", () => {
      const lid = nameSel && nameSel.value !== "" ? +nameSel.value : null;
      const name = lid !== null ? DATA.legend[lid].name : KINDS[kindSel.value];
      commit(i, `retyped as ${kindSel.value}`, [{ op: "kind", id: n.id, kind: kindSel.value, name, legend_id: lid }]);
    });
    on("rv-tagb", () => commit(i, `tag set to "${card.querySelector("#rv-tag").value.trim()}"`, [{ op: "tag", id: n.id, tag: card.querySelector("#rv-tag").value.trim() }]));
    on("rv-connect", () => { picking = { from: n.id, f: i }; reviewKey = ""; update(); });
    on("rv-del", () => commit(i, "removed: not a symbol", [{ op: "delete", id: n.id }]));
    card.querySelectorAll("[data-unlink]").forEach((b) => (b.onclick = () => commit(i, `disconnected #${n.id} from #${b.dataset.unlink}`, [{ op: "unlink", a: n.id, b: +b.dataset.unlink }])));
  }
}

/* ---------- viewBox zoom / pan ---------- */
function clampView(v) {
  const w = Math.max(W / 10, Math.min(W, v.w)), h = (w * H) / W;
  return { w, h, x: Math.max(0, Math.min(W - w, v.x)), y: Math.max(0, Math.min(H - h, v.y)) };
}
function setView(v) { view = clampView(v); svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.h}`); }
let anim = 0;
function flyTo(target) {
  const from = { ...view }, to = clampView(target), t0 = performance.now();
  cancelAnimationFrame(anim);
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 350), e = 1 - Math.pow(1 - k, 3);
    setView(Object.fromEntries(["x", "y", "w"].map((p) => [p, from[p] + (to[p] - from[p]) * e])));
    if (k < 1) anim = requestAnimationFrame(step);
  };
  anim = requestAnimationFrame(step);
}
function toSvg(ev) {
  const r = svg.getBoundingClientRect();
  return [view.x + ((ev.clientX - r.left) / r.width) * view.w, view.y + ((ev.clientY - r.top) / r.height) * view.h];
}
svg.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  cancelAnimationFrame(anim);
  const [px, py] = toSvg(ev), k = ev.deltaY < 0 ? 0.85 : 1 / 0.85, w = Math.max(W / 10, Math.min(W, view.w * k));
  const fx = (px - view.x) / view.w, fy = (py - view.y) / view.h;
  setView({ x: px - fx * w, y: py - fy * ((w * H) / W), w });
}, { passive: false });
svg.addEventListener("mousedown", (ev) => {
  let lastX = ev.clientX, lastY = ev.clientY, moved = 0;
  dragged = false;
  const move = (e) => {
    const r = svg.getBoundingClientRect();
    moved += Math.abs(e.clientX - lastX) + Math.abs(e.clientY - lastY);
    if (moved > 5) { dragged = true; svg.classList.add("panning"); $("tip").hidden = true; }
    if (dragged) setView({ ...view, x: view.x - ((e.clientX - lastX) / r.width) * view.w, y: view.y - ((e.clientY - lastY) / r.height) * view.h });
    lastX = e.clientX; lastY = e.clientY;
  };
  const up = () => {
    window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up);
    svg.classList.remove("panning");
    setTimeout(() => (dragged = false), 0);
  };
  window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
});

init();
