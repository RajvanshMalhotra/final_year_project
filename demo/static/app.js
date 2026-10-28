// Live view of the demo: one WebSocket (/ws/ui) pushes a snapshot 4x per second; controls POST to /api/*.
"use strict";

const NODES = ["mac-m2", "win-gpu-0", "win-gpu-1"];
const WINDOW_S = 240;               // ribbon and charts show the last 4 minutes
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COLOR = {};
["healthy", "slow", "waiting", "dead", "removed", "agent", "ink", "muted", "line", "panel"].forEach(
  (k) => (COLOR[k] = css("--" + k)));
COLOR.joining = COLOR.waiting; COLOR.idle = COLOR.removed; COLOR.starting = COLOR.line; COLOR.training = COLOR.healthy;

const STATUS_WORDS = {
  training: "Training", slow: "Training, but slow", waiting: "Waiting for the group", joining: "Joining the group",
  dead: "Not responding", removed: "Removed from the group", idle: "Idle", starting: "Starting",
};

let timeline = [];                  // [{t, s: {node: status}, v: {node: speed ratio}}], kept by the controller
let snap = null;
let lastToast = 0;

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);
function effective(n) {
  if (n.status === "training" && n.speed_ratio != null && n.speed_ratio < 0.85) return "slow";
  return n.status;
}
function fmtTime(t) {
  return new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
function pct(x) { return x == null ? "–" : Math.round(x * 100) + "%"; }
function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  lastToast = Date.now();
  setTimeout(() => { if (Date.now() - lastToast >= 2400) el.classList.remove("show"); }, 2500);
}
async function post(path, body) {
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) { toast(j.detail || "The controller refused that"); return null; }
    return j;
  } catch (e) {
    toast("The controller is not reachable");
    return null;
  }
}
function setupCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  }
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  g.font = "12px Plex, sans-serif";
  return { g, w, h };
}

// ------------------------------------------------------------------ machines
function buildMachines() {
  const tpl = $("machine-tpl");
  NODES.forEach((id) => {
    const el = tpl.content.firstElementChild.cloneNode(true);
    el.id = "m-" + id;
    el.querySelector(".m-name").textContent = id;
    el.querySelector(".m-trace").setAttribute("aria-label", `Speed of ${id} compared with its normal speed`);
    el.querySelectorAll("button[data-act]").forEach((b) => {
      b.addEventListener("click", async () => {
        const act = b.dataset.act;
        const body = { node: id, kind: act };
        if (act === "slow") Object.assign(body, { factor: 0.4, duration: 30 });
        if (act === "net") Object.assign(body, { factor: 0.2, duration: 30 });
        const ok = await post("/api/fault", body);
        if (ok) toast({ crash: `Crashed ${id}`, slow: `${id} slowed to 40% for 30 s`,
                        net: `${id} link cut to 20% for 30 s`, heal: `Healed ${id}` }[act]);
      });
    });
    $("machines").appendChild(el);
  });
}

function renderMachines(s) {
  s.nodes.forEach((n) => {
    const el = $("m-" + n.id);
    const st = effective(n);
    el.dataset.status = st;
    el.querySelector(".m-hw").textContent =
      n.label.replace(" · ", ", ") + (n.speed_factor < 1 ? `; runs at ${n.speed_factor.toFixed(2)}x an NVIDIA laptop` : "");
    let word = STATUS_WORDS[st] || st;
    const reboot = s.reboot_in && s.reboot_in[n.id];
    if (reboot) word += `, restarts in ${Math.max(0, Math.round(reboot))} s`;
    el.querySelector(".m-status").textContent = word;
    el.querySelector(".m-speed").textContent = st === "dead" ? "–" : pct(n.speed_ratio);
    el.querySelector(".m-share").textContent = pct(n.share);
    el.querySelector(".m-count").textContent = `${n.count} (fits ${n.cap})`;
    el.querySelector(".m-hb").textContent = n.hb_age == null ? "–" : `${n.hb_age.toFixed(1)} s ago`;
    const f = [];
    if (n.throttle < 0.999) f.push(`Slowed to ${pct(n.throttle)} of normal`);
    if (n.bw_factor < 0.999) f.push(`Link at ${pct(n.bw_factor)} of normal`);
    if (n.compensated) f.push("Running in lighter mode");
    el.querySelector(".m-faults").textContent = f.join(". ");
    drawTrace(el.querySelector(".m-trace"), timeline, n.id, s.t, COLOR[st] || COLOR.muted);
  });
}

function drawTrace(c, tl, id, now, color) {
  const { g, w, h } = setupCanvas(c);
  const y = (v) => h - 4 - Math.min(v, 1.15) / 1.15 * (h - 8);
  const t0 = now - WINDOW_S;
  g.strokeStyle = COLOR.line; g.setLineDash([4, 4]); g.beginPath();
  g.moveTo(0, y(0.85)); g.lineTo(w, y(0.85)); g.stroke(); g.setLineDash([]);
  g.fillStyle = COLOR.muted; g.fillText("85% of normal", 4, y(0.85) + 14);
  g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
  let pen = false;
  tl.forEach((p) => {
    const v = p.v[id], x = (p.t - t0) / WINDOW_S * w;
    if (v == null || x < 0) { pen = false; return; }
    pen ? g.lineTo(x, y(v)) : g.moveTo(x, y(v));
    pen = true;
  });
  g.stroke();
}

// ------------------------------------------------------------------ ribbon
function drawRibbon(s) {
  const { g, w, h } = setupCanvas($("ribbon"));
  const left = w < 700 ? 78 : 92, right = w - 8, now = s.t, t0 = now - WINDOW_S;
  const X = (t) => left + (t - t0) / WINDOW_S * (right - left);
  const laneH = 34, top = 34;
  const laneY = (i) => top + i * (laneH + 10);
  // time axis
  g.fillStyle = COLOR.muted; g.strokeStyle = COLOR.line; g.textAlign = "center";
  const every = w < 700 ? 60 : 30;
  for (let k = 0; k <= WINDOW_S; k += every) {
    const x = X(t0 + k);
    g.beginPath(); g.moveTo(x, top - 6); g.lineTo(x, laneY(2) + laneH); g.stroke();
    const ago = WINDOW_S - k;
    g.fillText(ago === 0 ? "now" : `${Math.floor(ago / 60)}:${String(ago % 60).padStart(2, "0")} ago`, x, top - 12);
  }
  // lanes coloured by status over time
  g.textAlign = "left";
  NODES.forEach((id, i) => {
    g.fillStyle = COLOR.ink; g.font = "bold 13px Plex, sans-serif";
    g.fillText(id, 0, laneY(i) + laneH / 2 + 4);
    g.font = "12px Plex, sans-serif";
    for (let k = 0; k < timeline.length; k++) {
      const a = timeline[k], b = timeline[k + 1];
      const ta = Math.max(a.t, t0), tb = b ? b.t : now;
      if (tb < t0) continue;
      g.fillStyle = COLOR[a.s[id]] || COLOR.line;
      g.fillRect(X(ta), laneY(i), Math.max(1, X(tb) - X(ta) + 0.5), laneH);
    }
  });
  // markers: faults (red triangles) and agent decisions (indigo diamonds) on the lane they hit
  (s.events || []).forEach((e) => {
    if (e.t < t0) return;
    const i = NODES.indexOf(e.node);
    const x = X(e.t);
    if (e.kind === "fault" && i >= 0) {
      const y = laneY(i) - 1;
      g.fillStyle = COLOR.dead; g.beginPath(); g.moveTo(x - 7, y - 9); g.lineTo(x + 7, y - 9); g.lineTo(x, y + 3); g.fill();
    } else if (e.kind === "decision") {
      const y = i >= 0 ? laneY(i) + laneH / 2 : laneY(0) - 4;
      g.fillStyle = COLOR.agent; g.strokeStyle = COLOR.panel; g.lineWidth = 2;
      g.beginPath(); g.moveTo(x, y - 8); g.lineTo(x + 8, y); g.lineTo(x, y + 8); g.lineTo(x - 8, y); g.closePath();
      g.fill(); g.stroke();
    }
  });
}

// ------------------------------------------------------------------ charts
function drawLine(canvas, pts, color, fmt) {
  const { g, w, h } = setupCanvas(canvas);
  const left = 44, right = w - 64, bottom = h - 18, topY = 8;
  if (pts.length < 2) { g.fillStyle = COLOR.muted; g.fillText("Waiting for training steps…", left, h / 2); return; }
  const now = snap.t, t0 = now - WINDOW_S;
  const vals = pts.map((p) => p.v);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-6) { hi += 1; lo = Math.max(0, lo - 1); }
  const pad = (hi - lo) * 0.1; lo = Math.max(0, lo - pad); hi += pad;
  const X = (t) => left + (t - t0) / WINDOW_S * (right - left);
  const Y = (v) => bottom - (v - lo) / (hi - lo) * (bottom - topY);
  g.strokeStyle = COLOR.line; g.fillStyle = COLOR.muted; g.textAlign = "right";
  for (let k = 0; k <= 3; k++) {
    const v = lo + (hi - lo) * k / 3, y = Y(v);
    g.beginPath(); g.moveTo(left, y); g.lineTo(right, y); g.stroke();
    g.fillText(fmt(v), left - 6, y + 4);
  }
  g.strokeStyle = color; g.lineWidth = 2; g.beginPath();
  let prev = null;
  pts.forEach((p) => {
    if (p.t < t0) return;
    const x = X(p.t), y = Y(p.v);
    // a gap of more than 3 s means training was stalled: break the line there
    if (prev && p.t - prev.t > 3) { g.stroke(); g.beginPath(); g.moveTo(x, y); } else if (prev) g.lineTo(x, y); else g.moveTo(x, y);
    prev = p;
  });
  g.stroke();
  const last = pts[pts.length - 1];
  g.fillStyle = COLOR.ink; g.textAlign = "left"; g.font = "bold 13px Plex, sans-serif";
  g.fillText(fmt(last.v), Math.min(X(last.t), right) + 6, Y(last.v) + 4);
}

function smooth(points, k) {
  return points.map((p, i) => {
    const a = points.slice(Math.max(0, i - k + 1), i + 1);
    return { t: p.t, v: a.reduce((s, q) => s + q.v, 0) / a.length };
  });
}

// ------------------------------------------------------------------ feeds
function renderFeeds(s) {
  const dl = $("decisions");
  if (s.decisions && s.decisions.length) {
    dl.innerHTML = "";
    s.decisions.slice().reverse().slice(0, 30).forEach((d) => {
      const li = document.createElement("li");
      const what = document.createElement("div");
      what.className = "what";
      what.textContent = `${d.text.charAt(0).toUpperCase() + d.text.slice(1)} (${d.mode === "rl" ? "RL agent" : "simple rule"})`;
      const why = document.createElement("div");
      why.className = "why";
      const seen = d.target && d.seen[d.target];
      why.textContent = `${fmtTime(d.t)}, step ${d.step}` + (seen
        ? `. It saw ${d.target} at ${seen.speed == null ? "no" : Math.round(seen.speed * 100) + "%"} speed, last heartbeat ${seen.hb_age} s ago`
        : "");
      li.append(what, why);
      dl.appendChild(li);
    });
  }
  const el = $("events");
  el.innerHTML = "";
  (s.events || []).slice().reverse().slice(0, 60).forEach((e) => {
    const li = document.createElement("li");
    li.dataset.kind = e.kind;
    const t = document.createElement("time"); t.textContent = fmtTime(e.t);
    const sp = document.createElement("span"); sp.textContent = e.text;
    li.append(t, sp);
    el.appendChild(li);
  });
}

// ------------------------------------------------------------------ main render
function render(s) {
  snap = s;
  const hist = s.history || [];
  const last = hist[hist.length - 1];
  $("r-step").textContent = s.step;
  $("r-group").textContent = s.members.length ? `${s.members.length} machines, ${s.group}` : s.group;
  $("r-loss").textContent = last && last.loss != null ? last.loss.toFixed(3) : "–";
  $("r-rate").textContent = last ? last.samples_s.toFixed(1) : "–";
  $("r-B").textContent = s.B;
  $("load-auto").checked = s.auto_load;
  $("chaos-on").checked = s.chaos;
  $("chaos-level").value = s.level;
  $("chaos-left").textContent = s.chaos ? `${s.chaos_left} scheduled faults left (seed ${s.scenario_seed})` : "";
  document.querySelectorAll("[data-agent]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.agent === s.agent)));
  timeline = s.timeline || [];
  renderMachines(s);
  drawRibbon(s);
  const loss = hist.filter((p) => p.loss != null).map((p) => ({ t: p.t, v: p.loss }));
  drawLine($("chart-loss"), smooth(loss, 8), COLOR.ink, (v) => v.toFixed(2));
  drawLine($("chart-rate"), smooth(hist.map((p) => ({ t: p.t, v: p.samples_s })), 8), COLOR.healthy, (v) => v.toFixed(0));
  renderFeeds(s);
  if (!$("citations").childElementCount && s.citations) {
    Object.entries(s.citations).forEach(([k, v]) => {
      const li = document.createElement("li");
      li.textContent = `${k}: ${v}`;
      $("citations").appendChild(li);
    });
    $("logfile").textContent = `Every event and decision is saved to ${s.log_file}`;
  }
}

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/ui`);
  ws.onopen = () => ($("connection").textContent = "Live: one Mac running three training programs and the agent");
  ws.onmessage = (m) => render(JSON.parse(m.data));
  ws.onclose = () => {
    $("connection").textContent = "Lost the controller. Retrying…";
    setTimeout(connect, 1500);
  };
}

// ------------------------------------------------------------------ controls
document.querySelectorAll("[data-agent]").forEach((b) =>
  b.addEventListener("click", async () => {
    if (await post("/api/agent", { mode: b.dataset.agent })) toast(`Faults are now handled by: ${b.textContent}`);
  }));
$("load-up").addEventListener("click", () => post("/api/load", { delta: 4 }));
$("load-down").addEventListener("click", () => post("/api/load", { delta: -4 }));
$("load-auto").addEventListener("change", (e) => post("/api/load", { auto: e.target.checked }));
$("chaos-on").addEventListener("change", (e) => post("/api/chaos", { on: e.target.checked, level: $("chaos-level").value }));
$("chaos-level").addEventListener("change", (e) => post("/api/chaos", { on: $("chaos-on").checked, level: e.target.value }));
window.addEventListener("resize", () => snap && render(snap));   // redraw canvases at the new size

buildMachines();
connect();
