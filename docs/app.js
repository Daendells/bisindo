"use strict";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
// SRI hash of ORT_URL + "ort.min.js"; update both together when bumping the version.
const ORT_SRI = "sha384-4uQjzVxVRlLf2zq2sqRNKh3IzetqQzu15fI6xgLQzhXYGJMtTZc/fvdszanED0Zp";
const CONF_MIN = 0.6;    // below this the app says "not sure" (precision on validation: see results.json)
const DETECT_MIN = 0.25; // below this there is no box at all (Ultralytics' default)
const WINDOW = 8;        // camera frames in the vote
const AGREE = 6;         // frames that must agree before a letter counts as steady
const HOLD_MS = 1000;    // steady this long -> typed
const MAX_UPLOAD = 20e6;

const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat("en-US");
const pct = (x, digits = 1) => (x * 100).toFixed(digits) + "%";

let results = null;
let sessionPromise = null;
let session = null;
let mode = null;          // null | "camera" | "image"
let stream = null;
let busy = false;
let votes = [];
let hold = { cls: -1, since: 0, typed: false };
let lastSteady = -1;
let token = 0;            // invalidates stale image predictions

const stage = $("stage"), video = $("video"), view = $("view"), overlay = $("overlay");

/* ---------- results page ---------- */

function get(obj, path) {
  return path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);
}

function bindStats() {
  document.querySelectorAll("[data-stat]").forEach((el) => {
    const [path, kind] = el.dataset.stat.split("|");
    const v = get(results, path);
    if (v == null) return;
    el.textContent = kind === "pct" ? pct(v) : kind === "pct0" ? pct(v, 0) : kind === "int" ? nf.format(v) : String(v);
  });
  if (!results.eval.test) {
    const tile = $("test-tile");
    tile.dataset.pending = "true";
    tile.querySelector(".tile-sub").textContent = "Scored after the final model is chosen.";
  }
}

function bucket(rate) {
  return rate === 1 ? 0 : rate >= 0.9 ? 1 : rate >= 0.7 ? 2 : rate >= 0.4 ? 3 : 4;
}

function describe(letter, s, setName) {
  const wrong = Object.entries(s.confused).map(([p, k]) => (p === "no detection" ? `no box ×${k}` : `${p} ×${k}`));
  return `${letter}, ${setName}: ${s.ok} of ${s.n} correct (${pct(s.ok / s.n, 0)})` +
    (wrong.length ? `. Mistaken for ${wrong.join(", ")}` : "");
}

function renderHeat() {
  const sets = [["test", "Held-out test"], ["external", "Unseen signer"]].filter(([k]) => results.eval[k]);
  const heat = $("heat");
  heat.replaceChildren();
  for (const [key, name] of sets) {
    const ev = results.eval[key];
    const row = document.createElement("div");
    row.className = "heat-row";
    row.innerHTML = `<h4>${name} <span>· ${nf.format(ev.n)} photos</span></h4>`;
    const list = document.createElement("ul");
    list.className = "cells";
    list.setAttribute("aria-label", `${name}, accuracy per letter`);
    for (const L of LETTERS) {
      const s = ev.per_letter[L];
      const li = document.createElement("li");
      li.className = "cell";
      const b = bucket(s.ok / s.n);
      if (b) li.dataset.b = b;
      li.tabIndex = 0;
      li.dataset.tip = describe(L, s, name.toLowerCase());
      li.setAttribute("aria-label", li.dataset.tip);
      li.innerHTML = `<b>${L}</b><small>${Math.round((100 * s.ok) / s.n)}</small>`;
      list.append(li);
    }
    row.append(list);
    heat.append(row);
  }

  const head = `<thead><tr><th>Letter</th>${sets.map(([, n]) => `<th class="num">${n}</th>`).join("")}</tr></thead>`;
  const body = LETTERS.split("").map((L) => `<tr><td>${L}</td>${sets.map(([k]) => {
    const s = results.eval[k].per_letter[L];
    return `<td class="num">${s.ok}/${s.n}</td>`;
  }).join("")}</tr>`).join("");
  $("heat-table").innerHTML = head + `<tbody>${body}</tbody>`;
}

function setupTooltip() {
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  document.body.append(tip);
  const show = (el) => {
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    const r = el.getBoundingClientRect();
    const w = tip.offsetWidth;
    tip.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, r.left + r.width / 2 - w / 2)) + "px";
    tip.style.top = Math.max(8, r.top - tip.offsetHeight - 8) + "px";
  };
  const hide = () => { tip.hidden = true; };
  const heat = $("heat");
  heat.addEventListener("pointerover", (e) => { const c = e.target.closest(".cell"); if (c) show(c); });
  heat.addEventListener("pointerout", hide);
  heat.addEventListener("focusin", (e) => { const c = e.target.closest(".cell"); if (c) show(c); });
  heat.addEventListener("focusout", hide);
  window.addEventListener("scroll", hide, { passive: true });
}

function renderCases() {
  const ext = results.eval.external;
  const wrong = ext.n - ext.correct;
  const oy = ["O", "Y"].reduce((k, L) => k + ext.per_letter[L].n - ext.per_letter[L].ok, 0);
  const rest = LETTERS.split("").filter((L) => L !== "O" && L !== "Y");
  const restOk = rest.reduce((k, L) => k + ext.per_letter[L].ok, 0);
  const restN = rest.reduce((k, L) => k + ext.per_letter[L].n, 0);
  $("case-oy").textContent = (oy === wrong ? `All ${wrong} mistakes on the unseen signer are on O and Y. `
    : `Of the ${wrong} mistakes on the unseen signer, ${oy} are on O and Y. `) +
    `The other 24 letters are ${restOk} of ${restN} correct.`;
  const caption = (L) => {
    const s = ext.per_letter[L];
    const as = Object.entries(s.confused).filter(([p]) => p !== "no detection").map(([p, k]) => `${p} ×${k}`);
    const none = s.confused["no detection"];
    return `${L}: ${s.ok}/${s.n} right` + (as.length ? `; read as ${as.join(", ")}` : "") + (none ? `; no box ×${none}` : "");
  };
  $("cap-o").textContent = caption("O");
  $("cap-y").textContent = caption("Y");

  const val = results.eval.val;
  const mn = val.per_letter.M.confused.N || 0;
  const nm = val.per_letter.N.confused.M || 0;
  $("case-mn").textContent = `On validation, M read as N and N read as M account for ${mn + nm} of the ` +
    `${val.n - val.correct} mistakes.`;

  // The resolution experiment, told with both models' numbers.
  const dep = results.comparison.find((r) => r.deployed);
  const base = results.comparison.find((r) => r.id && !r.deployed && r.model.startsWith("YOLO11n"));
  if (dep && base) {
    const agrees = dep.external_correct >= base.external_correct;
    $("case-exp").textContent =
      `Result, ${dep.imgsz} vs ${base.imgsz} px: ${nf.format(dep.val_correct)} vs ${nf.format(base.val_correct)} of ` +
      `${nf.format(dep.val_n)} validation photos right (M/N mix-ups ${base.val_mn} → ${dep.val_mn}), ` +
      `${agrees ? "and" : "but"} ${dep.external_correct} vs ${base.external_correct} of ${dep.external_n} on the unseen signer. ` +
      `The ${dep.imgsz} px model is in the demo because the rule fixed before the run said validation decides; ` +
      `a gap of a few photos either way is noise, so the two are effectively tied.`;
  }
}

function renderHero() {
  const h = results.hero;
  if (!h) return;
  const [x1, y1, x2, y2] = h.box;
  Object.assign($("hero-box").style, { left: `${x1 * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` });
  $("hero-tag").textContent = `${h.letter} ${Math.round(h.conf * 100)}%`;
  $("hero-img").src = h.file;
  $("hero-fig").hidden = false;
}

function renderCompare() {
  const rows = results.comparison.map((r) => `<tr${r.deployed ? ' class="deployed"' : ""}>
    <td>${r.model}${r.deployed ? '<span class="tag">in the demo</span>' : ""}</td>
    <td>${r.approach}</td>
    <td class="num">${pct(r.val_top1)}</td>
    <td class="num">${pct(r.external_top1)}</td>
    <td class="num">${r.size_mb != null ? r.size_mb + " MB" : "—"}</td>
    <td class="num">${r.cpu_ms != null ? r.cpu_ms + " ms" : "—"}</td></tr>`).join("");
  $("compare").innerHTML = `<thead><tr><th>Model</th><th>Approach</th><th class="num">Validation</th>` +
    `<th class="num">Unseen signer</th><th class="num">Size</th><th class="num">CPU / frame</th></tr></thead><tbody>${rows}</tbody>`;
  const t = results.threshold;
  $("threshold-note").textContent = `At the demo's ${pct(t.conf, 0)} confidence threshold the deployed model answers ` +
    `${pct(t.coverage)} of validation photos and is right on ${pct(t.precision)} of those.`;
}

function renderLetters() {
  const box = $("letters");
  const next = new Map();
  for (const L of LETTERS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = L;
    btn.setAttribute("aria-pressed", "false");
    btn.setAttribute("aria-label", `Sample photo of ${L}`);
    btn.addEventListener("click", () => {
      const list = results.samples.filter((s) => s.letter === L);
      const i = next.get(L) || 0;
      next.set(L, (i + 1) % list.length);
      box.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      showImage(list[i].file, { letter: L, caption: `Sample ${L} · ${list[i].background} (${i + 1}/${list.length})` });
    });
    box.append(btn);
  }
}

/* ---------- model ---------- */

function loadScript(src, integrity) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.integrity = integrity;
    s.crossOrigin = "anonymous";
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error("Could not load ONNX Runtime from the CDN."));
    document.head.append(s);
  });
}

async function fetchModel(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Model download failed (HTTP ${res.status}).`);
  // content-length can be the compressed size, so it only drives the progress text, never the buffer.
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    const text = total ? `Downloading the model… ${Math.min(99, Math.round((100 * got) / total))}%`
      : `Downloading the model… ${(got / 1e6).toFixed(1)} MB`;
    setStatus(text);
    stageMessage(text, total ? Math.min(1, got / total) : null);
  }
  return new Uint8Array(await new Blob(chunks).arrayBuffer());
}

// Loading feedback inside the empty stage: a determinate bar while the model downloads,
// a busy bar while ONNX Runtime starts; text = null restores the idle hint.
const HINT = "Start the camera, upload a photo, or pick a sample letter below.";
function stageMessage(text, fraction) {
  const bar = $("progress");
  $("stage-hint").textContent = text || HINT;
  bar.hidden = text == null;
  bar.classList.toggle("progress-busy", fraction == null);
  if (fraction == null) {
    bar.removeAttribute("aria-valuenow");
  } else {
    const p = Math.round(fraction * 100);
    $("progress-bar").style.width = p + "%";
    bar.setAttribute("aria-valuenow", String(p));
  }
}

function loadModel() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      setStatus("Loading ONNX Runtime…");
      stageMessage("Loading ONNX Runtime…", null);
      if (!window.ort) await loadScript(ORT_URL + "ort.min.js", ORT_SRI);
      ort.env.wasm.wasmPaths = ORT_URL;
      ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin isolated, so threads are unavailable
      const bytes = await fetchModel(results.model.file);
      setStatus("Starting the model…");
      stageMessage("Starting the ONNX Runtime engine (11 MB on first use)…", null);
      session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      return session;
    })();
    sessionPromise.then(() => stageMessage(null), () => { sessionPromise = null; stageMessage(null); });
  }
  return sessionPromise;
}

const prep = document.createElement("canvas");
const prepCtx = prep.getContext("2d", { willReadFrequently: true });

// Letterbox exactly like Ultralytics: scale to fit, centre, pad with grey 114, RGB in [0, 1], CHW.
function toTensor(src, w, h) {
  const S = results.model.imgsz;
  prep.width = prep.height = S;
  prepCtx.fillStyle = "rgb(114,114,114)";
  prepCtx.fillRect(0, 0, S, S);
  const s = Math.min(S / w, S / h);
  const nw = Math.round(w * s), nh = Math.round(h * s);
  const dx = Math.round((S - nw) / 2), dy = Math.round((S - nh) / 2);
  prepCtx.drawImage(src, dx, dy, nw, nh);
  const px = prepCtx.getImageData(0, 0, S, S).data;
  const n = S * S, f = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    f[i] = px[4 * i] / 255;
    f[n + i] = px[4 * i + 1] / 255;
    f[2 * n + i] = px[4 * i + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", f, [1, 3, S, S]), s, dx, dy };
}

// One inference at a time: the session and the preprocessing canvas are shared by camera and photos.
let queue = Promise.resolve();
function infer(src, w, h) {
  const run = queue.then(() => inferNow(src, w, h));
  queue = run.catch(() => {});
  return run;
}

// Output is [1, 4 + 26, anchors]: box (cx, cy, w, h) then class probabilities (already sigmoid).
async function inferNow(src, w, h) {
  const t0 = performance.now();
  const { tensor, s, dx, dy } = toTensor(src, w, h);
  const out = (await session.run({ [session.inputNames[0]]: tensor }))[session.outputNames[0]];
  const [, ch, N] = out.dims;
  const d = out.data, nc = ch - 4;
  const perClass = new Float32Array(nc);
  let best = -1, bestCls = -1, conf = 0;
  for (let c = 0; c < nc; c++) {
    const row = (4 + c) * N;
    for (let i = 0; i < N; i++) {
      const v = d[row + i];
      if (v > perClass[c]) perClass[c] = v;
      if (v > conf) { conf = v; best = i; bestCls = c; }
    }
  }
  let box = null;
  if (conf >= DETECT_MIN) {
    const bw = d[2 * N + best], bh = d[3 * N + best];
    box = { x: (d[best] - bw / 2 - dx) / s, y: (d[N + best] - bh / 2 - dy) / s, w: bw / s, h: bh / s };
  }
  return { cls: conf >= DETECT_MIN ? bestCls : -1, conf, box, perClass, ms: performance.now() - t0 };
}

/* ---------- demo UI ---------- */

function setStatus(text, tone) {
  const el = $("status");
  el.textContent = text;
  if (tone) el.dataset.tone = tone; else delete el.dataset.tone;
}

function setStage(newMode, w, h) {
  mode = newMode;
  if (newMode) stage.dataset.mode = newMode; else delete stage.dataset.mode;
  if (w && h) stage.style.setProperty("--ar", `${w} / ${h}`);
}

function drawBox(r, w, h, mirrored) {
  overlay.width = w;
  overlay.height = h;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  if (!r.box) return;
  const strong = r.conf >= CONF_MIN;
  const x = mirrored ? w - r.box.x - r.box.w : r.box.x;
  const lw = Math.max(2, Math.round(w / 200));
  ctx.lineWidth = lw;
  ctx.strokeStyle = strong ? "#fb923c" : "#9aa8bb";
  ctx.strokeRect(x, r.box.y, r.box.w, r.box.h);
  const label = `${LETTERS[r.cls]} ${Math.round(r.conf * 100)}%`;
  const fs = Math.max(14, Math.round(w / 28));
  ctx.font = `700 ${fs}px "Atkinson Hyperlegible", system-ui, sans-serif`;
  const tw = ctx.measureText(label).width + fs * 0.6;
  const ty = Math.max(0, r.box.y - fs * 1.4);
  ctx.fillStyle = strong ? "#fb923c" : "#9aa8bb";
  ctx.fillRect(x - lw / 2, ty, tw, fs * 1.4);
  ctx.fillStyle = "#0b1220";
  ctx.fillText(label, x + fs * 0.3 - lw / 2, ty + fs * 1.05);
}

function showPrediction(r, steady) {
  const letter = $("letter");
  const shown = steady >= 0 ? steady : r.conf >= CONF_MIN ? r.cls : -1;
  letter.textContent = shown >= 0 ? LETTERS[shown] : "–";
  letter.dataset.state = shown < 0 ? "empty" : steady >= 0 || mode === "image" ? "steady" : "settling";

  $("conf").textContent = r.cls >= 0 ? pct(r.conf, 0) : "—";
  const meter = $("meter");
  meter.style.width = (r.cls >= 0 ? r.conf * 100 : 0) + "%";
  meter.dataset.low = String(r.conf < CONF_MIN);

  const order = [...r.perClass.keys()].sort((a, b) => r.perClass[b] - r.perClass[a]);
  $("alts").innerHTML = order.filter((c) => c !== r.cls).slice(0, 3)
    .filter((c) => r.perClass[c] >= 0.01)
    .map((c) => `<li>${LETTERS[c]} · ${Math.round(r.perClass[c] * 100)}%</li>`).join("");

  if (r.cls < 0) setStatus(mode === "camera" ? "No hand found. Move both hands into the frame." : "No hand found in this photo.");
  else if (r.conf < CONF_MIN) setStatus("Not sure. Try a clearer view of the hands, closer to the camera, in good light.");
  else if (mode === "camera" && steady < 0) setStatus("Hold it steady…");
  else setStatus(`Reads as ${LETTERS[shown]}.`);
  $("timing").textContent = `Inference ${Math.round(r.ms)} ms on this device`;
}

function steadyVote(r) {
  votes.push(r.conf >= CONF_MIN ? r.cls : -1);
  if (votes.length > WINDOW) votes.shift();
  const counts = new Map();
  for (const c of votes) if (c >= 0) counts.set(c, (counts.get(c) || 0) + 1);
  for (const [c, k] of counts) if (k >= AGREE) return c;
  return -1;
}

function autoType(steady, now) {
  const bar = $("hold");
  if (!$("auto").checked || steady < 0) {
    hold = { cls: -1, since: 0, typed: false };
    bar.style.width = "0";
    return;
  }
  if (steady !== hold.cls) hold = { cls: steady, since: now, typed: false };
  const p = Math.min(1, (now - hold.since) / HOLD_MS);
  bar.style.width = (hold.typed ? 0 : p * 100) + "%";
  if (p >= 1 && !hold.typed) {
    hold.typed = true;
    typeKey(LETTERS[steady]);
  }
}

function typeKey(key) {
  const out = $("typed");
  if (key === "back") out.textContent = out.textContent.slice(0, -1);
  else if (key === "clear") out.textContent = "";
  else if (key === "space") out.textContent += " ";
  else {
    out.textContent += key;
    $("live").textContent = `Typed ${key}`;
  }
}

function loop() {
  if (mode !== "camera") return;
  if (!busy && video.readyState >= 2 && video.videoWidth) {
    busy = true;
    const w = video.videoWidth, h = video.videoHeight;
    infer(video, w, h).then((r) => {
      if (mode !== "camera") return;
      const steady = steadyVote(r);
      drawBox(r, w, h, true);
      showPrediction(r, steady);
      autoType(steady, performance.now());
      if (steady !== lastSteady && steady >= 0) $("live").textContent = `Reads as ${LETTERS[steady]}`;
      lastSteady = steady;
    }).catch((e) => {
      stopCamera();
      setStatus(`Inference failed: ${e.message}`, "error");
    }).finally(() => { busy = false; });
  }
  requestAnimationFrame(loop);
}

function cameraError(e) {
  if (!window.isSecureContext) return "The camera needs a secure (https) page.";
  if (e.name === "NotAllowedError") return "Camera access was blocked. Allow it from the address bar, or try a sample letter.";
  if (e.name === "NotFoundError" || e.name === "OverconstrainedError") return "No camera was found. Try a sample letter or upload a photo.";
  if (e.name === "NotReadableError") return "The camera is busy in another app. Close it there and try again.";
  return `The camera could not start (${e.message || e.name}).`;
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("This browser cannot open a camera here. Try a sample letter or upload a photo.", "error");
    return;
  }
  const btn = $("camera-btn");
  btn.disabled = true;
  try {
    const [model, cam] = await Promise.allSettled([
      loadModel(),
      navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false }),
    ]);
    if (cam.status === "rejected") {
      setStatus(cameraError(cam.reason), "error");
      return;
    }
    if (model.status === "rejected") {
      cam.value.getTracks().forEach((t) => t.stop()); // don't leave the camera light on
      throw model.reason;
    }
    stream = cam.value;
    video.srcObject = stream;
    await video.play();
    token++;
    votes = [];
    lastSteady = -1;
    $("caption").textContent = "";
    $("verdict").textContent = "";
    $("letters").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    setStage("camera", video.videoWidth, video.videoHeight);
    setStatus("Camera on. Show a letter.");
    setCameraButton(true);
    requestAnimationFrame(loop);
  } catch (e) {
    setStatus(e.message, "error");
  } finally {
    btn.disabled = false;
  }
}

function stopCamera() {
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  if (mode === "camera") setStage(null);
  setCameraButton(false);
  autoType(-1, 0);
}

function setCameraButton(on) {
  const btn = $("camera-btn");
  btn.querySelector("use").setAttribute("href", on ? "#i-stop" : "#i-camera");
  btn.querySelector("span").textContent = on ? "Stop camera" : "Start camera";
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("That file could not be read as an image."));
    img.src = src;
  });
}

async function showImage(src, sample) {
  const mine = ++token;
  stopCamera();
  $("verdict").textContent = "";
  setStatus("Loading…");
  try {
    const [img] = await Promise.all([loadImage(src), loadModel()]);
    if (mine !== token) return;
    const w = img.naturalWidth, h = img.naturalHeight;
    view.width = w;
    view.height = h;
    view.getContext("2d").drawImage(img, 0, 0);
    setStage("image", w, h);
    $("caption").textContent = sample ? sample.caption : "Your photo";
    const r = await infer(img, w, h);
    if (mine !== token) return;
    drawBox(r, w, h, false);
    showPrediction(r, -1);
    if (sample) {
      const ok = r.conf >= CONF_MIN && LETTERS[r.cls] === sample.letter;
      const v = $("verdict");
      v.dataset.ok = String(ok);
      v.innerHTML = `<svg class="icon" aria-hidden="true"><use href="#i-${ok ? "check" : "x"}"/></svg>` +
        (ok ? `Correct, it is ${sample.letter}` : `The answer is ${sample.letter}`);
    }
    $("live").textContent = r.cls >= 0 ? `Reads as ${LETTERS[r.cls]}, ${Math.round(r.conf * 100)} percent` : "No hand found";
  } catch (e) {
    if (mine === token) setStatus(e.message, "error");
  }
}

/* ---------- start ---------- */

async function init() {
  try {
    const res = await fetch("results.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    results = await res.json();
  } catch (e) {
    setStatus(`Could not load results.json (${e.message}).`, "error");
    return;
  }
  bindStats();
  renderHero();
  renderHeat();
  setupTooltip();
  renderCases();
  renderCompare();
  renderLetters();

  $("camera-btn").addEventListener("click", () => (stream ? stopCamera() : startCamera()));
  $("upload-btn").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    if (!file.type.startsWith("image/")) return setStatus("Please choose an image file.", "error");
    if (file.size > MAX_UPLOAD) return setStatus("That image is over 20 MB. Please choose a smaller one.", "error");
    $("letters").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    const url = URL.createObjectURL(file);
    showImage(url, null).finally(() => URL.revokeObjectURL(url));
  });
  document.querySelectorAll("[data-key]").forEach((b) => b.addEventListener("click", () => typeKey(b.dataset.key)));
}

init();
