"use strict";

const TOKEN = new URLSearchParams(location.search).get("t") || "";
const VIDEO_EXT = /\.(mkv|mp4|avi|m4v|mov|webm|wmv|ts|mpg|mpeg|flv)$/i;
const SRT_EXT = /\.srt$/i;

const $ = (id) => document.getElementById(id);
const els = {
  deviceBadge: $("deviceBadge"),
  quitBtn: $("quitBtn"),
  gpuBanner: $("gpuBanner"),
  gpuBlurb: $("gpuBlurb"),
  gpuEnableBtn: $("gpuEnableBtn"),
  gpuDismissBtn: $("gpuDismissBtn"),
  gpuLog: $("gpuLog"),
  videoPath: $("videoPath"),
  srtPath: $("srtPath"),
  cueNote: $("cueNote"),
  modelSel: $("modelSel"),
  trackSel: $("trackSel"),
  samplesInput: $("samplesInput"),
  paddingInput: $("paddingInput"),
  runBtn: $("runBtn"),
  readyHint: $("readyHint"),
  progressCard: $("progressCard"),
  progressBar: $("progressBar"),
  progressMsg: $("progressMsg"),
  progressPct: $("progressPct"),
  sampleRows: $("sampleRows"),
  resultCard: $("resultCard"),
  verdict: $("verdict"),
  verdictOffset: $("verdictOffset"),
  verdictText: $("verdictText"),
  lowConfidence: $("lowConfidence"),
  lowConfidenceText: $("lowConfidenceText"),
  statConf: $("statConf"),
  statMad: $("statMad"),
  statTrack: $("statTrack"),
  statDevice: $("statDevice"),
  statElapsed: $("statElapsed"),
  savedPath: $("savedPath"),
  destPath: $("destPath"),
  destBrowseBtn: $("destBrowseBtn"),
  saveBtn: $("saveBtn"),
  saveNote: $("saveNote"),
  techDetails: $("techDetails"),
  errorCard: $("errorCard"),
  errorText: $("errorText"),
  pickerBackdrop: $("pickerBackdrop"),
  pickerTitle: $("pickerTitle"),
  pickerClose: $("pickerClose"),
  pickerUp: $("pickerUp"),
  pickerPathText: $("pickerPathText"),
  pickerDrives: $("pickerDrives"),
  pickerList: $("pickerList"),
  pickerEmpty: $("pickerEmpty"),
  pickerSelectFolderTopBtn: $("pickerSelectFolderTopBtn"),
  pickerDestBar: $("pickerDestBar"),
  pickerDestFileName: $("pickerDestFileName"),
  pickerSelectFolderBtn: $("pickerSelectFolderBtn"),
};

let currentJob = null;
let polling = null;
let gpuDismissed = false;
let pickerMode = null; // "video" | "subtitle" | "dest"
let currentBrowsePath = "";

// ---------------------------------------------------------------- api

async function api(path, options) {
  const opts = Object.assign({ headers: {} }, options || {});
  opts.headers["X-Auth-Token"] = TOKEN;
  if (opts.body) opts.headers["Content-Type"] = "application/json";
  const res = await fetch(path, opts);
  let data = {};
  try { data = await res.json(); } catch (_) { /* empty body */ }
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });

// ---------------------------------------------------------------- utils

function fmtSeconds(s) {
  if (s == null || !isFinite(s)) return "—";
  const abs = Math.abs(s);
  return abs < 60 ? `${abs.toFixed(2)}s` : `${Math.floor(abs / 60)}m ${(abs % 60).toFixed(0)}s`;
}

function fmtDuration(s) {
  if (!s) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : `${m}m`;
}

function fmtSize(bytes) {
  if (!bytes) return "";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, n = bytes;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
}

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

// ---------------------------------------------------------------- boot

async function boot() {
  try {
    const state = await api("/api/state");
    els.deviceBadge.textContent = state.gpu_enabled
      ? "Running on GPU" : "Running on CPU";
    els.deviceBadge.classList.add(state.gpu_enabled ? "gpu" : "cpu");

    if (state.has_nvidia && !state.gpu_enabled && !gpuDismissed) {
      els.gpuBlurb.textContent =
        "An NVIDIA graphics card was found, so this can run much faster. " +
        "Enabling it downloads about 2 GB the first time, and works for every " +
        "future run.";
      show(els.gpuBanner);
    }
    if (state.gpu_status === "done") {
      els.gpuBlurb.textContent = state.gpu_message;
      els.gpuEnableBtn.classList.add("hidden");
    }
  } catch (err) {
    showError(err.message);
  }
  refreshReady();
}

function refreshReady() {
  const v = els.videoPath.value.trim();
  const s = els.srtPath.value.trim();
  const ok = v.length > 0 && s.length > 0;
  els.runBtn.disabled = !ok || !!currentJob;
  if (ok) {
    els.readyHint.textContent = "Ready when you are.";
    els.readyHint.classList.add("ready");
  } else {
    els.readyHint.textContent = "Choose a video and a subtitle file to begin.";
    els.readyHint.classList.remove("ready");
  }
}

// ---------------------------------------------------------------- tracks

let trackSignature = "";

async function loadTracks() {
  const path = els.videoPath.value.trim();
  if (!path || !VIDEO_EXT.test(path)) {
    resetTracks();
    return;
  }
  try {
    const data = await api("/api/tracks?path=" + encodeURIComponent(path));
    const sig = path + "|" + data.tracks.map((t) => t.ordinal).join(",");
    if (sig === trackSignature) return;
    trackSignature = sig;

    els.trackSel.innerHTML = "";
    const auto = document.createElement("option");
    auto.value = "";
    auto.textContent = data.tracks.length > 1
      ? "Choose automatically (recommended)" : "The only audio track";
    els.trackSel.appendChild(auto);

    data.tracks.forEach((t) => {
      const o = document.createElement("option");
      o.value = String(t.ordinal);
      const bits = [];
      if (t.language) bits.push(t.language);
      if (t.codec) bits.push(t.codec);
      if (t.layout) bits.push(t.layout);
      if (t.is_default) bits.push("default");
      o.textContent = `Track ${t.ordinal}${bits.length ? " — " + bits.join(", ") : ""}`;
      els.trackSel.appendChild(o);
    });
    els.cueNote.textContent = data.duration
      ? `Video length: ${fmtDuration(data.duration)}` : "";
  } catch (err) {
    resetTracks();
    els.cueNote.textContent = "";
    if (!/choose a video/i.test(err.message)) showError(err.message);
  }
}

function resetTracks() {
  trackSignature = "";
  els.trackSel.innerHTML =
    '<option value="">Choose automatically</option>';
}

// ---------------------------------------------------------------- run

async function run() {
  hide(els.errorCard);
  hide(els.resultCard);
  hide(els.progressCard);
  els.sampleRows.innerHTML = "";

  const payload = {
    video: els.videoPath.value.trim(),
    subtitle: els.srtPath.value.trim(),
    model: els.modelSel.value,
    track: els.trackSel.value === "" ? null : Number(els.trackSel.value),
    samples: Number(els.samplesInput.value) || 15,
    padding: Number(els.paddingInput.value) || 15,
  };

  els.runBtn.disabled = true;
  els.runBtn.textContent = "Working…";

  try {
    const { job_id } = await post("/api/sync", payload);
    currentJob = job_id;
    show(els.progressCard);
    setProgress(0, "Starting…");
    polling = setInterval(pollProgress, 600);
  } catch (err) {
    els.runBtn.textContent = "Find the desync";
    els.runBtn.disabled = false;
    currentJob = null;
    showError(err.message);
  }
}

async function pollProgress() {
  if (!currentJob) return;
  let data;
  try {
    data = await api("/api/progress?job=" + encodeURIComponent(currentJob));
  } catch (err) {
    clearInterval(polling);
    finish();
    showError(err.message);
    return;
  }

  setProgress(data.fraction, data.message);

  if (data.status === "running") return;

  clearInterval(polling);
  if (data.status === "error") {
    finish();
    showError(data.error || "The check did not finish.");
    return;
  }
  renderResult(data.result);
  finish();
}

function setProgress(fraction, message) {
  const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  els.progressBar.style.width = pct + "%";
  els.progressPct.textContent = pct + "%";
  if (message) els.progressMsg.textContent = message;
}

function finish() {
  els.runBtn.textContent = "Find the desync";
  els.runBtn.disabled = false;
  currentJob = null;
  refreshReady();
}

// ---------------------------------------------------------------- result

function renderResult(r) {
  if (!r) return;
  const offset = r.offset;
  const abs = Math.abs(offset);

  show(els.resultCard);
  hide(els.lowConfidence);

  if (abs < 0.5) {
    els.verdict.classList.remove("warn");
    els.verdictOffset.textContent = "Already in sync";
    els.verdictText.textContent =
      "The subtitles line up with the speech. A corrected copy was saved anyway, but " +
      "you do not need to use it.";
  } else if (offset > 0) {
    els.verdict.classList.remove("warn");
    els.verdictOffset.textContent = fmtSeconds(offset) + " too early";
    els.verdictText.textContent =
      "The subtitles were showing before the dialogue started. In the fixed file they " +
      "have been delayed by " + fmtSeconds(offset) + ".";
  } else {
    els.verdict.classList.remove("warn");
    els.verdictOffset.textContent = fmtSeconds(offset) + " too late";
    els.verdictText.textContent =
      "The subtitles were showing after the dialogue started. In the fixed file they " +
      "have been moved earlier by " + fmtSeconds(offset) + ".";
  }

  if (r.low_confidence) {
    show(els.lowConfidence);
    els.lowConfidence.classList.add("notice-warn");
    els.verdict.classList.add("warn");
    const pct = r.sample_count
      ? Math.round((r.accepted / r.sample_count) * 100) : 0;
    els.lowConfidenceText.textContent =
      `Only ${r.accepted} of ${r.sample_count} samples matched (${pct}%). ` +
      `The offset of ${fmtSeconds(offset)} is a best guess and may be wrong. `;
  }

  els.statConf.textContent = `${r.accepted} / ${r.sample_count} matched`;
  els.statMad.textContent = r.mad == null ? "—" : "±" + r.mad.toFixed(2) + "s";
  els.statTrack.textContent =
    r.track_language ? `${r.track_ordinal} (${r.track_language})` : String(r.track_ordinal);
  els.statDevice.textContent = r.device === "cuda" ? "Graphics card (GPU)" : "Processor (CPU)";
  els.statElapsed.textContent = fmtSeconds(r.elapsed);

  if (r.output_path) {
    els.savedPath.textContent = r.output_path;
    els.destPath.value = r.output_path;
    els.saveNote.textContent = "";
    els.saveNote.classList.remove("ok");
  }

  els.techDetails.textContent =
    `offset      ${r.offset_text}\n` +
    `confidence  ${r.accepted}/${r.sample_count}\n` +
    `mad         ${r.mad}s\n` +
    `track       ${r.track_ordinal} (${r.track_language || "none"})\n` +
    `device      ${r.device}\n` +
    `duration    ${fmtDuration(r.duration)}\n` +
    `elapsed     ${r.elapsed}s\n` +
    `video       ${r.video_path}\n` +
    `subtitle    ${r.srt_path}\n` +
    `saved       ${r.output_path || "(not saved)"}`;

  renderSamples(r.per_sample || []);
}

function renderSamples(rows) {
  if (!rows.length) return;
  const frag = document.createDocumentFragment();
  rows.forEach((s) => {
    const tr = document.createElement("tr");
    tr.className = s.accepted ? "ok" : "no";
    const cells = [
      { t: s.accepted ? "✓" : "·", cls: "" },
      { t: fmtSeconds(s.start), cls: "num" },
      { t: (s.offset > 0 ? "+" : "") + s.offset.toFixed(2) + "s", cls: "num" },
      { t: Math.round((s.similarity || 0) * 100) + "%", cls: "num" },
      { t: s.transcribed || "", cls: "hear" },
    ];
    cells.forEach((c) => {
      const td = document.createElement("td");
      td.textContent = c.t;
      if (c.cls) td.className = c.cls;
      tr.appendChild(td);
    });
    frag.appendChild(tr);
  });
  els.sampleRows.innerHTML = "";
  els.sampleRows.appendChild(frag);
}

function showError(message) {
  els.errorText.textContent = message;
  show(els.errorCard);
  els.errorCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------------------------------------------------------- save

async function saveAs() {
  if (!currentJob && !els.destPath.value) return;
  hide(els.errorCard);
  try {
    const res = await post("/api/save", {
      video: els.videoPath.value.trim(),
      subtitle: els.srtPath.value.trim(),
      dest: els.destPath.value.trim(),
      job: currentJob,
    });
    els.savedPath.textContent = res.path;
    els.saveNote.textContent = "Saved to " + res.path;
    els.saveNote.classList.add("ok");
  } catch (err) {
    showError(err.message);
  }
}

// ---------------------------------------------------------------- picker

function getDefaultSrtFileName() {
  if (els.destPath.value.trim()) {
    const base = els.destPath.value.trim().split(/[\\/]/).pop();
    if (base && base.toLowerCase().endsWith(".srt")) return base;
  }
  if (els.srtPath.value.trim()) {
    const base = els.srtPath.value.trim().split(/[\\/]/).pop();
    if (base) {
      return base.replace(/\.srt$/i, "") + ".corrected.srt";
    }
  }
  if (els.videoPath.value.trim()) {
    const base = els.videoPath.value.trim().split(/[\\/]/).pop();
    if (base) {
      return base.replace(/\.[^.]+$/, "") + ".corrected.srt";
    }
  }
  return "corrected.srt";
}

function chooseFolderAsDest(folderPath) {
  if (!folderPath) return;
  let name = els.pickerDestFileName.value.trim();
  if (!name) name = getDefaultSrtFileName();
  if (!name.toLowerCase().endsWith(".srt")) {
    name += ".srt";
  }
  const sep = (folderPath.includes("/") && !folderPath.includes("\\")) ? "/" : "\\";
  const fullPath = (folderPath.endsWith("\\") || folderPath.endsWith("/"))
    ? (folderPath + name)
    : (folderPath + sep + name);
  els.destPath.value = fullPath;
  closePicker();
}

function openPicker(mode) {
  pickerMode = mode;
  const titles = {
    video: "Choose your video file",
    subtitle: "Choose your subtitle file (.srt)",
    dest: "Choose where to save",
  };
  els.pickerTitle.textContent = titles[mode] || "Choose a file";
  show(els.pickerBackdrop);

  if (mode === "dest") {
    show(els.pickerDestBar);
    show(els.pickerSelectFolderTopBtn);
    els.pickerDestFileName.value = getDefaultSrtFileName();
  } else {
    hide(els.pickerDestBar);
    hide(els.pickerSelectFolderTopBtn);
  }

  api("/api/drives").then((d) => {
    els.pickerDrives.innerHTML = "";
    d.drives.forEach((drv) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = drv;
      b.onclick = () => browse(drv, mode);
      els.pickerDrives.appendChild(b);
    });
  }).catch(() => {});

  let seed = "";
  if (mode === "dest") {
    seed = els.destPath.value.trim() || els.srtPath.value.trim() || els.videoPath.value.trim();
  } else if (mode === "video") {
    seed = els.videoPath.value.trim();
  } else {
    seed = els.srtPath.value.trim() || els.videoPath.value.trim();
  }
  const start = seed && seed.length > 3 ? seed.replace(/[\\/][^\\/]*$/, "") : "";
  browse(start, mode);
}

async function browse(path, mode) {
  try {
    const kind = (mode === "subtitle" || mode === "dest") ? "subtitle" : "video";
    const data = await api("/api/browse?kind=" + kind + "&path=" + encodeURIComponent(path || ""));
    currentBrowsePath = data.path;
    els.pickerPathText.textContent = data.path;
    els.pickerUp.disabled = !data.parent;
    els.pickerUp.onclick = () => browse(data.parent, mode);
    renderPickerList(data.entries, mode);
    els.pickerEmpty.textContent = data.truncated
      ? "Showing the first 2000 items. Open a smaller folder for a full list."
      : (data.entries.length === 0 ? "This folder is empty." : "");
  } catch (err) {
    els.pickerList.innerHTML = "";
    els.pickerEmpty.textContent = err.message;
  }
}

function renderPickerList(entries, mode) {
  els.pickerList.innerHTML = "";
  const frag = document.createDocumentFragment();

  entries.forEach((e) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "picker-item" + (e.is_dir ? " dir" : "");

    const ico = document.createElement("span");
    ico.className = "ico";
    ico.textContent = e.is_dir ? "▸" : "▫";

    const nm = document.createElement("span");
    nm.className = "nm";
    nm.textContent = e.name;

    btn.appendChild(ico);
    btn.appendChild(nm);

    if (e.is_dir) {
      if (mode === "dest") {
        const chooseBtn = document.createElement("span");
        chooseBtn.className = "btn-select-dir";
        chooseBtn.textContent = "Select folder";
        chooseBtn.title = "Save into " + e.name;
        chooseBtn.onclick = (evt) => {
          evt.stopPropagation();
          chooseFolderAsDest(e.path);
        };
        btn.appendChild(chooseBtn);
      }
      btn.onclick = () => browse(e.path, mode);
    } else {
      const sz = document.createElement("span");
      sz.className = "sz";
      sz.textContent = fmtSize(e.size);
      btn.appendChild(sz);

      if (mode === "dest") {
        btn.onclick = () => {
          els.destPath.value = e.path;
          closePicker();
        };
      } else {
        btn.onclick = () => {
          if (mode === "video") els.videoPath.value = e.path;
          else els.srtPath.value = e.path;
          closePicker();
          refreshReady();
          if (mode === "video") loadTracks();
        };
      }
    }
    frag.appendChild(btn);
  });

  els.pickerList.appendChild(frag);
}

function closePicker() {
  hide(els.pickerBackdrop);
  pickerMode = null;
}

// ---------------------------------------------------------------- gpu

async function enableGpu() {
  els.gpuEnableBtn.disabled = true;
  els.gpuEnableBtn.textContent = "Starting…";
  show(els.gpuLog);
  try {
    await post("/api/gpu", {});
    const watch = setInterval(async () => {
      try {
        const g = await api("/api/gpu");
        els.gpuLog.textContent = g.log.join("\n");
        els.gpuLog.scrollTop = els.gpuLog.scrollHeight;
        if (g.status === "running") {
          els.gpuEnableBtn.textContent = "Downloading…";
        } else if (g.status === "done") {
          clearInterval(watch);
          els.gpuEnableBtn.classList.add("hidden");
          els.gpuBlurb.textContent = g.message;
        } else if (g.status === "error") {
          clearInterval(watch);
          els.gpuEnableBtn.disabled = false;
          els.gpuEnableBtn.textContent = "Try again";
          els.gpuBlurb.textContent = g.message;
        }
      } catch (_) { clearInterval(watch); }
    }, 1200);
  } catch (err) {
    els.gpuEnableBtn.disabled = false;
    els.gpuEnableBtn.textContent = "Enable GPU";
    showError(err.message);
  }
}

// ---------------------------------------------------------------- wiring

els.quitBtn.onclick = async () => {
  try { await post("/api/shutdown", {}); } catch (_) {}
  document.body.innerHTML =
    '<div style="display:grid;place-items:center;height:100vh;font:16px sans-serif;' +
    'color:#96a1af;text-align:center;padding:24px">' +
    "SubtitleSync has stopped.<br>You can close this tab." +
    "</div>";
};

els.runBtn.onclick = run;
els.saveBtn.onclick = saveAs;
els.destBrowseBtn.onclick = () => openPicker("dest");
els.gpuEnableBtn.onclick = enableGpu;
els.gpuDismissBtn.onclick = () => { gpuDismissed = true; hide(els.gpuBanner); };
els.pickerClose.onclick = closePicker;
els.pickerBackdrop.onclick = (e) => { if (e.target === els.pickerBackdrop) closePicker(); };
els.pickerSelectFolderBtn.onclick = () => chooseFolderAsDest(currentBrowsePath);
els.pickerSelectFolderTopBtn.onclick = () => chooseFolderAsDest(currentBrowsePath);
els.pickerDestFileName.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    chooseFolderAsDest(currentBrowsePath);
  }
});

document.querySelectorAll(".btn-browse[data-kind]").forEach((btn) => {
  btn.onclick = () => openPicker(btn.dataset.kind);
});

[els.videoPath, els.srtPath].forEach((input) => {
  input.addEventListener("input", () => { refreshReady(); resetTracks(); });
});

let trackTimer = null;
els.videoPath.addEventListener("change", () => {
  clearTimeout(trackTimer);
  trackTimer = setTimeout(loadTracks, 350);
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !els.pickerBackdrop.classList.contains("hidden")) closePicker();
});

boot();
