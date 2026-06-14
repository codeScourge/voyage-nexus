const state = {
  autoScale: true,
  fixedScaleUv: 150,
  channelCount: 32,
  pollTimer: null,
  lastWaveform: null,
  layoutReady: false,
};

const els = {
  status: document.getElementById("status-text"),
  rail: document.getElementById("rail-text"),
  trial: document.getElementById("trial-text"),
  plotArea: document.getElementById("plot-area"),
  stackEeg: document.getElementById("stack-eeg"),
  stackEmg: document.getElementById("stack-emg"),
  toast: document.getElementById("toast"),
  fixedScale: document.getElementById("fixed-scale"),
  scaleProfile: document.getElementById("scale-profile"),
  collectPanel: document.getElementById("collect-panel"),
  collectHint: document.getElementById("collect-hint"),
  wordButtons: document.getElementById("word-buttons"),
  collectPrompt: document.getElementById("collect-prompt"),
  collectPromptWord: document.getElementById("collect-prompt-word"),
  collectPromptMain: document.getElementById("collect-prompt-main"),
  collectPromptSub: document.getElementById("collect-prompt-sub"),
  scrambleBtn: document.getElementById("btn-scramble"),
  scrambleSet: document.getElementById("scramble-set"),
  scrambleRep: document.getElementById("scramble-rep"),
  scrambleSetVal: document.getElementById("scramble-set-val"),
  scrambleRepVal: document.getElementById("scramble-rep-val"),
};

const MARGIN = { left: 44, right: 6, top: 3, bottom: 16 };

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.classList.toggle("error", isError);
  els.toast.classList.remove("hidden");
  window.clearTimeout(showToast._timer);
  showToast._timer = window.setTimeout(() => els.toast.classList.add("hidden"), 5000);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `Request failed (${response.status})`);
  }
  return body;
}

function post(path, payload = {}) {
  return api(path, { method: "POST", body: JSON.stringify(payload) });
}

function yRange(trace) {
  if (state.autoScale) {
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const v of trace.y) {
      ymin = Math.min(ymin, v);
      ymax = Math.max(ymax, v);
    }
    if (!Number.isFinite(ymin)) return [-50, 50];
    const pad = Math.max(5, (ymax - ymin) * 0.08);
    return [ymin - pad, ymax + pad];
  }
  const s = state.fixedScaleUv;
  return [-s, s];
}

function formatUv(v) {
  const abs = Math.abs(v);
  if (abs >= 1000) return `${(v / 1000).toFixed(1)}k`;
  if (abs >= 100) return `${Math.round(v)}`;
  if (abs >= 10) return v.toFixed(0);
  return v.toFixed(1);
}

function formatTime(t) {
  return t.toFixed(1);
}

function setupCanvas(canvas) {
  const wrap = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, wrap.clientWidth);
  const h = Math.max(1, wrap.clientHeight);
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function drawTrace(canvas, trace, timeS, showXAxis) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);

  const bottom = showXAxis ? MARGIN.bottom : 4;
  const plotLeft = MARGIN.left;
  const plotRight = w - MARGIN.right;
  const plotTop = MARGIN.top;
  const plotBottom = h - bottom;
  const plotW = plotRight - plotLeft;
  const plotH = plotBottom - plotTop;
  if (plotW < 8 || plotH < 8) return;

  const [ymin, ymax] = yRange(trace);
  const ySpan = ymax - ymin || 1;
  const n = trace.y.length;
  const t0 = timeS.length ? timeS[0] : -2;
  const t1 = timeS.length ? timeS[timeS.length - 1] : 0;

  const yToPx = (v) => plotBottom - ((v - ymin) / ySpan) * plotH;
  const xToPx = (i) => plotLeft + (i / Math.max(1, n - 1)) * plotW;

  ctx.strokeStyle = varGrid();
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = plotTop + (plotH * tick) / 4;
    ctx.beginPath();
    ctx.moveTo(plotLeft, y);
    ctx.lineTo(plotRight, y);
    ctx.stroke();
  }
  if (showXAxis) {
    for (let tick = 0; tick <= 4; tick += 1) {
      const x = plotLeft + (plotW * tick) / 4;
      ctx.beginPath();
      ctx.moveTo(x, plotTop);
      ctx.lineTo(x, plotBottom);
      ctx.stroke();
    }
  }

  ctx.strokeStyle = "#6b7585";
  ctx.fillStyle = "#9aa3b2";
  ctx.lineWidth = 1;
  ctx.font = "10px system-ui, sans-serif";

  ctx.beginPath();
  ctx.moveTo(plotLeft, plotTop);
  ctx.lineTo(plotLeft, plotBottom);
  ctx.lineTo(plotRight, plotBottom);
  ctx.stroke();

  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const yTicks = [ymax, (ymax + ymin) / 2, ymin];
  const yTickPx = [plotTop, plotTop + plotH / 2, plotBottom];
  for (let i = 0; i < yTicks.length; i += 1) {
    ctx.fillText(formatUv(yTicks[i]), plotLeft - 4, yTickPx[i]);
  }

  if (showXAxis && timeS.length >= 2) {
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let tick = 0; tick <= 4; tick += 1) {
      const t = t0 + ((t1 - t0) * tick) / 4;
      const x = plotLeft + (plotW * tick) / 4;
      ctx.fillText(formatTime(t), x, plotBottom + 3);
    }
  }

  if (n < 2) return;

  ctx.strokeStyle = "#8ab4ff";
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  for (let i = 0; i < n; i += 1) {
    const x = xToPx(i);
    const y = yToPx(trace.y[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function varGrid() {
  return getComputedStyle(document.documentElement).getPropertyValue("--grid").trim() || "#2a3040";
}

function ensureChannelRows(traces) {
  const eeg = traces.filter((t) => t.index < 16);
  const emg = traces.filter((t) => t.index >= 16);
  buildStack(els.stackEeg, eeg);
  buildStack(els.stackEmg, emg);
  state.layoutReady = true;
}

function buildStack(container, traces) {
  const existing = new Map();
  container.querySelectorAll(".plot-row").forEach((row) => {
    existing.set(Number(row.dataset.channel), row);
  });

  const needed = new Set(traces.map((t) => t.index));
  for (const [idx, row] of existing) {
    if (!needed.has(idx)) row.remove();
  }

  traces.forEach((trace, i) => {
    let row = existing.get(trace.index);
    if (!row) {
      row = document.createElement("div");
      row.className = "plot-row";
      row.dataset.channel = String(trace.index);
      const label = document.createElement("label");
      label.textContent = trace.name;
      const wrap = document.createElement("div");
      wrap.className = "plot-canvas-wrap";
      const canvas = document.createElement("canvas");
      canvas.dataset.channel = String(trace.index);
      wrap.appendChild(canvas);
      row.appendChild(label);
      row.appendChild(wrap);
      container.appendChild(row);
    }
    row.querySelector("label").textContent = trace.name;
    row.dataset.showX = i === traces.length - 1 ? "1" : "0";
  });
}

function renderWaveform(waveform) {
  state.lastWaveform = waveform;
  const traces = waveform.traces || [];
  if (!traces.length) return;
  ensureChannelRows(traces);
  const timeS = waveform.time_s || [];
  els.plotArea.querySelectorAll("canvas[data-channel]").forEach((canvas) => {
    const row = canvas.closest(".plot-row");
    const idx = Number(canvas.dataset.channel);
    const trace = traces.find((t) => t.index === idx);
    if (!trace) return;
    const showXAxis = row?.dataset.showX === "1";
    drawTrace(canvas, trace, timeS, showXAxis);
  });
}

function redrawIfReady() {
  if (state.lastWaveform) renderWaveform(state.lastWaveform);
}

function ensureWordButtons(words) {
  if (els.wordButtons.dataset.built === words.join(",")) return;
  els.wordButtons.innerHTML = "";
  for (const word of words) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "word-btn";
    btn.textContent = word;
    btn.dataset.word = word;
    btn.addEventListener("click", async () => {
      try {
        await post("/collect/word", { word });
      } catch (err) {
        showToast(err.message, true);
      }
    });
    els.wordButtons.appendChild(btn);
  }
  els.wordButtons.dataset.built = words.join(",");
}

function syncScrambleSliderLabels(collect = {}) {
  if (els.scrambleSet) {
    els.scrambleSet.min = String(collect.scramble_set_min ?? 1);
    els.scrambleSet.max = String(collect.scramble_set_max ?? 20);
    if (!els.scrambleSet.dataset.touched) {
      els.scrambleSet.value = String(collect.default_scramble_set ?? 5);
    }
    els.scrambleSetVal.textContent = els.scrambleSet.value;
  }
  if (els.scrambleRep) {
    els.scrambleRep.min = String(collect.scramble_rep_min ?? 1);
    els.scrambleRep.max = String(collect.scramble_rep_max ?? 20);
    if (!els.scrambleRep.dataset.touched) {
      els.scrambleRep.value = String(collect.default_scramble_rep ?? 7);
    }
    els.scrambleRepVal.textContent = els.scrambleRep.value;
  }
}

function updateCollectUi(status) {
  const collect = status.collect || { phase: "disabled", words: [] };
  const recording = status.recording_enabled;
  const phase = collect.phase;
  const mode = collect.mode || "single";

  syncScrambleSliderLabels(collect);
  ensureWordButtons(collect.words || []);

  const picking = recording && phase === "pick_word";
  const busy = phase === "countdown" || phase === "say";

  els.collectPanel.classList.toggle("hidden", !recording);
  const beforeS = collect.before_s ?? 1;
  const betweenS = collect.between_s ?? 0.4;
  const sayS = collect.say_s ?? 1.6;
  els.collectHint.textContent = picking
    ? `Choose a word (7 reps each) or run Scramble — ${beforeS}s before each label, ${betweenS}s between reps, ${sayS}s to speak`
    : "Recording active — finish current collection to pick another";

  els.wordButtons.querySelectorAll(".word-btn").forEach((btn) => {
    btn.disabled = !picking;
  });
  if (els.scrambleBtn) els.scrambleBtn.disabled = !picking;
  if (els.scrambleSet) els.scrambleSet.disabled = !picking;
  if (els.scrambleRep) els.scrambleRep.disabled = !picking;

  els.collectPrompt.classList.toggle("hidden", !busy);
  if (busy) {
    const word = collect.word || "";
    const rep = collect.repetition ?? 1;
    const total = collect.repetitions_total ?? 7;
    const setIdx = collect.set_index ?? 1;
    const setTotal = collect.sets_total ?? 1;
    const remaining = collect.phase_remaining_s ?? 0;
    const scramble = mode === "scramble";
    els.collectPromptWord.textContent = word;

    const setPrefix = scramble ? `Set ${setIdx}/${setTotal} · ` : "";
    if (phase === "countdown") {
      els.collectPromptMain.textContent = remaining.toFixed(1);
      els.collectPromptMain.classList.remove("say-it");
      els.collectPromptSub.textContent = scramble && rep === 1
        ? `${setPrefix}new word — get ready`
        : `${setPrefix}Rep ${rep}/${total} — get ready`;
    } else {
      els.collectPromptMain.textContent = "SAY IT";
      els.collectPromptMain.classList.add("say-it");
      els.collectPromptSub.textContent = scramble
        ? `${setPrefix}Rep ${rep}/${total} — speak “${word}” now`
        : `Repetition ${rep}/${total} — speak “${word}” now`;
    }
  }

  let trialText = "Collection: start recording";
  if (!recording) {
    trialText = "Collection: not recording";
  } else if (phase === "pick_word") {
    trialText = "Collection: choose a word or scramble";
  } else if (mode === "scramble") {
    if (phase === "countdown") {
      trialText = `Scramble: “${collect.word}” set ${collect.set_index}/${collect.sets_total} rep ${collect.repetition}/${collect.repetitions_total} — countdown`;
    } else if (phase === "say") {
      trialText = `Scramble: “${collect.word}” set ${collect.set_index}/${collect.sets_total} rep ${collect.repetition}/${collect.repetitions_total} — say it`;
    }
  } else if (phase === "countdown") {
    trialText = `Collection: “${collect.word}” rep ${collect.repetition}/${collect.repetitions_total} — countdown`;
  } else if (phase === "say") {
    trialText = `Collection: “${collect.word}” rep ${collect.repetition}/${collect.repetitions_total} — say it`;
  }
  els.trial.textContent = trialText;
}

function updateControls(status) {
  const recording = status.recording_enabled;
  const collect = status.collect || { phase: "disabled" };
  const canStopRecording = collect.phase === "disabled" || collect.phase === "pick_word";

  document.getElementById("btn-record").textContent = recording
    ? "Stop Session Recording"
    : "Start Session Recording";
  document.getElementById("btn-record").disabled = recording && !canStopRecording;
  document.getElementById("btn-validate").disabled =
    (recording && !canStopRecording) || status.pending_asr_jobs > 0;

  updateCollectUi(status);
}

function updateStatusBar(status) {
  document.querySelector(".top-bar")?.classList.toggle("test-mode", !!status.test_mode);
  const prefix = status.test_mode ? "Test mode | " : "";
  let text = `${prefix}Frames: ${status.total_frames} | Last sample: ${status.latest_sample_index}`;
  if (status.serial_error) text += ` | Serial error: ${status.serial_error}`;
  if (status.recording_enabled && status.session_dir) {
    text += ` | Recording: ${status.session_dir.split("/").pop()}`;
  }
  if (status.pending_asr_jobs > 0) {
    text += ` | ASR jobs: ${status.pending_asr_jobs}`;
  }
  els.status.textContent = text;

  const rail = status.rail_warning || { level: "waiting", text: "Rail warning: waiting for data" };
  els.rail.textContent = rail.text;
  els.rail.className = "rail-text";
  if (rail.level === "ok") els.rail.classList.add("rail-ok");
  else if (rail.level === "warn") els.rail.classList.add("rail-warn");
  else els.rail.classList.add("rail-waiting");
}

async function poll() {
  try {
    const params = new URLSearchParams({ mode: "all", page_start: "0", single_channel: "0" });
    const data = await api(`/api/dashboard?${params}`);
    const status = data.status;
    state.channelCount = status.channel_count || 32;
    if (!state.autoScale && status.default_fixed_scale_uv) {
      state.fixedScaleUv = Number(els.fixedScale.value) || status.default_fixed_scale_uv;
    }
    updateStatusBar(status);
    updateControls(status);
    renderWaveform(data.waveform);
  } catch (err) {
    els.status.textContent = `Poll error: ${err.message}`;
  }
}

function startPolling() {
  const hz = 30;
  if (state.pollTimer) window.clearInterval(state.pollTimer);
  state.pollTimer = window.setInterval(poll, 1000 / hz);
  poll();
}

document.getElementById("btn-scale").addEventListener("click", () => {
  state.autoScale = !state.autoScale;
  document.getElementById("btn-scale").textContent = state.autoScale ? "Scale: Auto" : "Scale: Fixed";
  els.fixedScale.disabled = state.autoScale;
  els.scaleProfile.disabled = state.autoScale;
  redrawIfReady();
});

els.scaleProfile.addEventListener("change", () => {
  const v = els.scaleProfile.value;
  if (v) {
    els.fixedScale.value = v;
    state.fixedScaleUv = Number(v);
    redrawIfReady();
  }
});

els.fixedScale.addEventListener("change", () => {
  state.fixedScaleUv = Number(els.fixedScale.value) || 150;
  redrawIfReady();
});

document.getElementById("btn-record").addEventListener("click", async () => {
  try {
    const recording = document.getElementById("btn-record").textContent.startsWith("Stop");
    if (recording) {
      const body = await post("/recording/stop");
      showToast(`Stopped: ${body.session_dir || "session"}`);
    } else {
      const body = await post("/recording/start", {});
      showToast(`Recording: ${body.session_dir}`);
    }
  } catch (err) {
    showToast(err.message, true);
  }
});

document.getElementById("btn-validate").addEventListener("click", async () => {
  try {
    const body = await post("/session/validate", {});
    const lines = [
      body.ok ? "Validation passed" : "Validation failed",
      ...(body.stats || []).map((s) => `${s.key}: ${s.value}`),
    ];
    if (body.warnings?.length) lines.push("", "Warnings:", ...body.warnings.map((w) => `- ${w}`));
    if (body.errors?.length) lines.push("", "Errors:", ...body.errors.map((e) => `- ${e}`));
    showToast(lines.join("\n"), !body.ok);
  } catch (err) {
    showToast(err.message, true);
  }
});

if (els.scrambleSet) {
  els.scrambleSet.addEventListener("input", () => {
    els.scrambleSet.dataset.touched = "1";
    els.scrambleSetVal.textContent = els.scrambleSet.value;
  });
}

if (els.scrambleRep) {
  els.scrambleRep.addEventListener("input", () => {
    els.scrambleRep.dataset.touched = "1";
    els.scrambleRepVal.textContent = els.scrambleRep.value;
  });
}

if (els.scrambleBtn) {
  els.scrambleBtn.addEventListener("click", async () => {
    try {
      await post("/collect/scramble", {
        set: Number(els.scrambleSet.value),
        rep: Number(els.scrambleRep.value),
      });
    } catch (err) {
      showToast(err.message, true);
    }
  });
}

window.addEventListener("resize", () => {
  window.requestAnimationFrame(redrawIfReady);
});

const resizeObserver = new ResizeObserver(() => {
  window.requestAnimationFrame(redrawIfReady);
});
resizeObserver.observe(els.plotArea);

startPolling();
