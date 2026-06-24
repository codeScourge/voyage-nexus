const DEFAULT_AMPLITUDE_UV = 150;
const BAND_EEG_SCALE_RATIO = 200_000 / DEFAULT_AMPLITUDE_UV;
const BAND_EMG_SCALE_RATIO = 150 / DEFAULT_AMPLITUDE_UV;

const state = {
  autoScale: true,
  amplitudeUv: DEFAULT_AMPLITUDE_UV,
  channelCount: 32,
  pollTimer: null,
  pollInFlight: false,
  pollAbort: null,
  wordSwitchFlashKey: null,
  lastWaveform: null,
  layoutReady: false,
  alignmentResult: null,
  statusPollTimer: null,
  alignmentResultFetchKey: null,
  modelTestResult: null,
  modelTestResultFetchKey: null,
  visibleEegBands: new Set(),
  visibleEmgBands: new Set(),
};

const els = {
  status: document.getElementById("status-text"),
  rail: document.getElementById("rail-text"),
  trial: document.getElementById("trial-text"),
  plotArea: document.getElementById("plot-area"),
  stackEeg: document.getElementById("stack-eeg"),
  stackEmg: document.getElementById("stack-emg"),
  legendEeg: document.getElementById("legend-eeg"),
  legendEmg: document.getElementById("legend-emg"),
  toast: document.getElementById("toast"),
  amplitudeSlider: document.getElementById("amplitude-slider"),
  amplitudeVal: document.getElementById("amplitude-val"),
  collectPanel: document.getElementById("collect-panel"),
  collectHint: document.getElementById("collect-hint"),
  wordButtons: document.getElementById("word-buttons"),
  wordWeightSliders: document.getElementById("word-weight-sliders"),
  negativeLabelMixSliders: document.getElementById("negative-label-mix-sliders"),
  collectPrompt: document.getElementById("collect-prompt"),
  collectPromptWord: document.getElementById("collect-prompt-word"),
  collectPromptMain: document.getElementById("collect-prompt-main"),
  collectPromptSub: document.getElementById("collect-prompt-sub"),
  scrambleFastBtn: document.getElementById("btn-scramble-fast"),
  scrambleBreaksBtn: document.getElementById("btn-scramble-breaks"),
  scrambleSet: document.getElementById("scramble-set"),
  scrambleRep: document.getElementById("scramble-rep"),
  scrambleSetVal: document.getElementById("scramble-set-val"),
  scrambleRepVal: document.getElementById("scramble-rep-val"),
  negativeLabelsBtn: document.getElementById("btn-negative-labels"),
  wordSwitchFlash: document.getElementById("word-switch-flash"),
  alignmentBtn: document.getElementById("btn-alignment-test"),
  alignmentDismiss: document.getElementById("btn-alignment-dismiss"),
  alignmentPrompt: document.getElementById("alignment-prompt"),
  alignmentPromptMain: document.getElementById("alignment-prompt-main"),
  alignmentPromptSub: document.getElementById("alignment-prompt-sub"),
  alignmentResult: document.getElementById("alignment-result"),
  alignmentResultMeta: document.getElementById("alignment-result-meta"),
  alignmentResultPlots: document.getElementById("alignment-result-plots"),
  modelTestBtn: document.getElementById("btn-model-test"),
  modelTestDismiss: document.getElementById("btn-model-test-dismiss"),
  modelTestPrompt: document.getElementById("model-test-prompt"),
  modelTestPromptMain: document.getElementById("model-test-prompt-main"),
  modelTestPromptSub: document.getElementById("model-test-prompt-sub"),
  modelTestResult: document.getElementById("model-test-result"),
  modelTestResultMeta: document.getElementById("model-test-result-meta"),
  modelTestResultRows: document.getElementById("model-test-result-rows"),
  modelUseBtn: document.getElementById("btn-model-use"),
  modelUsePanel: document.getElementById("model-use-panel"),
  modelUseMeta: document.getElementById("model-use-meta"),
  modelUsePrediction: document.getElementById("model-use-prediction"),
  modelUseBars: document.getElementById("model-use-bars"),
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

function fixedBandMax(channelIdx) {
  const ratio = channelIdx < 16 ? BAND_EEG_SCALE_RATIO : BAND_EMG_SCALE_RATIO;
  return state.amplitudeUv * ratio;
}

function formatAmplitudeLabel(uv) {
  return `±${Math.round(uv)} µV`;
}

function syncAmplitudeUi() {
  els.amplitudeSlider.value = String(state.amplitudeUv);
  els.amplitudeVal.textContent = formatAmplitudeLabel(state.amplitudeUv);
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
  const s = state.amplitudeUv;
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

function drawAlignmentTrace(canvas, trace, timeS, { labelT0, labelT1, showXAxis = false }) {
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
  const t0 = timeS.length ? timeS[0] : 0;
  const t1 = timeS.length ? timeS[timeS.length - 1] : 1;
  const tSpan = t1 - t0 || 1;

  const yToPx = (v) => plotBottom - ((v - ymin) / ySpan) * plotH;
  const tToPx = (t) => plotLeft + ((t - t0) / tSpan) * plotW;
  const xToPx = (i) => plotLeft + (i / Math.max(1, n - 1)) * plotW;

  if (Number.isFinite(labelT0) && Number.isFinite(labelT1) && labelT1 > labelT0) {
    const x0 = tToPx(Math.max(t0, labelT0));
    const x1 = tToPx(Math.min(t1, labelT1));
    ctx.fillStyle = "rgba(126, 231, 135, 0.12)";
    ctx.fillRect(x0, plotTop, Math.max(1, x1 - x0), plotH);
    ctx.strokeStyle = "rgba(126, 231, 135, 0.65)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    for (const x of [x0, x1]) {
      ctx.beginPath();
      ctx.moveTo(x, plotTop);
      ctx.lineTo(x, plotBottom);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

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

function bandId(band) {
  return band.id || band.name;
}

function visibleBandSet(channelIdx) {
  return channelIdx < 16 ? state.visibleEegBands : state.visibleEmgBands;
}

function ensureBandVisibility(bands, channelIdx) {
  const visible = visibleBandSet(channelIdx);
  for (const band of bands || []) {
    const id = bandId(band);
    if (!visible.has(id)) visible.add(id);
  }
}

function filterVisibleBands(bands, channelIdx) {
  const visible = visibleBandSet(channelIdx);
  return (bands || []).filter((band) => visible.has(bandId(band)));
}

function bandYRange(bands, channelIdx) {
  if (!state.autoScale) {
    return [0, fixedBandMax(channelIdx)];
  }
  let ymin = Infinity;
  let ymax = -Infinity;
  for (const band of bands) {
    for (const v of band.y) {
      ymin = Math.min(ymin, v);
      ymax = Math.max(ymax, v);
    }
  }
  if (!Number.isFinite(ymin)) return [0, 1];
  if (ymax <= ymin) return [0, Math.max(1, ymax)];
  const pad = (ymax - ymin) * 0.08;
  return [Math.max(0, ymin - pad), ymax + pad];
}

function drawBandPlot(canvas, bands, timeS, showXAxis, channelIdx) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  if (!bands?.length) return;

  const bottom = showXAxis ? MARGIN.bottom : 4;
  const plotLeft = MARGIN.left;
  const plotRight = w - MARGIN.right;
  const plotTop = MARGIN.top;
  const plotBottom = h - bottom;
  const plotW = plotRight - plotLeft;
  const plotH = plotBottom - plotTop;
  if (plotW < 8 || plotH < 8) return;

  const [ymin, ymax] = bandYRange(bands, channelIdx);
  const ySpan = ymax - ymin || 1;
  const n = bands[0]?.y?.length || 0;
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
    ctx.fillText(formatBandPower(yTicks[i]), plotLeft - 4, yTickPx[i]);
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

  for (const band of bands) {
    ctx.strokeStyle = band.color || "#8ab4ff";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (let i = 0; i < n; i += 1) {
      const x = xToPx(i);
      const y = yToPx(band.y[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

function formatBandPower(v) {
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  if (v >= 100) return `${Math.round(v)}`;
  if (v >= 10) return v.toFixed(0);
  if (v >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

function renderBandLegend(container, bands, channelIdx) {
  if (!container || !bands?.length) return;
  ensureBandVisibility(bands, channelIdx);
  const key = bands.map((b) => `${bandId(b)}:${b.range || ""}:${b.color}`).join("|");
  if (container.dataset.built === key) return;
  container.innerHTML = "";
  for (const band of bands) {
    const id = bandId(band);
    const item = document.createElement("label");
    item.className = "legend-item";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.className = "legend-toggle";
    checkbox.checked = visibleBandSet(channelIdx).has(id);
    checkbox.addEventListener("change", () => {
      const visible = visibleBandSet(channelIdx);
      if (checkbox.checked) visible.add(id);
      else visible.delete(id);
      redrawIfReady();
    });
    item.appendChild(checkbox);
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.background = band.color;
    item.appendChild(swatch);
    const label = band.range ? `${band.name} ${band.range}` : band.name;
    item.appendChild(document.createTextNode(label));
    container.appendChild(item);
  }
  container.dataset.built = key;
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

function makePlotCanvas(channelIdx, role) {
  const wrap = document.createElement("div");
  wrap.className = "plot-canvas-wrap";
  const canvas = document.createElement("canvas");
  canvas.dataset.role = role;
  canvas.dataset.channel = String(channelIdx);
  wrap.appendChild(canvas);
  return wrap;
}

function ensureFilteredCanvas(row, channelIdx) {
  if (row.querySelector('canvas[data-role="filtered"]')) return;
  const bandsWrap = row.querySelector('canvas[data-role="bands"]')?.parentElement;
  if (!bandsWrap) return;
  row.insertBefore(makePlotCanvas(channelIdx, "filtered"), bandsWrap);
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

      row.appendChild(label);
      row.appendChild(makePlotCanvas(trace.index, "waveform"));
      row.appendChild(makePlotCanvas(trace.index, "filtered"));
      row.appendChild(makePlotCanvas(trace.index, "bands"));
      container.appendChild(row);
    } else {
      ensureFilteredCanvas(row, trace.index);
    }
    row.querySelector("label").textContent = trace.name;
    row.dataset.showX = "1";
  });
}

function renderWaveform(waveform) {
  state.lastWaveform = waveform;
  const traces = waveform.traces || [];
  if (!traces.length) return;
  ensureChannelRows(traces);
  const timeS = waveform.time_s || [];
  renderBandLegend(els.legendEeg, waveform.eeg_bands, 0);
  renderBandLegend(els.legendEmg, waveform.emg_bands, 16);
  els.plotArea.querySelectorAll("canvas[data-channel]").forEach((canvas) => {
    const row = canvas.closest(".plot-row");
    const idx = Number(canvas.dataset.channel);
    const trace = traces.find((t) => t.index === idx);
    if (!trace) return;
    const showXAxis = row?.dataset.showX === "1";
    if (canvas.dataset.role === "bands") {
      const visibleBands = filterVisibleBands(trace.bands || [], idx);
      drawBandPlot(canvas, visibleBands, trace.band_time_s || [], showXAxis, idx);
    } else if (canvas.dataset.role === "filtered") {
      drawTrace(canvas, { ...trace, y: trace.y_filtered || trace.y }, timeS, showXAxis);
    } else {
      drawTrace(canvas, trace, timeS, showXAxis);
    }
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

let wordWeightPostTimer = null;
let negMixPostTimer = null;

function formatDistributionWeight(value) {
  return Number(value).toFixed(2);
}

function normalizedMixSummary(weights, labels) {
  const entries = Object.entries(weights || {}).filter(([, value]) => Number(value) > 0);
  const total = entries.reduce((sum, [, value]) => sum + Number(value), 0);
  if (!total) return "no active segments";
  return entries
    .map(([key, value]) => {
      const label = labels?.[key] || key;
      const pct = Math.round((Number(value) / total) * 100);
      return `${label} ${pct}%`;
    })
    .join(", ");
}

async function postWordWeight(word, weight) {
  try {
    await post("/collect/word-weights", { weights: { [word]: Number(weight) } });
  } catch (err) {
    showToast(err.message, true);
  }
}

function queueWordWeightPost(word, weight) {
  if (wordWeightPostTimer) window.clearTimeout(wordWeightPostTimer);
  wordWeightPostTimer = window.setTimeout(() => {
    wordWeightPostTimer = null;
    postWordWeight(word, weight);
  }, 120);
}

async function postNegMixWeight(key, weight) {
  try {
    await post("/collect/negative-label-mix", { weights: { [key]: Number(weight) } });
  } catch (err) {
    showToast(err.message, true);
  }
}

function queueNegMixWeightPost(key, weight) {
  if (negMixPostTimer) window.clearTimeout(negMixPostTimer);
  negMixPostTimer = window.setTimeout(() => {
    negMixPostTimer = null;
    postNegMixWeight(key, weight);
  }, 120);
}

function buildDistributionSliders({
  container,
  builtKey,
  items,
  collect,
  weightsKey,
  defaultsKey,
  onInput,
  inputDatasetKey,
}) {
  if (!container) return;
  if (container.dataset.built === builtKey) {
    syncDistributionSliderValues(container, collect, weightsKey, defaultsKey, inputDatasetKey);
    return;
  }

  const min = collect.word_weight_min ?? 0;
  const max = collect.word_weight_max ?? 1;
  const step = collect.word_weight_step ?? 0.05;
  const defaults = collect[defaultsKey] || {};

  container.innerHTML = "";
  for (const item of items) {
    const label = document.createElement("label");
    label.className = "distribution-slider";

    const name = document.createElement("span");
    name.textContent = item.label;

    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.id = `${item.idPrefix}-${item.key}`;
    input.dataset[inputDatasetKey] = item.key;
    input.value = String(defaults[item.key] ?? 0);

    const output = document.createElement("output");
    output.htmlFor = input.id;
    output.textContent = formatDistributionWeight(input.value);

    input.addEventListener("input", () => {
      input.dataset.touched = "1";
      output.textContent = formatDistributionWeight(input.value);
      onInput(item.key, input.value);
    });

    label.append(name, input, output);
    container.appendChild(label);
  }
  container.dataset.built = builtKey;
  syncDistributionSliderValues(container, collect, weightsKey, defaultsKey, inputDatasetKey);
}

function syncDistributionSliderValues(container, collect, weightsKey, defaultsKey, inputDatasetKey) {
  const weights = collect[weightsKey] || collect[defaultsKey] || {};
  container.querySelectorAll("input[type='range']").forEach((input) => {
    const key = input.dataset[inputDatasetKey];
    if (!key || input.dataset.touched === "1") return;
    if (weights[key] === undefined) return;
    input.value = String(weights[key]);
    const output = input.parentElement?.querySelector("output");
    if (output) output.textContent = formatDistributionWeight(input.value);
  });
}

function setDistributionSlidersDisabled(container, disabled) {
  if (!container) return;
  container.querySelectorAll("input[type='range']").forEach((input) => {
    input.disabled = disabled;
  });
}

function ensureWordWeightSliders(collect = {}) {
  const words = collect.words || [];
  buildDistributionSliders({
    container: els.wordWeightSliders,
    builtKey: words.join(","),
    items: words.map((word) => ({
      key: word,
      label: word,
      idPrefix: "word-weight",
    })),
    collect,
    weightsKey: "word_weights",
    defaultsKey: "default_word_weights",
    inputDatasetKey: "word",
    onInput: queueWordWeightPost,
  });
}

function ensureNegativeLabelMixSliders(collect = {}) {
  const keys = collect.neg_mix_keys || ["still", "negative_word", "positive_word"];
  const labels = collect.neg_mix_labels || {};
  buildDistributionSliders({
    container: els.negativeLabelMixSliders,
    builtKey: keys.join(","),
    items: keys.map((key) => ({
      key,
      label: labels[key] || key,
      idPrefix: "neg-mix",
    })),
    collect,
    weightsKey: "neg_mix_weights",
    defaultsKey: "default_neg_mix_weights",
    inputDatasetKey: "mixKey",
    onInput: queueNegMixWeightPost,
  });
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
      els.scrambleRep.value = String(collect.default_scramble_rep ?? 5);
    }
    els.scrambleRepVal.textContent = els.scrambleRep.value;
  }
}

function flashWordSwitchEdges() {
  const el = els.wordSwitchFlash;
  if (!el) return;
  el.classList.remove("active");
  void el.offsetWidth;
  el.classList.add("active");
}

function maybeFlashWordSwitch(collect) {
  if (collect.phase !== "countdown" || !collect.word_switch) return;
  const key = `${collect.word}-${collect.set_index ?? 0}`;
  if (state.wordSwitchFlashKey === key) return;
  state.wordSwitchFlashKey = key;
  flashWordSwitchEdges();
}

function updateCollectUi(status) {
  const collect = status.collect || { phase: "disabled", words: [] };
  const recording = status.recording_enabled;
  const phase = collect.phase;
  const mode = collect.mode || "single";

  if (phase === "disabled" || phase === "pick_word") {
    state.wordSwitchFlashKey = null;
  } else {
    maybeFlashWordSwitch(collect);
  }

  syncScrambleSliderLabels(collect);
  ensureWordButtons(collect.words || []);
  ensureWordWeightSliders(collect);
  ensureNegativeLabelMixSliders(collect);

  const picking = recording && phase === "pick_word";
  const busy = phase === "countdown" || phase === "say" || phase === "still";
  const negativeLabels = mode === "negative_labels";
  const mixSummary = normalizedMixSummary(
    collect.neg_mix_weights || collect.default_neg_mix_weights,
    collect.neg_mix_labels,
  );

  els.collectPanel.classList.toggle("hidden", !recording);
  const beforeS = collect.before_s ?? 1;
  const betweenS = collect.between_s ?? 0.4;
  const wordSwitchS = collect.word_switch_s ?? beforeS + 0.3;
  const sayS = collect.say_s ?? 1.6;
  els.collectHint.textContent = negativeLabels && busy
    ? `Negative labels running — target mix: ${mixSummary} — click Stop Negative Labels when finished`
    : picking
      ? `Choose a word, tune distributions below, then Scramble or Negative Labels — Fast: ${beforeS}s / ${betweenS}s / ${wordSwitchS}s gaps; Breaks: ${sayS}s between reps`
      : "Recording active — finish current collection to pick another";

  els.wordButtons.querySelectorAll(".word-btn").forEach((btn) => {
    btn.disabled = !picking;
  });
  if (els.scrambleFastBtn) els.scrambleFastBtn.disabled = !picking;
  if (els.scrambleBreaksBtn) els.scrambleBreaksBtn.disabled = !picking;
  if (els.scrambleSet) els.scrambleSet.disabled = !picking;
  if (els.scrambleRep) els.scrambleRep.disabled = !picking;
  setDistributionSlidersDisabled(els.wordWeightSliders, !picking);
  setDistributionSlidersDisabled(els.negativeLabelMixSliders, !picking);
  if (els.negativeLabelsBtn) {
    els.negativeLabelsBtn.disabled = !recording || (!picking && !negativeLabels);
    els.negativeLabelsBtn.textContent = negativeLabels && busy ? "Stop Negative Labels" : "Negative Labels";
    els.negativeLabelsBtn.classList.toggle("active", negativeLabels && busy);
  }

  els.collectPrompt.classList.toggle("hidden", !busy);
  const wordSwitchCountdown = phase === "countdown" && !!collect.word_switch;
  els.collectPrompt.classList.toggle("word-switch", wordSwitchCountdown);
  if (busy) {
    const word = collect.word || "";
    const rep = collect.repetition ?? 1;
    const total = collect.repetitions_total ?? 7;
    const setIdx = collect.set_index ?? 1;
    const setTotal = collect.sets_total ?? 1;
    const remaining = collect.phase_remaining_s ?? 0;
    const scramble = mode === "scramble" || mode === "scramble-breaks";
    const negKind = collect.neg_segment_kind || "";

    if (phase === "still") {
      els.collectPromptWord.textContent = "Sit still";
      els.collectPromptMain.textContent = remaining.toFixed(1);
      els.collectPromptMain.classList.remove("say-it");
      els.collectPromptSub.textContent = "Negative labels — no speech";
    } else {
      els.collectPromptWord.textContent = wordSwitchCountdown ? "Next word" : word;

      const setPrefix = scramble ? `Set ${setIdx}/${setTotal} · ` : "";
      if (phase === "countdown") {
        els.collectPromptMain.textContent = wordSwitchCountdown ? word : remaining.toFixed(1);
        els.collectPromptMain.classList.remove("say-it");
        els.collectPromptSub.textContent = wordSwitchCountdown
          ? `${setPrefix}${remaining.toFixed(1)}s — new word`
          : collect.trailing_break
            ? `${setPrefix}final break — ${remaining.toFixed(1)}s`
          : negativeLabels
            ? `${negKind === "positive_word" ? "Target word" : "Off-list word"} — get ready`
            : scramble && rep === 1
              ? `${setPrefix}new word — get ready`
              : `${setPrefix}Rep ${rep}/${total} — get ready`;
      } else {
        els.collectPromptMain.textContent = "SAY IT";
        els.collectPromptMain.classList.add("say-it");
        els.collectPromptSub.textContent = negativeLabels
          ? `${negKind === "positive_word" ? "Target word" : "Off-list word"} — speak “${word}” now`
          : scramble
            ? `${setPrefix}Rep ${rep}/${total} — speak “${word}” now`
            : `Repetition ${rep}/${total} — speak “${word}” now`;
      }
    }
  }

  let trialText = "Collection: start recording";
  if (!recording) {
    trialText = "Collection: not recording";
  } else if (phase === "pick_word") {
    trialText = "Collection: choose a word, scramble fast/breaks, or negative labels";
  } else if (negativeLabels) {
    if (phase === "still") {
      trialText = "Negative labels: sit still";
    } else if (phase === "countdown") {
      trialText = `Negative labels: “${collect.word}” — countdown`;
    } else if (phase === "say") {
      trialText = `Negative labels: “${collect.word}” — say it`;
    }
  } else if (mode === "scramble" || mode === "scramble-breaks") {
    const scrambleLabel = mode === "scramble-breaks" ? "Scramble Breaks" : "Scramble Fast";
    if (phase === "countdown") {
      trialText = `${scrambleLabel}: “${collect.word}” set ${collect.set_index}/${collect.sets_total} rep ${collect.repetition}/${collect.repetitions_total} — countdown`;
    } else if (phase === "say") {
      trialText = `${scrambleLabel}: “${collect.word}” set ${collect.set_index}/${collect.sets_total} rep ${collect.repetition}/${collect.repetitions_total} — say it`;
    }
  } else if (phase === "countdown") {
    trialText = `Collection: “${collect.word}” rep ${collect.repetition}/${collect.repetitions_total} — countdown`;
  } else if (phase === "say") {
    trialText = `Collection: “${collect.word}” rep ${collect.repetition}/${collect.repetitions_total} — say it`;
  }
  els.trial.textContent = trialText;
}

function updateAlignmentUi(status) {
  const align = status.alignment_test || { phase: "idle", before_s: 1, say_s: 1.6, has_result: false };
  const recording = status.recording_enabled;
  const phase = align.phase;
  const busy = phase === "countdown" || phase === "blink";
  const canStart =
    !recording &&
    !busy &&
    status.trial_state === "idle" &&
    (status.collect?.phase === "disabled" || status.collect?.phase === "pick_word") &&
    status.model_test?.phase !== "countdown" &&
    status.model_test?.phase !== "say" &&
    !status.model_use?.active;

  if (els.alignmentBtn) {
    els.alignmentBtn.disabled = !canStart;
    els.alignmentBtn.textContent = busy ? "Alignment test…" : "Alignment Test";
  }

  els.alignmentPrompt?.classList.toggle("hidden", !busy);
  if (busy && els.alignmentPromptMain && els.alignmentPromptSub) {
    const remaining = align.phase_remaining_s ?? 0;
    const beforeS = align.before_s ?? 1;
    const sayS = align.say_s ?? 1.6;
    if (phase === "countdown") {
      els.alignmentPromptMain.textContent = remaining.toFixed(1);
      els.alignmentPromptMain.classList.remove("blink-it");
      els.alignmentPromptSub.textContent = `Get ready — blink test starts in ${beforeS}s window`;
    } else {
      els.alignmentPromptMain.textContent = "BLINK ×3";
      els.alignmentPromptMain.classList.add("blink-it");
      els.alignmentPromptSub.textContent = `Blink 3 times now (${sayS}s labeled window)`;
    }
  }

  if (phase === "result" && align.has_result && state.alignmentResultFetchKey !== "pending") {
    if (!state.alignmentResult) {
      state.alignmentResultFetchKey = "pending";
      fetchAlignmentResult().catch((err) => showToast(err.message, true));
    }
  } else if (phase !== "result") {
    state.alignmentResultFetchKey = null;
  }

  const showResult = phase === "result" && !!state.alignmentResult && !state.alignmentResult.error;
  els.alignmentResult?.classList.toggle("hidden", !showResult);
  if (showResult) {
    renderAlignmentResult(state.alignmentResult);
  }
}

function renderAlignmentResult(result) {
  if (!els.alignmentResultMeta || !els.alignmentResultPlots) return;
  const idxStart = result.sample_index_start;
  const idxEnd = result.sample_index_end;
  const method = result.alignment_method || "host_time_regression";
  els.alignmentResultMeta.textContent = `samples ${idxStart}–${idxEnd} · ${method} · ${result.say_s ?? 1.6}s window`;
  els.alignmentResultPlots.replaceChildren();

  const timeS = result.time_s || [];
  const labelT0 = result.label_t0_s;
  const labelT1 = result.label_t1_s;
  for (const trace of result.traces || []) {
    const row = document.createElement("div");
    row.className = "alignment-trace-row";

    const label = document.createElement("div");
    label.className = "alignment-trace-label";
    label.textContent = trace.name;

    const wrap = document.createElement("div");
    wrap.className = "alignment-trace-wrap";
    const canvas = document.createElement("canvas");
    wrap.appendChild(canvas);
    row.append(label, wrap);
    els.alignmentResultPlots.appendChild(row);

    const filteredTrace = { y: trace.y_filtered || trace.y };
    drawAlignmentTrace(canvas, filteredTrace, timeS, { labelT0, labelT1, showXAxis: true });
  }
}

async function fetchAlignmentResult() {
  const result = await api("/alignment-test/result");
  state.alignmentResult = result;
  state.alignmentResultFetchKey = "done";
  if (result.error) {
    showToast(result.error, true);
    els.alignmentResult?.classList.add("hidden");
    return;
  }
  renderAlignmentResult(result);
}

function updateModelTestUi(status) {
  const model = status.model_test || {
    phase: "idle",
    before_s: 1,
    say_s: 1.6,
    trials_total: 3,
    has_result: false,
  };
  const recording = status.recording_enabled;
  const phase = model.phase;
  const busy = phase === "countdown" || phase === "say";
  const canStart =
    !recording &&
    !busy &&
    status.trial_state === "idle" &&
    (status.collect?.phase === "disabled" || status.collect?.phase === "pick_word") &&
    status.alignment_test?.phase !== "countdown" &&
    status.alignment_test?.phase !== "blink" &&
    !status.model_use?.active;

  if (els.modelTestBtn) {
    els.modelTestBtn.disabled = !canStart;
    els.modelTestBtn.textContent = busy ? "Model test…" : "Test Model";
  }

  els.modelTestPrompt?.classList.toggle("hidden", !busy);
  if (busy && els.modelTestPromptMain && els.modelTestPromptSub) {
    const remaining = model.phase_remaining_s ?? 0;
    const beforeS = model.before_s ?? 1;
    const sayS = model.say_s ?? 1.6;
    const trial = model.trial ?? 1;
    const total = model.trials_total ?? 3;
    if (phase === "countdown") {
      els.modelTestPromptMain.textContent = remaining.toFixed(1);
      els.modelTestPromptMain.classList.remove("say-it");
      els.modelTestPromptSub.textContent = `Trial ${trial}/${total} — get ready (${beforeS}s new-word countdown)`;
    } else {
      els.modelTestPromptMain.textContent = "SAY IT";
      els.modelTestPromptMain.classList.add("say-it");
      els.modelTestPromptSub.textContent = `Trial ${trial}/${total} — say any word (${sayS}s)`;
    }
  }

  if (phase === "result" && model.has_result && state.modelTestResultFetchKey !== "pending") {
    if (!state.modelTestResult) {
      state.modelTestResultFetchKey = "pending";
      fetchModelTestResult().catch((err) => showToast(err.message, true));
    }
  } else if (phase !== "result") {
    state.modelTestResultFetchKey = null;
  }

  const showResult = phase === "result" && !!state.modelTestResult;
  els.modelTestResult?.classList.toggle("hidden", !showResult);
  if (showResult) {
    renderModelTestResult(state.modelTestResult);
  }
}

function formatPct(probability) {
  const pct = Number(probability) * 100;
  if (!Number.isFinite(pct)) return "—";
  if (pct > 0 && pct < 0.01) return "<0.01%";
  return `${pct.toFixed(2)}%`;
}

function barWidthPct(probability) {
  const pct = Number(probability) * 100;
  if (!Number.isFinite(pct) || pct <= 0) return 0;
  // Keep tiny non-zero probs visible on the bar chart.
  return Math.max(pct, 0.4);
}

function renderModelTestResult(result) {
  if (!els.modelTestResultMeta || !els.modelTestResultRows) return;
  const checkpoint = result.checkpoint || "";
  const ckptName = checkpoint.split("/").pop() || checkpoint;
  els.modelTestResultMeta.textContent = checkpoint ? `checkpoint: ${ckptName}` : "";
  els.modelTestResultRows.replaceChildren();

  for (const trial of result.trials || []) {
    const row = document.createElement("div");
    row.className = "model-test-trial";

    const head = document.createElement("div");
    head.className = "model-test-trial-head";
    const title = document.createElement("strong");
    title.textContent = `Trial ${trial.trial}`;
    head.appendChild(title);

    if (trial.error) {
      const err = document.createElement("span");
      err.className = "model-test-error";
      err.textContent = trial.error;
      head.appendChild(err);
    } else if (trial.predicted_label) {
      const guess = document.createElement("span");
      guess.className = "model-test-guess";
      guess.textContent = `→ ${trial.predicted_label} (${formatPct(trial.predicted_probability)})`;
      head.appendChild(guess);
    }
    row.appendChild(head);

    const labelRows = trial.predictions_by_label || trial.predictions || [];
    if (!trial.error && labelRows.length) {
      const bars = document.createElement("div");
      bars.className = "model-test-bars";
      for (const pred of labelRows) {
        const item = document.createElement("div");
        item.className = "model-test-bar-row";

        const label = document.createElement("span");
        label.className = "model-test-bar-label";
        label.textContent = pred.label;

        const track = document.createElement("div");
        track.className = "model-test-bar-track";
        const fill = document.createElement("div");
        fill.className = "model-test-bar-fill";
        if (pred.label === trial.predicted_label) {
          fill.classList.add("is-top");
        }
        fill.style.width = `${Math.min(100, barWidthPct(pred.probability))}%`;
        track.appendChild(fill);

        const pct = document.createElement("span");
        pct.className = "model-test-bar-pct";
        pct.textContent = formatPct(pred.probability);

        item.append(label, track, pct);
        bars.appendChild(item);
      }
      row.appendChild(bars);
    }

    if (trial.probs_by_label && typeof trial.probs_by_label === "object") {
      const summary = document.createElement("div");
      summary.className = "model-test-debug";
      summary.textContent = Object.entries(trial.probs_by_label)
        .map(([label, prob]) => `${label} ${formatPct(prob)}`)
        .join(" · ");
      row.appendChild(summary);
    } else if (Array.isArray(trial.logits) && trial.logits.length) {
      const debug = document.createElement("div");
      debug.className = "model-test-debug";
      debug.textContent = `logits: [${trial.logits.map((v) => Number(v).toFixed(3)).join(", ")}]`;
      row.appendChild(debug);
    }

    els.modelTestResultRows.appendChild(row);
  }
}

async function fetchModelTestResult() {
  const result = await api("/model-test/result");
  state.modelTestResult = result;
  state.modelTestResultFetchKey = "done";
  renderModelTestResult(result);
}

function renderModelUseBars(container, latest) {
  if (!container) return;
  container.replaceChildren();
  const labelRows = (latest?.predictions || latest?.predictions_by_label || []).slice().sort(
    (a, b) => (b.probability ?? 0) - (a.probability ?? 0),
  );
  if (!labelRows.length || latest?.error) return;

  for (const pred of labelRows) {
    const item = document.createElement("div");
    item.className = "model-test-bar-row";

    const label = document.createElement("span");
    label.className = "model-test-bar-label";
    label.textContent = pred.label;

    const track = document.createElement("div");
    track.className = "model-test-bar-track";
    const fill = document.createElement("div");
    fill.className = "model-test-bar-fill";
    if (pred.label === latest.predicted_label) {
      fill.classList.add("is-top");
    }
    fill.style.width = `${Math.min(100, barWidthPct(pred.probability))}%`;
    track.appendChild(fill);

    const pct = document.createElement("span");
    pct.className = "model-test-bar-pct";
    pct.textContent = formatPct(pred.probability);

    item.append(label, track, pct);
    container.appendChild(item);
  }
}

function updateModelUseUi(status) {
  const modelUse = status.model_use || {
    active: false,
    window_s: 1.6,
    interval_s: 0.4,
    prediction_count: 0,
    latest: null,
  };
  const active = !!modelUse.active;
  const modelTestBusy =
    status.model_test?.phase === "countdown" || status.model_test?.phase === "say";
  const alignBusy =
    status.alignment_test?.phase === "countdown" || status.alignment_test?.phase === "blink";
  const canStart = !active && !modelTestBusy && !alignBusy;

  if (els.modelUseBtn) {
    els.modelUseBtn.disabled = !canStart && !active;
    els.modelUseBtn.textContent = active ? "Stop Model" : "Use Model";
    els.modelUseBtn.classList.toggle("primary", active);
  }

  els.modelUsePanel?.classList.toggle("hidden", !active);

  if (!active || !els.modelUseMeta || !els.modelUsePrediction) {
    return;
  }

  const checkpoint = modelUse.checkpoint || "";
  const ckptName = checkpoint.split("/").pop() || checkpoint;
  const windowS = modelUse.window_s ?? 1.6;
  const intervalS = modelUse.interval_s ?? 0.4;
  const count = modelUse.prediction_count ?? 0;
  els.modelUseMeta.textContent = [
    ckptName ? `checkpoint: ${ckptName}` : null,
    `${windowS}s window · every ${intervalS}s`,
    count ? `${count} predictions` : "waiting…",
  ]
    .filter(Boolean)
    .join(" · ");

  els.modelUsePrediction.replaceChildren();
  const latest = modelUse.latest;
  if (!latest) {
    const waiting = document.createElement("span");
    waiting.className = "model-use-confidence";
    waiting.textContent = "Waiting for first prediction…";
    els.modelUsePrediction.appendChild(waiting);
    els.modelUseBars?.replaceChildren();
    return;
  }

  if (latest.error) {
    const err = document.createElement("span");
    err.className = "model-use-error";
    err.textContent = latest.error;
    els.modelUsePrediction.appendChild(err);
    els.modelUseBars?.replaceChildren();
    return;
  }

  const label = document.createElement("span");
  label.className = "model-use-label";
  label.textContent = latest.predicted_label || "—";

  const confidence = document.createElement("span");
  confidence.className = "model-use-confidence";
  confidence.textContent = latest.predicted_probability
    ? formatPct(latest.predicted_probability)
    : "";

  els.modelUsePrediction.append(label, confidence);
  renderModelUseBars(els.modelUseBars, latest);
}

function syncAlignmentStatusPolling(status) {
  const align = status.alignment_test || {};
  const model = status.model_test || {};
  const modelUse = status.model_use || {};
  const phase = align.phase;
  const modelPhase = model.phase;
  const fast =
    phase === "countdown" ||
    phase === "blink" ||
    modelPhase === "countdown" ||
    modelPhase === "say" ||
    !!modelUse.active;
  const hz = fast ? 20 : 0;
  if (hz > 0) {
    if (state.statusPollTimer) return;
    state.statusPollTimer = window.setInterval(async () => {
      try {
        const next = await api("/status");
        updateStatusBar(next);
        updateControls(next);
        updateAlignmentUi(next);
        updateModelTestUi(next);
        updateModelUseUi(next);
        const alignNext = next.alignment_test?.phase;
        const modelNext = next.model_test?.phase;
        const modelUseNext = next.model_use?.active;
        if (
          alignNext !== "countdown" &&
          alignNext !== "blink" &&
          modelNext !== "countdown" &&
          modelNext !== "say" &&
          !modelUseNext
        ) {
          window.clearInterval(state.statusPollTimer);
          state.statusPollTimer = null;
        }
      } catch (err) {
        /* dashboard poll will surface errors */
      }
    }, 1000 / hz);
  } else if (state.statusPollTimer) {
    window.clearInterval(state.statusPollTimer);
    state.statusPollTimer = null;
  }
}

function updateControls(status) {
  const recording = status.recording_enabled;
  const collect = status.collect || { phase: "disabled" };
  const canStopRecording = collect.phase === "disabled" || collect.phase === "pick_word";

  document.getElementById("btn-record").textContent = recording
    ? "Stop Session Recording"
    : "Start Session Recording";
  document.getElementById("btn-record").disabled =
    (recording && !canStopRecording) || (recording && status.pending_asr_jobs > 0);

  updateCollectUi(status);
  updateAlignmentUi(status);
  updateModelTestUi(status);
  updateModelUseUi(status);
  syncAlignmentStatusPolling(status);
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

async function refreshStatus() {
  const status = await api("/status");
  updateStatusBar(status);
  updateControls(status);
  return status;
}

async function poll() {
  if (state.pollInFlight) return;
  state.pollInFlight = true;
  const abort = new AbortController();
  state.pollAbort = abort;
  try {
    const params = new URLSearchParams({ mode: "all", page_start: "0", single_channel: "0" });
    const data = await api(`/api/dashboard?${params}`, { signal: abort.signal });
    const status = data.status;
    state.channelCount = status.channel_count || 32;
    if (!state.autoScale && status.default_fixed_scale_uv) {
      state.amplitudeUv = Number(els.amplitudeSlider.value) || status.default_fixed_scale_uv;
    }
    updateStatusBar(status);
    updateControls(status);
    renderWaveform(data.waveform);
  } catch (err) {
    if (err.name === "AbortError") return;
    els.status.textContent = `Poll error: ${err.message}`;
  } finally {
    state.pollInFlight = false;
    if (state.pollAbort === abort) state.pollAbort = null;
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
  els.amplitudeSlider.disabled = state.autoScale;
  if (state.autoScale) {
    els.amplitudeVal.textContent = "per graph";
  } else {
    state.amplitudeUv = Number(els.amplitudeSlider.value) || DEFAULT_AMPLITUDE_UV;
    syncAmplitudeUi();
  }
  redrawIfReady();
});

els.amplitudeSlider.addEventListener("input", () => {
  state.amplitudeUv = Number(els.amplitudeSlider.value) || DEFAULT_AMPLITUDE_UV;
  els.amplitudeVal.textContent = formatAmplitudeLabel(state.amplitudeUv);
  redrawIfReady();
});

document.getElementById("btn-record").addEventListener("click", async () => {
  const btn = document.getElementById("btn-record");
  const recording = btn.textContent.startsWith("Stop");
  state.pollAbort?.abort();
  btn.disabled = true;
  try {
    if (recording) {
      const body = await post("/recording/stop");
      showToast(`Stopped: ${body.session_dir || "session"}`);
    } else {
      const body = await post("/recording/start", {});
      showToast(`Recording: ${body.session_dir}`);
    }
    await refreshStatus();
  } catch (err) {
    showToast(err.message, true);
    try {
      await refreshStatus();
    } catch (_) {
      /* keep existing UI if status refresh also fails */
    }
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

async function startScramble(endpoint) {
  try {
    await post(endpoint, {
      set: Number(els.scrambleSet.value),
      rep: Number(els.scrambleRep.value),
    });
  } catch (err) {
    showToast(err.message, true);
  }
}

if (els.scrambleFastBtn) {
  els.scrambleFastBtn.addEventListener("click", () => startScramble("/collect/scramble"));
}

if (els.scrambleBreaksBtn) {
  els.scrambleBreaksBtn.addEventListener("click", () => startScramble("/collect/scramble-breaks"));
}

if (els.negativeLabelsBtn) {
  els.negativeLabelsBtn.addEventListener("click", async () => {
    try {
      await post("/collect/negative-labels/toggle");
    } catch (err) {
      showToast(err.message, true);
    }
  });
}

if (els.alignmentBtn) {
  els.alignmentBtn.addEventListener("click", async () => {
    els.alignmentBtn.disabled = true;
    state.alignmentResult = null;
    state.alignmentResultFetchKey = null;
    els.alignmentResult?.classList.add("hidden");
    try {
      await post("/alignment-test/start");
      await refreshStatus();
    } catch (err) {
      showToast(err.message, true);
      try {
        await refreshStatus();
      } catch (_) {
        /* keep existing UI */
      }
    }
  });
}

if (els.alignmentDismiss) {
  els.alignmentDismiss.addEventListener("click", async () => {
    try {
      await post("/alignment-test/clear");
      state.alignmentResult = null;
      state.alignmentResultFetchKey = null;
      els.alignmentResult?.classList.add("hidden");
      await refreshStatus();
    } catch (err) {
      showToast(err.message, true);
    }
  });
}

if (els.modelTestBtn) {
  els.modelTestBtn.addEventListener("click", async () => {
    els.modelTestBtn.disabled = true;
    state.modelTestResult = null;
    state.modelTestResultFetchKey = null;
    els.modelTestResult?.classList.add("hidden");
    try {
      await post("/model-test/start");
      await refreshStatus();
    } catch (err) {
      showToast(err.message, true);
      try {
        await refreshStatus();
      } catch (_) {
        /* keep existing UI */
      }
    }
  });
}

if (els.modelUseBtn) {
  els.modelUseBtn.addEventListener("click", async () => {
    const active = els.modelUseBtn.classList.contains("primary");
    els.modelUseBtn.disabled = true;
    try {
      await post(active ? "/model-use/stop" : "/model-use/start");
      await refreshStatus();
    } catch (err) {
      showToast(err.message, true);
      try {
        await refreshStatus();
      } catch (_) {
        /* keep existing UI */
      }
    }
  });
}

if (els.modelTestDismiss) {
  els.modelTestDismiss.addEventListener("click", async () => {
    try {
      await post("/model-test/clear");
      state.modelTestResult = null;
      state.modelTestResultFetchKey = null;
      els.modelTestResult?.classList.add("hidden");
      await refreshStatus();
    } catch (err) {
      showToast(err.message, true);
    }
  });
}

window.addEventListener("resize", () => {
  window.requestAnimationFrame(redrawIfReady);
  if (state.alignmentResult) {
    window.requestAnimationFrame(() => renderAlignmentResult(state.alignmentResult));
  }
});

const resizeObserver = new ResizeObserver(() => {
  window.requestAnimationFrame(redrawIfReady);
});
resizeObserver.observe(els.plotArea);

startPolling();
