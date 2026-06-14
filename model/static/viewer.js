const state = {
  sessions: [],
  sessionIndex: 0,
  mode: "timeline",
  windowMs: 200,
  preMs: 100,
  postMs: 100,
  targetRow: 0,
  events: [],
  eventIndex: 0,
  plotCells: [],
  fetchToken: 0,
  lastPayload: null,
};

const els = {
  sessionBar: document.getElementById("session-bar"),
  sessionMeta: document.getElementById("session-meta"),
  windowMs: document.getElementById("window-ms"),
  preMs: document.getElementById("pre-ms"),
  postMs: document.getElementById("post-ms"),
  applyMargins: document.getElementById("apply-margins"),
  marginInfo: document.getElementById("margin-info"),
  timelinePanel: document.getElementById("timeline-panel"),
  eventsPanel: document.getElementById("events-panel"),
  timelineSlider: document.getElementById("timeline-slider"),
  timelineInfo: document.getElementById("timeline-info"),
  eventBar: document.getElementById("event-bar"),
  eventInfo: document.getElementById("event-info"),
  eventPrev: document.getElementById("event-prev"),
  eventNext: document.getElementById("event-next"),
  eventJump: document.getElementById("event-jump"),
  eventGo: document.getElementById("event-go"),
  plotTitle: document.getElementById("plot-title"),
  plotScroll: document.getElementById("plot-scroll"),
  plotGrid: document.getElementById("plot-grid"),
};

function marginsQuery() {
  return `pre_ms=${state.preMs}&post_ms=${state.postMs}`;
}

function timelineQuery() {
  return `${marginsQuery()}&window_ms=${state.windowMs}`;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function buildIndexBar(container, count, active, onSelect) {
  container.replaceChildren();
  for (let i = 0; i < count; i += 1) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = String(i);
    if (i === active) {
      button.classList.add("active");
    }
    button.addEventListener("click", () => onSelect(i));
    container.appendChild(button);
  }
}

function ensurePlotCells() {
  if (state.plotCells.length === 32) {
    return;
  }
  els.plotGrid.replaceChildren();
  state.plotCells = [];
  for (let ch = 0; ch < 32; ch += 1) {
    const cell = document.createElement("div");
    cell.className = "channel-cell";
    const title = document.createElement("h3");
    const canvas = document.createElement("canvas");
    cell.append(title, canvas);
    els.plotGrid.appendChild(cell);
    state.plotCells.push({ cell, title, canvas });
  }
}

function updatePlotLayout(plotWidthCm) {
  const scrollWidth = els.plotScroll.clientWidth;
  const plotWidthPx = (plotWidthCm / 2.54) * 96;
  const wide = plotWidthPx >= scrollWidth * 0.92;
  els.plotGrid.style.setProperty("--plot-width", `${plotWidthCm}cm`);
  els.plotGrid.classList.toggle("wide", wide);
}

function niceTicks(min, max, count = 4) {
  if (min === max) {
    return [min];
  }
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = span / step / count;
  let niceStep = step;
  if (err >= 7.5) niceStep = step * 10;
  else if (err >= 3.5) niceStep = step * 5;
  else if (err >= 1.5) niceStep = step * 2;
  const start = Math.ceil(min / niceStep) * niceStep;
  const ticks = [];
  for (let v = start; v <= max + niceStep * 0.01; v += niceStep) {
    ticks.push(v);
  }
  if (ticks.length === 0) {
    return [min, max];
  }
  return ticks;
}

function formatTick(value, decimals = 1) {
  if (Math.abs(value) >= 1000) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 10) {
    return value.toFixed(decimals > 0 ? 0 : decimals);
  }
  return value.toFixed(decimals);
}

function drawChannel(canvas, timeMs, values, markerStart, markerEnd) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);

  const marginLeft = 44;
  const marginRight = 8;
  const marginTop = 8;
  const marginBottom = 26;
  const plotLeft = marginLeft;
  const plotTop = marginTop;
  const plotW = Math.max(1, width - marginLeft - marginRight);
  const plotH = Math.max(1, height - marginTop - marginBottom);

  const axisColor = getComputedStyle(document.documentElement).getPropertyValue("--muted").trim();
  const gridColor = "#3a3d44";
  const textColor = axisColor;

  ctx.strokeStyle = axisColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plotLeft, plotTop);
  ctx.lineTo(plotLeft, plotTop + plotH);
  ctx.lineTo(plotLeft + plotW, plotTop + plotH);
  ctx.stroke();

  if (!timeMs.length || !values.length) {
    ctx.fillStyle = textColor;
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillText("ms", plotLeft + plotW - 14, height - 6);
    ctx.fillText("µV", 4, plotTop + 10);
    return;
  }

  const tMin = timeMs[0];
  const tMax = timeMs[timeMs.length - 1];
  const tSpan = tMax - tMin || 1;
  let yMin = values[0];
  let yMax = values[0];
  for (const y of values) {
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  const yPad = (yMax - yMin || 1) * 0.08;
  yMin -= yPad;
  yMax += yPad;
  const ySpan = yMax - yMin || 1;

  const xAt = (t) => plotLeft + ((t - tMin) / tSpan) * plotW;
  const yAt = (v) => plotTop + plotH - ((v - yMin) / ySpan) * plotH;

  const xTicks = niceTicks(tMin, tMax, 4);
  const yTicks = niceTicks(yMin, yMax, 3);

  ctx.strokeStyle = gridColor;
  ctx.lineWidth = 1;
  ctx.font = "9px system-ui, sans-serif";
  ctx.fillStyle = textColor;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";

  for (const t of xTicks) {
    const x = xAt(t);
    ctx.beginPath();
    ctx.moveTo(x, plotTop);
    ctx.lineTo(x, plotTop + plotH);
    ctx.stroke();
    ctx.fillText(formatTick(t, 1), x, plotTop + plotH + 4);
  }

  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of yTicks) {
    const y = yAt(v);
    ctx.beginPath();
    ctx.moveTo(plotLeft, y);
    ctx.lineTo(plotLeft + plotW, y);
    ctx.stroke();
    ctx.fillText(formatTick(v, 1), plotLeft - 4, y);
  }

  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--trace").trim();
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < values.length; i += 1) {
    const x = xAt(timeMs[i]);
    const y = yAt(values[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--marker").trim();
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  for (const t of [markerStart, markerEnd]) {
    const x = xAt(t);
    ctx.beginPath();
    ctx.moveTo(x, plotTop);
    ctx.lineTo(x, plotTop + plotH);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  ctx.fillStyle = textColor;
  ctx.font = "10px system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("ms", plotLeft + plotW, height - 4);
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("µV", 4, plotTop + 2);
}

function renderPlot(payload) {
  state.lastPayload = payload;
  ensurePlotCells();
  updatePlotLayout(payload.plot_width_cm);

  els.plotTitle.textContent = payload.title || "";
  const timeMs = payload.time_ms || [];
  const names = payload.channel_names || [];
  const channels = payload.channels || [];
  const markerStart = payload.marker_start_ms ?? 0;
  const markerEnd = payload.marker_end_ms ?? 0;

  for (let ch = 0; ch < state.plotCells.length; ch += 1) {
    const { title, canvas } = state.plotCells[ch];
    title.textContent = names[ch] || `Ch ${ch}`;
    drawChannel(canvas, timeMs, channels[ch] || [], markerStart, markerEnd);
  }
}

function setMarginInfo(text, isError = false) {
  els.marginInfo.textContent = text;
  els.marginInfo.classList.toggle("error", isError);
}

function readMargins() {
  state.preMs = Number(els.preMs.value);
  state.postMs = Number(els.postMs.value);
  state.windowMs = Number(els.windowMs.value);
}

function updateMarginInfo() {
  setMarginInfo(`pre ${state.preMs} ms · post ${state.postMs} ms (shared)`);
}

function currentMode() {
  return document.querySelector('input[name="mode"]:checked')?.value || "timeline";
}

async function loadSessions() {
  const data = await fetchJson("/api/sessions");
  state.sessions = data.sessions;
  buildIndexBar(els.sessionBar, state.sessions.length, state.sessionIndex, selectSession);
  updateSessionMeta();
}

function updateSessionMeta() {
  const session = state.sessions[state.sessionIndex];
  if (!session) {
    els.sessionMeta.textContent = "";
    return;
  }
  if (session.error) {
    els.sessionMeta.textContent = session.error;
    return;
  }
  els.sessionMeta.textContent =
    `${state.sessionIndex + 1}/${state.sessions.length} — ${session.name} ` +
    `(${session.frame_count} frames, ${session.duration_s} s, ${session.sample_rate_hz} Hz)`;
}

async function updateTimelineRange() {
  readMargins();
  const data = await fetchJson(
    `/api/sessions/${state.sessionIndex}/timeline-range?window_ms=${state.windowMs}&pre_ms=${state.preMs}`
  );
  els.timelineSlider.min = String(data.min_row);
  els.timelineSlider.max = String(data.max_row);
  state.targetRow = Math.max(data.min_row, Math.min(state.targetRow, data.max_row));
  els.timelineSlider.value = String(state.targetRow);
}

async function loadEvents() {
  const data = await fetchJson(`/api/sessions/${state.sessionIndex}/events`);
  state.events = data.events || [];
  if (data.message) {
    els.eventInfo.textContent = data.message;
  } else {
    const suffix = data.skipped ? ` (${data.skipped} skipped)` : "";
    els.eventInfo.textContent = `${state.events.length} plottable events${suffix}`;
  }
  state.eventIndex = Math.min(state.eventIndex, Math.max(0, state.events.length - 1));
  buildIndexBar(els.eventBar, state.events.length, state.eventIndex, selectEvent);
}

async function refreshPlot() {
  const token = ++state.fetchToken;
  let payload;
  try {
    if (state.mode === "timeline") {
      readMargins();
      payload = await fetchJson(
        `/api/sessions/${state.sessionIndex}/timeline?${timelineQuery()}` +
          `&target_row=${state.targetRow}`
      );
      els.timelineInfo.textContent = payload.info || "";
    } else {
      if (!state.events.length) {
        els.plotTitle.textContent = "No plottable events";
        return;
      }
      readMargins();
      payload = await fetchJson(
        `/api/sessions/${state.sessionIndex}/events/${state.eventIndex}?${marginsQuery()}`
      );
      els.eventInfo.textContent = payload.info || "";
    }
  } catch (err) {
    if (token !== state.fetchToken) return;
    if (state.mode === "timeline") {
      els.timelineInfo.textContent = err.message;
    } else {
      els.eventInfo.textContent = err.message;
    }
    return;
  }
  if (token !== state.fetchToken) return;
  renderPlot(payload);
}

async function selectSession(index) {
  state.sessionIndex = index;
  state.eventIndex = 0;
  buildIndexBar(els.sessionBar, state.sessions.length, state.sessionIndex, selectSession);
  updateSessionMeta();
  await updateTimelineRange();
  await loadEvents();
  await refreshPlot();
}

function selectEvent(index) {
  if (index < 0 || index >= state.events.length) return;
  state.eventIndex = index;
  buildIndexBar(els.eventBar, state.events.length, state.eventIndex, selectEvent);
  refreshPlot();
}

async function applyMargins() {
  readMargins();
  if (!(state.preMs > 0 && state.postMs > 0)) {
    setMarginInfo("Pre and post must be positive", true);
    return;
  }
  if (state.mode === "timeline" && !(state.windowMs > 0)) {
    setMarginInfo("Window must be positive", true);
    return;
  }
  updateMarginInfo();
  if (state.mode === "timeline") {
    await updateTimelineRange();
  }
  await refreshPlot();
}

async function applyWindow() {
  readMargins();
  if (!(state.windowMs > 0)) {
    setMarginInfo("Window must be positive", true);
    return;
  }
  updateMarginInfo();
  await updateTimelineRange();
  await refreshPlot();
}

function setMode(mode) {
  state.mode = mode;
  const timeline = mode === "timeline";
  els.timelinePanel.classList.toggle("hidden", !timeline);
  els.eventsPanel.classList.toggle("hidden", timeline);
  refreshPlot();
}

let sliderTimer = null;
function onSliderInput() {
  state.targetRow = Number(els.timelineSlider.value);
  clearTimeout(sliderTimer);
  sliderTimer = setTimeout(refreshPlot, 40);
}

els.applyMargins.addEventListener("click", applyMargins);
els.preMs.addEventListener("change", applyMargins);
els.postMs.addEventListener("change", applyMargins);
els.windowMs.addEventListener("change", applyWindow);
els.timelineSlider.addEventListener("input", onSliderInput);
document.querySelectorAll('input[name="mode"]').forEach((input) => {
  input.addEventListener("change", () => setMode(currentMode()));
});
els.eventPrev.addEventListener("click", () => {
  if (!state.events.length) return;
  selectEvent((state.eventIndex - 1 + state.events.length) % state.events.length);
});
els.eventNext.addEventListener("click", () => {
  if (!state.events.length) return;
  selectEvent((state.eventIndex + 1) % state.events.length);
});
els.eventGo.addEventListener("click", () => {
  const index = Number(els.eventJump.value);
  if (Number.isFinite(index)) selectEvent(index);
});

window.addEventListener("resize", () => {
  if (state.lastPayload) renderPlot(state.lastPayload);
});

async function init() {
  ensurePlotCells();
  updateMarginInfo();
  await loadSessions();
  await updateTimelineRange();
  await loadEvents();
  await refreshPlot();
}

init().catch((err) => {
  els.sessionMeta.textContent = `Failed to load: ${err.message}`;
});
