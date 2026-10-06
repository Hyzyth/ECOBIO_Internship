"use strict";
const C = ReviewCore,
  $ = (id) => document.getElementById(id),
  colors = { coma: "#f57581", awake: "#78dac0", unknown: "#9cacc1" };
let project = {
    schema_version: 1,
    mode: "sequence",
    interval: 1,
    frames: [],
    wells: [],
    records: {},
    history: [],
  },
  files = [],
  urls = [],
  index = 0,
  selected = null,
  image = null,
  models = [],
  playing = null,
  draft = [],
  drag = null,
  busy = false,
  cancelled = false,
  loadGeneration = 0;
const canvas = $("canvas"),
  ctx = canvas.getContext("2d");
const status = (message) => ($("status").textContent = message);
const selectedWell = () => project.wells.find((w) => w.uid === selected);
const record = () => project.records[C.key(index, selected)] || {};
function history(action, details) {
  project.history.push({ at: new Date().toISOString(), action, ...details });
}
function stop() {
  clearInterval(playing);
  playing = null;
  $("play").textContent = "Play";
}
function pointsPath(context, points, w, h) {
  context.beginPath();
  points.forEach((p, i) =>
    i ? context.lineTo(p[0] * w, p[1] * h) : context.moveTo(p[0] * w, p[1] * h),
  );
  context.closePath();
}
function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!image) return;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  if ($("mask").checked && project.wells.length) {
    const layer = document.createElement("canvas");
    layer.width = canvas.width;
    layer.height = canvas.height;
    const lc = layer.getContext("2d");
    lc.fillStyle = "rgba(5,12,20,.8)";
    lc.fillRect(0, 0, layer.width, layer.height);
    lc.globalCompositeOperation = "destination-out";
    project.wells.forEach((w) => {
      pointsPath(lc, C.geometry(w, index), layer.width, layer.height);
      lc.fill();
    });
    ctx.drawImage(layer, 0, 0);
  }
  project.wells.forEach((w) => {
    const pts = C.geometry(w, index),
      state = C.effective(project.records[C.key(index, w.uid)]);
    pointsPath(ctx, pts, canvas.width, canvas.height);
    ctx.strokeStyle = colors[state];
    ctx.lineWidth = w.uid === selected ? 5 : 3;
    ctx.setLineDash(
      w.detection.review_required || w.review_frames?.includes(index)
        ? [8, 5]
        : [],
    );
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "bold 18px system-ui";
    ctx.fillStyle = "#07141c";
    const x = pts[0][0] * canvas.width,
      y = pts[0][1] * canvas.height;
    ctx.fillRect(x, y - 24, ctx.measureText(w.id).width + 14, 24);
    ctx.fillStyle = colors[state];
    ctx.fillText(w.id, x + 7, y - 6);
    if (w.uid === selected && $("tool").value === "vertex") {
      ctx.fillStyle = "#f9e7aa";
      pts.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p[0] * canvas.width, p[1] * canvas.height, 7, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  });
  const region =
    drag?.type === "region" ? drag.pending : project.detection_region;
  if (region) {
    ctx.strokeStyle = "#e8cb7d";
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 5]);
    ctx.strokeRect(
      region[0] * canvas.width,
      region[1] * canvas.height,
      (region[2] - region[0]) * canvas.width,
      (region[3] - region[1]) * canvas.height,
    );
    ctx.setLineDash([]);
  }
  if (draft.length) {
    ctx.beginPath();
    draft.forEach((p, i) =>
      i
        ? ctx.lineTo(p[0] * canvas.width, p[1] * canvas.height)
        : ctx.moveTo(p[0] * canvas.width, p[1] * canvas.height),
    );
    ctx.strokeStyle = "#f9e7aa";
    ctx.lineWidth = 3;
    ctx.stroke();
  }
}
async function renderFrame() {
  const generation = ++loadGeneration;
  image = null;
  draw();
  $("empty").hidden = !!files.length;
  $("timeline").max = Math.max(0, files.length - 1);
  $("timeline").value = index;
  $("jump").max = Math.max(1, files.length);
  $("jump").value = index + 1;
  $("frameCounter").textContent = files.length
    ? `${index + 1} / ${files.length}`
    : "0 / 0";
  const alignment = project.alignments?.[index];
  const alignmentFailure = project.alignment_failures?.[index];
  $("frameName").textContent =
    (files[index]?.name || "") +
    (alignment
      ? ` · ${alignment.method} alignment · ${(alignment.inlier_ratio * 100).toFixed(1)}% matched-feature inliers · ${alignment.median_error_px.toFixed(2)} px median feature error (resized image)`
      : "");
  const sequence = project.mode === "sequence";
  $("analyze").textContent = sequence ? "Analyze sequence" : "Analyze image";
  ["previous", "next", "play", "timeline", "speed", "jump"].forEach(
    (id) => ($(id).disabled = !sequence || !files.length || busy),
  );
  updateDetails();
  if (!files.length) {
    await previews();
    return;
  }
  const img = new Image();
  img.src = urls[index];
  try {
    await img.decode();
    if (generation !== loadGeneration) return;
    image = img;
    const scale = Math.min(1, 1400 / img.width);
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    draw();
    await previews();
  } catch {
    if (generation === loadGeneration)
      status(
        "Could not decode this image. Choose a PNG, JPEG, BMP or WebP image.",
      );
  }
}
async function previews() {
  const w = selectedWell(),
    current = index,
    generation = loadGeneration;
  await Promise.all(
    ["prevCrop", "crop", "nextCrop"].map(async (id, j) => {
      const c = $(id),
        context = c.getContext("2d");
      context.clearRect(0, 0, c.width, c.height);
      const f = current + j - 1;
      if (
        !w ||
        f < 0 ||
        f >= files.length ||
        (project.mode === "single" && j !== 1)
      ) {
        context.fillStyle = "#8fa8b7";
        context.fillText("No temporal context", 20, 90);
        return;
      }
      const img = new Image();
      img.src = urls[f];
      try {
        await img.decode();
        if (generation !== loadGeneration || selected !== w.uid) return;
        const points = C.geometry(w, f);
        const xs = points.map((p) => p[0] * img.width),
          ys = points.map((p) => p[1] * img.height);
        const x = Math.min(...xs),
          y = Math.min(...ys),
          width = Math.max(...xs) - x,
          height = Math.max(...ys) - y;
        if (width <= 0 || height <= 0) return;
        const scale = Math.min(c.width / width, c.height / height),
          ox = (c.width - width * scale) / 2,
          oy = (c.height - height * scale) / 2;
        context.save();
        context.beginPath();
        points.forEach((p, k) => {
          const px = ox + (p[0] * img.width - x) * scale,
            py = oy + (p[1] * img.height - y) * scale;
          k ? context.lineTo(px, py) : context.moveTo(px, py);
        });
        context.closePath();
        context.clip();
        context.drawImage(
          img,
          x,
          y,
          width,
          height,
          ox,
          oy,
          width * scale,
          height * scale,
        );
        context.restore();
      } catch {
        context.fillStyle = "#8fa8b7";
        context.fillText("Unreadable image", 20, 90);
      }
    }),
  );
}
function updateDetails() {
  const w = selectedWell(),
    r = record();
  for (const id of ["wellSelect", "proposalSelect"]) {
    $(id).replaceChildren(
      new Option("Select a well", ""),
      ...project.wells.map((w) => new Option(w.id, w.uid)),
    );
    $(id).value = selected || "";
  }
  [
    "wellName",
    "occupancy",
    "confirm",
    "delete",
    "removeSelected",
    "correction",
    "eggCount",
    "notes",
    "saveCorrection",
    "clearCorrection",
  ].forEach((id) => ($(id).disabled = !w || busy));
  $("wellName").value = w?.id || "";
  $("occupancy").value = w?.occupancy || "unknown";
  $("correction").value = r.correction?.state || "";
  $("eggCount").value = r.correction?.egg_count ?? "";
  $("notes").value = r.correction?.notes || "";
  $("prediction").textContent = w
    ? `${w.id} · ${C.effective(r).toUpperCase()} · Model: ${r.model?.name || "No prediction"} ${r.model?.version || ""} · Confidence: ${r.confidence == null ? "Not available" : Math.round(r.confidence * 100) + "%"}${r.correction ? " · Manual annotation applied" : ""} · Eggs: ${r.correction?.egg_count ?? r.egg_count ?? "Not counted"}${w.detection.review_required || w.review_frames?.includes(index) ? " · Boundary needs review" : ""}${(w.partial_frames?.[index] ?? w.detection.partial) ? " · Partially visible well" : ""}`
    : "No well selected.";
  renderSummary();
}
function renderSummary() {
  const body = $("summary");
  body.replaceChildren();
  C.summary(project).forEach((s) => {
    const tr = document.createElement("tr");
    [
      s.well_id,
      s.total_frames,
      project.mode === "sequence"
        ? `${s.awake_frames} frames / ${s.awake_seconds.toFixed(2)} s`
        : s.awake_frames,
      project.mode === "sequence"
        ? `${s.coma_frames} frames / ${s.coma_seconds.toFixed(2)} s`
        : s.coma_frames,
      s.unknown_frames,
      s.transitions ?? "—",
      s.average_model_confidence == null
        ? "—"
        : `${Math.round(s.average_model_confidence * 100)}%`,
    ].forEach((v) => {
      const td = document.createElement("td");
      td.textContent = v;
      tr.append(td);
    });
    const td = document.createElement("td"),
      spark = document.createElement("div");
    spark.className = "spark";
    s.states.forEach((state, i) => {
      const span = document.createElement("span");
      span.style.background = colors[state];
      span.title = `Frame ${i + 1}: ${state}`;
      spark.append(span);
    });
    td.append(spark);
    tr.append(td);
    body.append(tr);
  });
}
function select(uid) {
  selected = uid;
  draft = [];
  updateDetails();
  draw();
  previews();
}
function addWell(
  points,
  detection = { source: "manual", review_required: false, confidence: null },
) {
  if (!C.isSimplePolygon(points))
    throw Error(
      "Boundary must be a simple polygon with nonzero area; edges cannot cross or retrace.",
    );
  const uid = crypto.randomUUID();
  let n = 1;
  while (project.wells.some((w) => w.id === `W${n}`)) n++;
  project.wells.push({
    uid,
    id: `W${n}`,
    points,
    overrides: {},
    detection,
    occupancy: "unknown",
    review_frames: [],
  });
  history("add-well", { uid, frame: index });
  select(uid);
}
function frameMetadata(file) {
  return {
    id: `${file.webkitRelativePath || file.name}|${file.size}|${file.lastModified}`,
    name: file.webkitRelativePath || file.name,
    timestamp: C.timestamp(file.name),
    size: file.size,
    last_modified: file.lastModified,
  };
}
async function loadFiles(input) {
  if (busy) return;
  if (
    project.wells.length &&
    !confirm(
      "Loading new images clears this review. Export JSON first to keep your work. Continue?",
    )
  )
    return;
  const list = [...input].filter((f) =>
    /\.(png|jpe?g|bmp|webp)$/i.test(f.name),
  );
  if (!list.length) {
    status("No supported images found. Select PNG, JPEG, BMP or WebP images.");
    return;
  }
  stop();
  urls.forEach(URL.revokeObjectURL);
  files = C.orderFiles(list, $("order").value);
  if ($("mode").value === "single" && files.length > 1) {
    files = files.slice(0, 1);
    status(
      "Single-image mode loaded the first ordered file. Choose an individual image to change it.",
    );
  } else
    status(
      `${files.length} images loaded. Confirm order, interval and well boundaries.`,
    );
  urls = files.map((f) => URL.createObjectURL(f));
  project = {
    schema_version: 1,
    mode: $("mode").value,
    interval: Number($("interval").value) || 1,
    order: $("order").value,
    frames: files.map(frameMetadata),
    wells: [],
    records: {},
    history: [],
  };
  index = 0;
  selected = null;
  $("frameList").replaceChildren(
    ...project.frames.map((f, i) => {
      const li = document.createElement("li");
      li.textContent = `${i + 1}. ${f.name}`;
      return li;
    }),
  );
  $("inputInfo").textContent =
    `${files.length} image${files.length === 1 ? "" : "s"} selected. No image data is stored on the server.`;
  await renderFrame();
  if ($("layout").value === "individual") await detect();
}
function moveTo(i) {
  if (busy || !files.length) return;
  index = Math.max(0, Math.min(files.length - 1, i));
  draft = [];
  renderFrame();
}
function setBusy(value) {
  busy = value;
  [
    "browse",
    "filesButton",
    "mode",
    "order",
    "layout",
    "detect",
    "detectionMethod",
    "diameter",
    "proposalSelect",
    "align",
    "alignSequence",
    "clearRegion",
    "confirmFrame",
    "whole",
    "tool",
    "scope",
    "model",
    "analyze",
    "importButton",
    "interval",
    "wellSelect",
    "mask",
  ].forEach((id) => ($(id).disabled = value));
  $("cancel").hidden = !value;
  $("progress").hidden = !value;
  updateDetails();
}
async function post(url, form) {
  const response = await fetch(url, { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) throw Error(data.error || "Request failed.");
  return data;
}
async function detect() {
  if (!files.length) return status("Load an image first.");
  if (
    project.wells.length &&
    !confirm(
      "Replace all well boundaries and annotations with new proposals? Export first to preserve them.",
    )
  )
    return;
  stop();
  setBusy(true);
  try {
    const form = new FormData();
    form.append("image", files[index]);
    form.append("method", $("detectionMethod").value);
    form.append(
      "layout",
      $("layout").value === "individual" ? "individual" : "grid",
    );
    form.append("diameter", $("diameter").value);
    if (project.detection_region)
      form.append("region", JSON.stringify(project.detection_region));
    const data = await post("/api/detect", form);
    if (!data.wells.length) {
      status(
        "No matching wells found. Existing wells are preserved. Try a diameter override, another detection method, or draw boundaries manually.",
      );
      return;
    }
    project.wells = [];
    project.records = {};
    project.detection_settings = data.settings;
    selected = null;
    data.wells.forEach((w) =>
      addWell(w.points, {
        source: w.source,
        confidence: w.confidence,
        ring_coverage: w.ring_coverage ?? null,
        partial: w.partial ?? false,
        review_required: true,
      }),
    );
    status(`${data.wells.length} proposals. ${data.warning}`);
  } catch (e) {
    status(e.message);
  } finally {
    setBusy(false);
    renderFrame();
  }
}
function invalidate(uid, frame) {
  const frames = frame == null ? project.frames.map((_, i) => i) : [frame];
  for (const f of frames) {
    const k = C.key(f, uid),
      r = project.records[k];
    if (r) {
      history("invalidate-prediction", { frame: f, uid, previous: r });
      project.records[k] = r.correction ? { correction: r.correction } : {};
    }
  }
}
function saveGeometry(w, points) {
  if ($("scope").value === "frame") {
    w.overrides[index] = points;
    invalidate(w.uid, index);
  } else {
    w.points = points;
    w.overrides = {};
    invalidate(w.uid, null);
  }
  if ($("scope").value === "frame")
    w.review_frames = [...new Set([...(w.review_frames || []), index])];
  else {
    w.detection.review_required = true;
    w.review_frames = [];
  }
  history("edit-boundary", {
    uid: w.uid,
    frame: index,
    scope: $("scope").value,
    points,
  });
  updateDetails();
  previews();
}
function position(event) {
  const b = canvas.getBoundingClientRect();
  return [
    Math.max(0, Math.min(1, (event.clientX - b.left) / b.width)),
    Math.max(0, Math.min(1, (event.clientY - b.top) / b.height)),
  ];
}
canvas.addEventListener("pointerdown", (e) => {
  if (!image || busy) return;
  stop();
  const p = position(e),
    tool = $("tool").value;
  if (tool === "polygon") return;
  if (tool === "region") {
    drag = { type: "region", start: p, pending: null };
    canvas.setPointerCapture(e.pointerId);
    return;
  }
  if (tool === "vertex" && selectedWell()) {
    const w = selectedWell(),
      points = C.geometry(w, index).map((p) => [...p]);
    let nearest = -1,
      distance = Infinity;
    points.forEach((pt, i) => {
      const d = Math.hypot(
        (pt[0] - p[0]) * canvas.clientWidth,
        (pt[1] - p[1]) * canvas.clientHeight,
      );
      if (d < distance) {
        distance = d;
        nearest = i;
      }
    });
    if (distance < 20) drag = { well: w, vertex: nearest, points, start: p };
  } else {
    const w = [...project.wells].reverse().find((w) => {
      pointsPath(ctx, C.geometry(w, index), canvas.width, canvas.height);
      return ctx.isPointInPath(p[0] * canvas.width, p[1] * canvas.height);
    });
    select(w?.uid || null);
    if (tool === "remove") {
      if (w) removeSelectedWell();
      return;
    }
    if (w)
      drag = {
        well: w,
        points: C.geometry(w, index).map((p) => [...p]),
        start: p,
      };
  }
  if (drag) canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const p = position(e);
  if (drag.type === "region") {
    drag.pending = [
      Math.min(p[0], drag.start[0]),
      Math.min(p[1], drag.start[1]),
      Math.max(p[0], drag.start[0]),
      Math.max(p[1], drag.start[1]),
    ];
    draw();
    return;
  }
  let points;
  if (drag.vertex != null) {
    points = drag.points.map((pt) => [...pt]);
    points[drag.vertex] = p;
  } else {
    const xs = drag.points.map((p) => p[0]),
      ys = drag.points.map((p) => p[1]);
    const dx = Math.max(
        -Math.min(...xs),
        Math.min(1 - Math.max(...xs), p[0] - drag.start[0]),
      ),
      dy = Math.max(
        -Math.min(...ys),
        Math.min(1 - Math.max(...ys), p[1] - drag.start[1]),
      );
    points = drag.points.map((pt) => [pt[0] + dx, pt[1] + dy]);
  }
  if (!C.isSimplePolygon(points)) {
    status(
      "Boundary edges cannot cross or retrace. Move the vertex to a valid position.",
    );
    return;
  }
  drag.pending = points;
  const w = drag.well;
  if ($("scope").value === "frame") w.overrides[index] = points;
  else w.points = points;
  draw();
});
function endDrag() {
  if (drag?.type === "region") {
    if (
      drag.pending &&
      drag.pending[2] - drag.pending[0] > 0.01 &&
      drag.pending[3] - drag.pending[1] > 0.01
    ) {
      project.detection_region = drag.pending;
      history("set-detection-region", { region: drag.pending });
      status(
        "Detection region saved. Only complete boundary proposals inside it will be considered.",
      );
    }
    drag = null;
    draw();
    return;
  }
  if (drag?.pending) saveGeometry(drag.well, drag.pending);
  drag = null;
  draw();
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("click", (e) => {
  if (!image || busy || $("tool").value !== "polygon" || e.detail !== 1) return;
  const p = position(e);
  const same = (q) =>
    q &&
    Math.hypot(
      (q[0] - p[0]) * canvas.clientWidth,
      (q[1] - p[1]) * canvas.clientHeight,
    ) < 1;
  // Click events carry click counts; pointerdown does not. Do not append the
  // second half of a double-click or a repeated closing vertex.
  if (!same(draft.at(-1)) && !(draft.length >= 3 && same(draft[0])))
    draft.push(p);
  draw();
});
canvas.addEventListener("dblclick", () => {
  if (busy || $("tool").value !== "polygon" || draft.length < 3) return;
  const xs = draft.map((p) => p[0]),
    ys = draft.map((p) => p[1]);
  if (
    Math.max(...xs) - Math.min(...xs) < 0.002 ||
    Math.max(...ys) - Math.min(...ys) < 0.002
  )
    return status("Draw a boundary with a nonzero area.");
  if (!C.isSimplePolygon(draft))
    return status(
      "Boundary edges cannot cross or retrace. Press Escape to cancel and draw around the perimeter in order.",
    );
  addWell(draft.map((p) => [...p]));
  draft = [];
  draw();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    draft = [];
    draw();
  }
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  if ((e.key === "Delete" || e.key === "Backspace") && selected && !busy) {
    e.preventDefault();
    removeSelectedWell();
  }
  if (e.key === "ArrowRight") moveTo(index + 1);
  if (e.key === "ArrowLeft") moveTo(index - 1);
});
$("browse").onclick = () => $("folder").click();
$("filesButton").onclick = () => $("files").click();
$("folder").onchange = (e) => {
  loadFiles(e.target.files);
  e.target.value = "";
};
$("files").onchange = (e) => {
  loadFiles(e.target.files);
  e.target.value = "";
};
$("mode").onchange = () => {
  if (
    files.length &&
    !confirm("Changing mode clears this review. Export JSON first. Continue?")
  ) {
    $("mode").value = project.mode;
    return;
  }
  stop();
  urls.forEach(URL.revokeObjectURL);
  files = [];
  urls = [];
  index = 0;
  selected = null;
  draft = [];
  project = {
    schema_version: 1,
    mode: $("mode").value,
    interval: Number($("interval").value) || 1,
    order: $("order").value,
    frames: [],
    wells: [],
    records: {},
    history: [],
  };
  $("frameList").replaceChildren();
  $("inputInfo").textContent = "Choose new images for this analysis mode.";
  status("Mode changed. Choose a folder or image files to begin.");
  renderFrame();
};
$("order").onchange = () => {
  if (files.length) {
    $("order").value = project.order || "name";
    status(
      "Choose ordering before loading images. Reload to reorder without misaligning annotations.",
    );
  }
};
$("interval").onchange = () => {
  const v = Number($("interval").value);
  if (!Number.isFinite(v) || v <= 0) {
    $("interval").value = project.interval;
    return;
  }
  project.interval = v;
  history("set-interval", { seconds: v });
  renderSummary();
};
$("previous").onclick = () => {
  stop();
  moveTo(index - 1);
};
$("next").onclick = () => {
  stop();
  moveTo(index + 1);
};
$("timeline").oninput = (e) => {
  stop();
  moveTo(Number(e.target.value));
};
$("jump").onchange = (e) => {
  stop();
  moveTo(Number(e.target.value) - 1);
};
function play() {
  if (playing) return stop();
  if (project.mode !== "sequence" || !files.length) return;
  const speed = Math.max(0.1, Math.min(30, Number($("speed").value) || 2));
  $("speed").value = speed;
  $("play").textContent = "Pause";
  playing = setInterval(() => {
    if (index >= files.length - 1) {
      stop();
      return;
    }
    moveTo(index + 1);
  }, 1000 / speed);
}
$("play").onclick = play;
$("speed").onchange = () => {
  if (playing) {
    stop();
    play();
  }
};
$("wellSelect").onchange = (e) => select(e.target.value);
$("mask").onchange = draw;
$("tool").onchange = () => {
  draft = [];
  draw();
};
$("whole").onclick = () => {
  if (files.length)
    addWell([
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ]);
  else status("Load an image first.");
};
$("detect").onclick = detect;
$("wellName").onchange = () => {
  const w = selectedWell(),
    name = $("wellName").value.trim();
  if (!w) return;
  if (
    !name ||
    project.wells.some((other) => other.uid !== w.uid && other.id === name)
  ) {
    status("Well identities must be nonempty and unique.");
    updateDetails();
    return;
  }
  history("rename-well", { uid: w.uid, previous: w.id, next: name });
  w.id = name;
  updateDetails();
  draw();
};
$("occupancy").onchange = () => {
  const w = selectedWell();
  if (w) {
    w.occupancy = $("occupancy").value;
    history("set-occupancy", { uid: w.uid, occupancy: w.occupancy });
  }
};
$("confirm").onclick = () => {
  const w = selectedWell();
  if (w) {
    w.detection.review_required = false;
    w.review_frames = (w.review_frames || []).filter(
      (frame) => frame !== index,
    );
    history("confirm-boundary", { uid: w.uid, frame: index });
    updateDetails();
    draw();
  }
};
function removeSelectedWell() {
  const w = selectedWell();
  if (!w || !confirm(`Delete ${w.id} and its annotations?`)) return;
  history("delete-well", {
    well: w,
    records: Object.fromEntries(
      Object.entries(project.records).filter(([k]) => k.endsWith(":" + w.uid)),
    ),
  });
  project.wells = project.wells.filter((other) => other.uid !== w.uid);
  Object.keys(project.records).forEach((k) => {
    if (k.endsWith(":" + w.uid)) delete project.records[k];
  });
  select(null);
  status(
    `${w.id} removed from all frames. Other well identities and annotations are preserved.`,
  );
}
$("delete").onclick = removeSelectedWell;
$("removeSelected").onclick = removeSelectedWell;
$("proposalSelect").onchange = (e) => select(e.target.value);
$("saveCorrection").onclick = () => {
  if (!selected) return;
  const value = $("eggCount").value,
    count = value === "" ? null : Number(value);
  if (count != null && (!Number.isInteger(count) || count < 0))
    return status("Egg count must be a nonnegative whole number.");
  const correction = {
    state: $("correction").value || null,
    egg_count: count,
    notes: $("notes").value,
    at: new Date().toISOString(),
  };
  const k = C.key(index, selected);
  history("manual-annotation", {
    frame: index,
    uid: selected,
    previous: project.records[k]?.correction || null,
    next: correction,
  });
  project.records[k] = { ...record(), correction };
  updateDetails();
  draw();
  status("Manual annotation saved in this project. Export JSON to keep it.");
};
$("clearCorrection").onclick = () => {
  if (!selected) return;
  const k = C.key(index, selected);
  history("clear-annotation", {
    frame: index,
    uid: selected,
    previous: record().correction,
  });
  const r = { ...record() };
  delete r.correction;
  project.records[k] = r;
  updateDetails();
  draw();
};
$("analyze").onclick = async () => {
  if ($("model").value === "manual")
    return status(
      "Manual review has no automatic predictions. Draw or detect wells, then annotate each frame.",
    );
  if (!files.length || !project.wells.length)
    return status("Load images and select wells first.");
  if (
    project.wells.some(
      (w) => w.detection.review_required || w.review_frames?.length,
    )
  )
    return status(
      "Review and confirm all proposed or edited well boundaries before model analysis.",
    );
  stop();
  cancelled = false;
  setBusy(true);
  $("progress").max = files.length;
  $("progress").value = 0;
  try {
    for (let i = 0; i < files.length; i++) {
      if (cancelled) break;
      const form = new FormData();
      form.append("image", files[i]);
      form.append("model", $("model").value);
      form.append(
        "wells",
        JSON.stringify(
          project.wells.map((w) => ({
            id: w.uid,
            points: C.geometry(w, i),
            occupancy: w.occupancy,
          })),
        ),
      );
      const data = await post("/api/analyze", form);
      if (cancelled) break;
      data.records.forEach((r) => {
        const k = C.key(i, r.well_id);
        history("model-prediction", {
          frame: i,
          uid: r.well_id,
          previous: project.records[k] || null,
        });
        project.records[k] = {
          ...project.records[k],
          prediction: r.prediction,
          confidence: r.confidence,
          model: data.model,
          egg_count: r.egg_count ?? null,
        };
      });
      $("progress").value = i + 1;
    }
    status(
      cancelled
        ? "Analysis cancelled. Completed frames are retained."
        : "Analysis complete. Manual annotations remain separate.",
    );
  } catch (e) {
    status(`Analysis stopped: ${e.message}. Completed frames are retained.`);
  } finally {
    setBusy(false);
    renderFrame();
  }
};
$("cancel").onclick = () => {
  cancelled = true;
  status("Cancelling after the current request…");
};
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type })),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("json").onclick = () =>
  download(
    "flyscope-project.json",
    JSON.stringify(C.exportProject(project), null, 2),
    "application/json",
  );
$("csv").onclick = () =>
  download("flyscope-results.csv", C.csv(project), "text/csv");
$("importButton").onclick = () => $("import").click();
$("import").onchange = async (e) => {
  try {
    const file = e.target.files[0];
    if (!file) return;
    const loaded = C.validateProject(
      JSON.parse(await file.text()),
      project.frames,
    );
    if (
      project.wells.length &&
      !confirm("Replace this review with the saved project?")
    )
      return;
    stop();
    project = {
      schema_version: 1,
      mode: loaded.mode,
      interval: loaded.interval,
      order: loaded.order,
      wells: loaded.wells,
      frames: loaded.frames,
      records: loaded.records,
      history: loaded.history || [],
      detection_region: loaded.detection_region || null,
      detection_settings: loaded.detection_settings || null,
      alignments: loaded.alignments || {},
      alignment_failures: loaded.alignment_failures || {},
    };
    $("mode").value = project.mode;
    $("interval").value = project.interval;
    $("order").value = project.order || "name";
    selected = project.wells[0]?.uid || null;
    status(
      "Project restored. Images matched by path, size and modification time.",
    );
    renderFrame();
  } catch (error) {
    status(error.message);
  } finally {
    e.target.value = "";
  }
};
window.addEventListener("beforeunload", (e) => {
  if (project.wells.length) {
    e.preventDefault();
    e.returnValue = "";
  }
});
(async () => {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw Error("Model registry unavailable.");
    models = await response.json();
    $("model").replaceChildren(
      ...models.map((m) => new Option(`${m.name} · ${m.version}`, m.id)),
    );
    $("model").onchange = () => {
      const m = models.find((m) => m.id === $("model").value);
      $("modelInfo").textContent = m?.description || "";
    };
    $("model").onchange();
  } catch (e) {
    $("model").replaceChildren(new Option("Manual review · 1", "manual"));
    $("modelInfo").textContent =
      "Model registry unavailable. Manual annotation still works.";
    status(e.message);
  }
  renderFrame();
})();

async function alignFrames(entireSequence) {
  if (
    project.mode !== "sequence" ||
    !project.wells.length ||
    files.length < 2 ||
    (!entireSequence && index === 0)
  )
    return status(
      "Define wells on the first frame, then select a later sequence frame or align the entire sequence.",
    );
  const targets = entireSequence ? files.map((_, i) => i).slice(1) : [index];
  const baseWells = project.wells.map((w) => ({
    id: w.uid,
    points: C.geometry(w, 0),
  }));
  stop();
  cancelled = false;
  setBusy(true);
  $("progress").max = targets.length;
  $("progress").value = 0;
  const failed = [];
  let completed = 0;
  try {
    for (const frame of targets) {
      if (cancelled) break;
      try {
        const form = new FormData();
        form.append("reference", files[0]);
        form.append("image", files[frame]);
        form.append("wells", JSON.stringify(baseWells));
        const data = await post("/api/align", form);
        if (cancelled) break;
        for (const item of data.wells) {
          const w = project.wells.find((w) => w.uid === item.id);
          w.overrides[frame] = item.points;
          w.partial_frames ??= {};
          w.partial_frames[frame] = !!item.partial || !!w.detection.partial;
          w.review_frames = [...new Set([...(w.review_frames || []), frame])];
          invalidate(w.uid, frame);
        }
        const metrics = {
          method: data.method,
          inlier_ratio: data.inlier_ratio,
          median_error_px: data.median_error_px,
          p95_error_px: data.p95_error_px,
          reference_frame: 0,
          working_image_size: data.working_image_size,
        };
        project.alignments ??= {};
        project.alignments[frame] = metrics;
        delete project.alignment_failures?.[frame];
        history("align-frame", { frame, metrics, wells: data.wells });
        completed++;
      } catch (error) {
        failed.push({ frame, error: error.message });
        project.alignment_failures ??= {};
        project.alignment_failures[frame] = {
          error: error.message,
          reference_frame: 0,
        };
        delete project.alignments?.[frame];
        project.wells.forEach((w) => {
          w.review_frames = [...new Set([...(w.review_frames || []), frame])];
          invalidate(w.uid, frame);
        });
        history("alignment-failed", { frame, error: error.message });
      }
      $("progress").value++;
    }
    status(
      `${cancelled ? "Alignment cancelled" : "Alignment finished"}: ${completed} frames aligned from the first-frame template; ${failed.length} failures.${failed.length ? " Review failed frames: " + failed.map((f) => `${f.frame + 1} (${f.error})`).join("; ") : ""} Inspect each frame and confirm its proposed boundaries. Identities were retained.`,
    );
  } finally {
    setBusy(false);
    renderFrame();
  }
}
$("align").onclick = () => alignFrames(false);
$("alignSequence").onclick = () => alignFrames(true);
$("clearRegion").onclick = () => {
  project.detection_region = null;
  history("clear-detection-region", {});
  draw();
};
$("confirmFrame").onclick = () => {
  if (!project.wells.length) return status("Select wells first.");
  if (
    !confirm(
      "Confirm the visible boundaries on this frame? Confirming the shared template also applies to frames without overrides.",
    )
  )
    return;
  project.wells.forEach((w) => {
    w.detection.review_required = false;
    w.review_frames = (w.review_frames || []).filter((f) => f !== index);
  });
  history("confirm-frame-boundaries", { frame: index });
  updateDetails();
  draw();
  status(
    `Frame ${index + 1} boundaries confirmed. Other alignment proposals still require review.`,
  );
};
