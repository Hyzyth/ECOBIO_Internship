"use strict";
const C = ReviewCore,
  $ = (id) => document.getElementById(id),
  colors = {
    coma: "#ff304f",
    awake: "#00ec57",
    unknown: "#9cacc1",
    unsure: "#ffd34d",
    empty: "#00cfff",
    invalid: "#dc70ff",
  },
  study = document.body.dataset.study || "ccrt";
let lastLabelOperation = null,
  workflowStage = "images";
let project = {
    schema_version: 1,
    time_origin: { confirmed: true, offset_seconds: 0 },
    task: study,
    experiment_id: crypto.randomUUID(),
    mode: study === "eggs" ? "batch" : "sequence",
    interval: 1,
    layout: "grid",
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
const visibleWells = () =>
  index < C.analysisStart(project)
    ? []
    : project.wells.filter((w) => C.activeWell(w, index));
const selectedWell = () => visibleWells().find((w) => w.uid === selected);
const record = () => project.records[C.key(index, selected)] || {};
function selectedRecoveryPolicy() {
  const modelPolicy = models.find(
    (m) => m.id === $("model").value,
  )?.recovery_policy;
  if ($("useModelPolicy").checked && modelPolicy)
    return structuredClone(modelPolicy);
  return {
    confidence_threshold: Number($("decisionThreshold").value),
    awake_consecutive_frames: Number($("awakeFrames").value),
    require_coma_before_awake: $("requireComaBaseline").checked,
    automatic_confirmation: true,
    source: "manual_override",
    infer_matching_gaps: $("inferMatchingGaps").checked,
  };
}
function updatePolicyControls() {
  const policy = project.recovery_policy || selectedRecoveryPolicy();
  $("decisionThreshold").value = policy.confidence_threshold;
  $("awakeFrames").value = policy.awake_consecutive_frames;
  $("requireComaBaseline").checked = policy.require_coma_before_awake ?? false;
  $("inferMatchingGaps").checked = policy.infer_matching_gaps ?? false;
  $("manualPolicyOptions").hidden = $("useModelPolicy").checked;
  if (policy.source === "validation") {
    $("policyInfo").textContent =
      policy.automatic_confirmation === false
        ? "Automatic recovery could not be validated. Awake decisions stay Unsure; inspect raw predictions and annotate manually. " +
          (policy.selection?.message ||
            "No candidate met the validation criteria.")
        : `Selected on validation individuals: ${Math.round(policy.confidence_threshold * 100)}% confidence; ${policy.awake_confirmation_seconds != null ? policy.awake_confirmation_seconds + " seconds of sustained evidence" : policy.awake_consecutive_frames + " consecutive observations"}. Test grids were not used to choose these settings.`;
  } else {
    $("policyInfo").textContent = $("useModelPolicy").checked
      ? "Stored model settings. This older model has no automatic validation selection; retrain with the updated workflow to obtain it."
      : "Manual recovery settings override the model’s recommendations.";
  }
}
function history(action, details) {
  project.history.push({ at: new Date().toISOString(), action, ...details });
}
function stop() {
  clearInterval(playing);
  playing = null;
  $("play").textContent = "Play";
}
function workflowSteps() {
  return study === "eggs" && $("eggScope").value === "image"
    ? ["images", "review", "export"]
    : ["images", "wells", "review", "export"];
}
function canEnterStep(step) {
  return (
    workflowSteps().includes(step) &&
    (step === "images" ||
      (files.length > 0 && (step === "wells" || project.wells.length > 0)))
  );
}
function goToStep(step) {
  if (busy || !canEnterStep(step)) return;
  stop();
  endDrag();
  draft = [];
  workflowStage = step;
  if (step !== "images" && index < C.analysisStart(project))
    moveTo(C.analysisStart(project));
  // Selecting a well during review must not accidentally move its boundary.
  if (step !== "wells") $("tool").value = "select";
  updateContext();
  draw();
  $(
    {
      images: "inputHeading",
      wells: "wellsHeading",
      review: "reviewHeading",
      export: "exportHeading",
    }[step],
  ).focus({ preventScroll: true });
}
function updateWorkflow() {
  const steps = workflowSteps(),
    labels = {
      images: "Images",
      wells: study === "ccrt" ? "Wells" : "Areas",
      review: study === "ccrt" ? "Review" : "Count",
      export: "Export",
    };
  if (!canEnterStep(workflowStage))
    workflowStage =
      files.length && steps.includes("wells") ? "wells" : "images";
  document.body.dataset.workflow = workflowStage;
  document.querySelectorAll("[data-step]").forEach((button) => {
    const step = button.dataset.step;
    button.hidden = !steps.includes(step);
    button.disabled = busy || !canEnterStep(step);
    button.textContent = `${steps.indexOf(step) + 1} · ${labels[step]}`;
    if (step === workflowStage) button.setAttribute("aria-current", "step");
    else button.removeAttribute("aria-current");
  });
  const next = steps[steps.indexOf(workflowStage) + 1],
    previous = steps[steps.indexOf(workflowStage) - 1];
  document.querySelectorAll("[data-next-step]").forEach((button) => {
    button.disabled = busy || !canEnterStep(next);
    button.textContent = `Continue to ${labels[next]?.toLowerCase() || "export"} →`;
  });
  $("workflowBack").hidden = !previous;
  $("workflowBack").disabled = busy;
  $("workflowBack").textContent =
    `← Back to ${labels[previous]?.toLowerCase() || "images"}`;
  document.querySelectorAll("[data-go-step]").forEach((button) => {
    button.disabled = busy || !canEnterStep(button.dataset.goStep);
  });
  $("workflowHint").textContent = {
    images: !files.length
      ? "Start by choosing your images. Later steps become available as you go."
      : project.mode === "sequence" && files.length > 1
        ? "Check image order and frame interval, then continue. You can also resume a saved review."
        : "Check the selected images, then continue. You can also resume a saved review.",
    wells: !project.wells.length
      ? "Suggest or draw boundaries before continuing. Nothing is classified at this step."
      : "Check the boundaries and approve stable positions, then continue. You can return here at any time.",
    review:
      study === "ccrt"
        ? "Check occupancy, then label states. Model predictions are optional; partial annotation is supported."
        : "Count each independent image or area. No sequence timing or model selection is required.",
    export:
      "Choose a readable report or a backup. Dataset and machine exports are optional.",
  }[workflowStage];
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
  if (
    $("mask").checked &&
    visibleWells().length &&
    workflowStage !== "images"
  ) {
    const layer = document.createElement("canvas");
    layer.width = canvas.width;
    layer.height = canvas.height;
    const lc = layer.getContext("2d");
    lc.fillStyle = "rgba(5,12,20,.8)";
    lc.fillRect(0, 0, layer.width, layer.height);
    lc.globalCompositeOperation = "destination-out";
    visibleWells().forEach((w) => {
      pointsPath(lc, C.geometry(w, index), layer.width, layer.height);
      lc.fill();
    });
    ctx.drawImage(layer, 0, 0);
  }
  visibleWells().forEach((w) => {
    if (study === "eggs" && $("eggScope").value === "image") return;
    const pts = C.geometry(w, index),
      state = C.effective(
        project.records[C.key(index, w.uid)],
        study === "ccrt" ? w : null,
      );
    pointsPath(ctx, pts, canvas.width, canvas.height);
    ctx.strokeStyle = "#07141c";
    ctx.lineWidth = w.uid === selected ? 8 : 6;
    ctx.stroke();
    ctx.strokeStyle = study === "eggs" ? "#ffe000" : colors[state];
    ctx.lineWidth = w.uid === selected ? 5 : 3;
    ctx.setLineDash(
      state === "empty"
        ? [3, 4]
        : ["unknown", "unsure"].includes(state)
          ? [10, 6]
          : w.detection.review_required || w.review_frames?.includes(index)
            ? [8, 5]
            : [],
    );
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "bold 18px system-ui";
    ctx.fillStyle = "#07141c";
    const x = pts[0][0] * canvas.width,
      y = pts[0][1] * canvas.height;
    ctx.fillRect(x, y - 24, ctx.measureText(w.id + " ?").width + 14, 24);
    ctx.fillStyle = colors[state];
    ctx.fillText(
      w.id +
        (study === "ccrt"
          ? " " +
            (state === "coma"
              ? "C"
              : state === "awake"
                ? "A"
                : state === "empty"
                  ? "E"
                  : state === "invalid"
                    ? "!"
                    : state === "unsure"
                      ? "U"
                      : "?")
          : ""),
      x + 7,
      y - 6,
    );
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
  if (study === "eggs")
    project.detection_region = project.detection_regions?.[index] || null;
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
    (alignment ? ` · Camera alignment applied` : "");
  if (!selectedWell()) selected = visibleWells()[0]?.uid || null;
  updateContext();
  const sequence = project.mode === "sequence";

  ["previous", "next", "timeline", "jump"].forEach(
    (id) => ($(id).disabled = files.length < 2 || busy),
  );
  ["play", "speed"].forEach(
    (id) => ($(id).disabled = !sequence || files.length < 2 || busy),
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
        f < C.analysisStart(project) ||
        f >= files.length ||
        (project.mode !== "sequence" && j !== 1)
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
      ...visibleWells().map((w) => new Option(w.id, w.uid)),
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
    "labelComa",
    "labelAwake",
    "labelUncertain",
    "comaUntil",
    "awakeFrom",
    "applyRange",
    "nextBoundary",
    "nextLabel",
  ].forEach((id) => ($(id).disabled = !w || busy));
  $("wellName").value = w?.id || "";
  if (study === "ccrt" && ["empty", "multiple"].includes(w?.occupancy))
    for (const id of [
      "correction",
      "labelComa",
      "labelAwake",
      "labelUncertain",
      "comaUntil",
      "awakeFrom",
      "applyRange",
    ])
      $(id).disabled = true;
  $("occupancy").value = w?.occupancy || "unknown";
  $("correction").value =
    study === "ccrt" && ["empty", "multiple"].includes(w?.occupancy)
      ? w.occupancy === "empty"
        ? "empty"
        : "invalid"
      : r.correction?.state || "";
  $("eggCount").value = r.correction?.egg_count ?? "";
  $("notes").value = r.correction?.notes || "";
  $("wellProgress").textContent = w
    ? `${w.id}: ${C.analysisIndices(project).filter((i) => !C.boundaryNeedsReview(w, i)).length} / ${C.analysisIndices(project).length} boundaries reviewed; ${C.analysisIndices(project).filter((i) => project.records[C.key(i, w.uid)]?.correction?.state).length} / ${C.analysisIndices(project).length} frames manually labeled.`
    : "Select a well to review and label its sequence.";
  $("undoLabels").disabled = busy || !lastLabelOperation;
  $("wellSpecies").value = w?.species || "";
  $("rangeStart").min = $("rangeEnd").min = C.analysisStart(project) + 1;
  $("rangeStart").value = Math.max(
    Number($("rangeStart").value),
    C.analysisStart(project) + 1,
  );
  $("rangeEnd").value = Math.max(
    Number($("rangeEnd").value),
    C.analysisStart(project) + 1,
  );
  $("rangeStart").max = $("rangeEnd").max = project.frames.length || 1;
  const state =
    study === "ccrt" ? C.effective(r, w).toUpperCase() : "EGG COUNT";
  $("prediction").textContent = w
    ? `${w.id} · ${state}${study === "eggs" ? " · Eggs: " + (r.correction?.egg_count ?? r.egg_count ?? "Not counted") : ""}${r.correction ? " · Manual annotation applied" : r.temporal_inference ? " · Inferred between matching frames" : ""}${r.confidence == null ? "" : " · Model confidence: " + Math.round(r.confidence * 100) + "%"}${C.boundaryNeedsReview(w, index) ? " · Boundary needs review" : ""}${(w.partial_frames?.[index] ?? w.detection.partial) ? " · Partially visible well" : ""}`
    : "No well selected.";
  $("predictionInfo").textContent = r.model
    ? `${r.model.name} · Version: ${r.model.version} · Model state: ${(r.prediction || "unknown").toUpperCase()} · Confidence: ${r.confidence == null ? "Not available" : Math.round(r.confidence * 100) + "%"}${r.raw_prediction ? " · Raw state: " + r.raw_prediction.toUpperCase() + " · Raw confidence: " + (r.raw_confidence == null ? "Not available" : Math.round(r.raw_confidence * 100) + "%") : ""}${r.recovery_decision?.confirmed_frame_index != null ? " · Recovery starts at frame " + (r.recovery_decision.onset_frame_index + 1) + ", confirmed at frame " + (r.recovery_decision.confirmed_frame_index + 1) : ""}${r.temporal_inference ? " · Inferred between frames " + (r.temporal_inference.left_frame_index + 1) + " and " + (r.temporal_inference.right_frame_index + 1) + "; not a direct confident prediction" : ""}${r.occupancy_method ? " · Occupancy: " + r.occupancy_prediction + " (" + r.occupancy_method + ")" : ""}`
    : "";
  if (r.inference_skipped)
    $("predictionInfo").textContent +=
      " · Inference skipped after confirmed Awake; no raw model output for this frame.";
  renderSummary();
  updateContext();
}
function renderSummary() {
  const rows = C.humanReport(project).results,
    body = $("summary");
  body.replaceChildren();
  $("summaryHead").replaceChildren();
  const head = document.createElement("tr");
  Object.keys(rows[0] || {}).forEach((key) => {
    const th = document.createElement("th");
    th.textContent = key
      .replace(/_/g, " ")
      .replace(/^./, (c) => c.toUpperCase());
    head.append(th);
  });
  const temporal = study === "ccrt" && project.mode === "sequence";
  if (temporal) {
    const th = document.createElement("th");
    th.textContent = "Timeline";
    head.append(th);
  }
  $("summaryHead").append(head);
  rows.forEach((row, rowIndex) => {
    const tr = document.createElement("tr");
    Object.values(row).forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value === "" ? "—" : value;
      tr.append(td);
    });
    if (temporal) {
      const td = document.createElement("td"),
        spark = document.createElement("div");
      spark.className = "spark";
      C.stateRuns(project, project.wells[rowIndex]).forEach(
        ({ state, start, end }) => {
          const span = document.createElement("span");
          span.style.background = colors[state];
          span.style.flexGrow = end - start + 1;
          span.title = `Frames ${start + 1}–${end + 1}: ${state}`;
          spark.append(span);
        },
      );
      spark.setAttribute("role", "img");
      spark.setAttribute(
        "aria-label",
        `${row.Well}: ${C.analysisIndices(project).length} frames, from first to last. Hover a segment for frame range and state.`,
      );
      td.append(spark);
      tr.append(td);
    }
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
  if (
    study === "ccrt" &&
    $("layout").value === "individual" &&
    project.wells.length
  ) {
    const well = project.wells[0];
    if (
      !confirm(
        `Single-well mode keeps one identity. Replace ${well.id}'s boundary in the selected edit scope? Manual labels are preserved.`,
      )
    )
      return;
    saveGeometry(well, points);
    select(well.uid);
    return;
  }
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
    area_type:
      study === "eggs"
        ? points.length === 4 &&
          points.every(
            (p) => (p[0] === 0 || p[0] === 1) && (p[1] === 0 || p[1] === 1),
          )
          ? "image"
          : "region"
        : undefined,
    active_frames: study === "eggs" ? [index] : undefined,
    review_frames:
      study === "eggs"
        ? []
        : project.frames.map((_, i) => i).filter((i) => i !== index),
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
  workflowStage = "images";
  files = C.orderFiles(list, $("order").value);
  if ($("mode").value === "single" && files.length > 1) {
    files = files.slice(0, 1);
    status(
      "Single-image mode loaded the first ordered file. Choose an individual image to change it.",
    );
  } else
    status(
      study === "eggs"
        ? `${files.length} independent images loaded. Count each image or define its counting regions.`
        : `${files.length} images loaded. Confirm order, interval and well boundaries.`,
    );
  urls = files.map((f) => URL.createObjectURL(f));
  project = {
    schema_version: 1,
    time_origin: {
      confirmed: $("timeConfirmed").checked,
      offset_seconds: Math.max(0, Number($("recoveryDelay").value) || 0),
    },
    analysis_start_index: 0,
    input_kind: $("inputKind").value,
    source_first_frame_number: $("sourceFirstFrame").value
      ? Number($("sourceFirstFrame").value)
      : null,
    species: $("gridSpecies").value.trim(),
    recovery_policy: selectedRecoveryPolicy(),
    task: study,
    experiment_id: $("experiment").value.trim() || crypto.randomUUID(),
    mode: $("mode").value,
    layout: $("layout").value,
    interval: Number($("interval").value) || 1,
    order: $("order").value,
    frames: files.map(frameMetadata),
    wells: [],
    records: {},
    history: [],
  };
  index = 0;
  $("analysisStart").value = 1;
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
  lastLabelOperation = null;
  if (study === "eggs") {
    for (index = 0; index < files.length; index++)
      addWell([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ]);
    index = 0;
    selected = visibleWells()[0]?.uid || null;
  }
  await renderFrame();
  if (study === "ccrt" && $("layout").value === "individual") await detect();
}
function moveTo(i) {
  if (busy || !files.length) return;
  index = Math.max(
    workflowStage === "images" ? 0 : C.analysisStart(project),
    Math.min(files.length - 1, i),
  );
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
    "rimMargin",
    "resizePercent",
    "resizeWells",
    "proposalSelect",
    "align",
    "alignSequence",
    "clearRegion",
    "confirmFrame",
    "confirmSequence",
    "importReportButton",
    "boundaryApproval",
    "whole",
    "tool",
    "scope",
    "model",
    "analyze",
    "importButton",
    "interval",
    "wellSelect",
    "mask",
    "json",
    "csv",
    "report",
    "machine",
    "timeConfirmed",
    "recoveryDelay",
    "eggScope",
    "confirmSingles",
    "training",
    "trainingFolder",
    "datasetTest",
    "cropDataset",
    "cropPadding",
    "cropEncoding",
    "comparisonSeed",
    "poorWell",
    "richWell",
    "experiment",
    "saveDraft",
    "drawWell",
    "setRegion",
    "inputKind",
    "analysisStart",
    "sourceFirstFrame",
    "startHere",
    "gridSpecies",
    "wellSpecies",
    "decisionThreshold",
    "awakeFrames",
    "requireComaBaseline",
    "useModelPolicy",
    "inferMatchingGaps",
    "stopAfterAwake",
    "inputPreview",
    "inputPreviewMode",
  ].forEach((id) => ($(id).disabled = value));
  $("cancel").hidden = !value;
  $("progress").hidden = !value;
  updateDetails();
  updateContext();
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
      study === "eggs"
        ? "Replace this image’s regions and annotations? Export first to preserve them."
        : "Replace all well boundaries and annotations with new proposals? Export first to preserve them.",
    )
  )
    return;
  stop();
  cancelled = false;
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
    form.append("margin", $("rimMargin").value);
    if (project.detection_region)
      form.append("region", JSON.stringify(project.detection_region));
    const data = await post("/api/detect", form);
    if (cancelled)
      return status(
        "Detection cancelled. Existing boundaries and annotations were retained.",
      );
    if (!data.settings?.algorithm_version?.startsWith("native-outer-rims-"))
      throw Error(
        "The app server is using an older detector. Restart the app after updating, then reload this page. Existing boundaries and annotations are unchanged.",
      );
    if (!data.wells.length) {
      status(
        "No matching wells found. Existing wells are preserved. Try a diameter override, another detection method, or draw boundaries manually.",
      );
      return;
    }
    const removed = study === "eggs" ? visibleWells() : project.wells;
    const removedIDs = new Set(removed.map((w) => w.uid));
    project.wells = project.wells.filter((w) => !removedIDs.has(w.uid));
    for (const k of Object.keys(project.records))
      if (removedIDs.has(k.split(":")[1])) delete project.records[k];
    lastLabelOperation = null;
    project.detection_settings = data.settings;
    selected = null;
    data.wells.forEach((w) =>
      addWell(w.points, {
        source: w.source,
        confidence: w.confidence,
        ring_coverage: w.ring_coverage ?? null,
        outer_rim_coverage: w.outer_rim_coverage ?? null,
        recovered_from_neighbours: !!w.recovered_from_neighbours,
        algorithm_version: data.settings?.algorithm_version || null,
        partial: w.partial ?? false,
        review_required: true,
      }),
    );
    const recovered = data.wells.filter(
      (w) => w.recovered_from_neighbours,
    ).length;
    status(
      `${data.wells.length} proposals.${recovered ? ` ${recovered} additional rim${recovered === 1 ? "" : "s"} found using neighbouring wells and visible edges.` : ""} ${data.warning}`,
    );
  } catch (e) {
    status(e.message);
  } finally {
    setBusy(false);
    renderFrame();
  }
}
function invalidate(uid, frame) {
  const temporal =
    frame != null &&
    project.mode === "sequence" &&
    project.records[C.key(frame, uid)]?.model?.requires_sequence;
  const frames =
    frame == null || temporal
      ? project.frames
          .map((_, i) => i)
          .filter((i) => frame == null || i >= frame)
      : [frame];
  for (const f of frames) {
    const k = C.key(f, uid),
      r = project.records[k];
    if (r) {
      history("invalidate-prediction", { frame: f, uid, previous: r });
      project.records[k] = r.correction ? { correction: r.correction } : {};
    }
  }
  if (study === "ccrt") C.applyRecoveryPolicy(project);
}
function saveGeometry(w, points) {
  if (study === "eggs") w.area_type = "region";
  if ($("scope").value === "frame" || study === "eggs") {
    w.overrides[index] = points;
    invalidate(w.uid, index);
  } else {
    w.points = points;
    w.overrides = {};
    w.review_frames = project.frames.map((_, i) => i);
    invalidate(w.uid, null);
  }
  if ($("scope").value === "frame")
    w.review_frames = [...new Set([...(w.review_frames || []), index])];
  else {
    w.detection.review_required = true;
    w.review_frames = project.frames.map((_, i) => i);
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
  if (!image || busy || (study === "eggs" && $("eggScope").value === "image"))
    return;
  stop();
  const p = position(e),
    tool = workflowStage === "wells" ? $("tool").value : "select";
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
    const w = [...visibleWells()].reverse().find((w) => {
      pointsPath(ctx, C.geometry(w, index), canvas.width, canvas.height);
      return ctx.isPointInPath(p[0] * canvas.width, p[1] * canvas.height);
    });
    select(w?.uid || null);
    if (tool === "remove") {
      if (w) removeSelectedWell();
      return;
    }
    if (w && workflowStage === "wells")
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
      if (study === "eggs") {
        project.detection_regions ??= {};
        project.detection_regions[index] = drag.pending;
      }
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
  if (
    !image ||
    busy ||
    workflowStage !== "wells" ||
    $("tool").value !== "polygon" ||
    e.detail !== 1
  )
    return;
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
  if (
    busy ||
    workflowStage !== "wells" ||
    $("tool").value !== "polygon" ||
    draft.length < 3
  )
    return;
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
  if (
    (e.key === "Delete" || e.key === "Backspace") &&
    selected &&
    !busy &&
    workflowStage === "wells"
  ) {
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
    $("experiment").value = project.experiment_id;
    lastLabelOperation = null;
    $("layout").value =
      project.layout || (project.wells.length > 1 ? "grid" : "individual");
    $("mode").value = project.mode;
    return;
  }
  stop();
  urls.forEach(URL.revokeObjectURL);
  files = [];
  workflowStage = "images";
  urls = [];
  index = 0;
  selected = null;
  draft = [];
  project = {
    schema_version: 1,
    time_origin: {
      confirmed: $("timeConfirmed").checked,
      offset_seconds: Math.max(0, Number($("recoveryDelay").value) || 0),
    },
    analysis_start_index: 0,
    input_kind: $("inputKind").value,
    source_first_frame_number: $("sourceFirstFrame").value
      ? Number($("sourceFirstFrame").value)
      : null,
    species: $("gridSpecies").value.trim(),
    recovery_policy: selectedRecoveryPolicy(),
    task: study,
    experiment_id: $("experiment").value.trim() || crypto.randomUUID(),
    mode: $("mode").value,
    layout: $("layout").value,
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
  lastLabelOperation = null;
  renderFrame();
};
$("order").onchange = () => {
  if (files.length) {
    status(
      "This ordering applies to the next image load. The current image order and annotations are unchanged. Save your review before loading replacement images.",
    );
  }
};
$("interval").onchange = () => {
  const v = Number($("interval").value);
  if (!Number.isFinite(v) || v <= 0) {
    $("inputKind").value = project.input_kind || "raw";
    $("analysisStart").value = C.analysisStart(project) + 1;
    $("sourceFirstFrame").value =
      project.source_first_frame_number === null
        ? ""
        : project.source_first_frame_number || 1;
    $("gridSpecies").value = project.species || "";
    $("decisionThreshold").value =
      project.recovery_policy?.confidence_threshold ?? 0.8;
    $("awakeFrames").value =
      project.recovery_policy?.awake_consecutive_frames ?? 3;
    $("requireComaBaseline").checked =
      project.recovery_policy?.require_coma_before_awake ?? false;
    $("interval").value = project.interval;
    return;
  }
  project.interval = v;
  if (study === "ccrt") C.applyRecoveryPolicy(project);
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
    updateDetails();
    draw();
  }
};
$("confirm").onclick = () => {
  const w = selectedWell();
  if (w) {
    if (
      project.mode === "sequence" &&
      $("boundaryApproval").value === "sequence"
    )
      return approveStable([w.uid]);
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
    state:
      study === "ccrt"
        ? ["empty", "multiple"].includes(selectedWell()?.occupancy)
          ? record().correction?.state || null
          : $("correction").value || null
        : null,
    egg_count: study === "eggs" ? count : null,
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
    project.wells.some((w) =>
      C.analysisIndices(project).some((i) => C.boundaryNeedsReview(w, i)),
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
  const sequenceSession = crypto.randomUUID();
  let workerSession = sequenceSession,
    lastInferredFrame = null;
  project.inference_profile = [];
  try {
    for (const i of C.analysisIndices(project)) {
      if (cancelled) break;
      const pending = C.prepareInference(
        project,
        i,
        sequenceSession,
        $("stopAfterAwake").checked,
      );
      if (!pending.length) {
        $("progress").value = i + 1;
        continue;
      }
      const form = new FormData();
      form.append("image", files[i]);
      form.append("model", $("model").value);
      if (project.mode === "sequence") {
        if (lastInferredFrame !== null && i !== lastInferredFrame + 1)
          workerSession = crypto.randomUUID();
        form.append("sequence_session", workerSession);
        form.append("frame_index", String(i));
      }
      form.append(
        "wells",
        JSON.stringify(
          pending.map((w) => ({
            id: w.uid,
            points: C.geometry(w, i),
            occupancy: w.occupancy,
            partial: w.partial_frames?.[i] ?? !!w.detection.partial,
          })),
        ),
      );
      const requestStarted = performance.now();
      const data = await post("/api/analyze", form);
      lastInferredFrame = i;
      project.inference_profile.push({
        frame_index: i,
        wells: pending.length,
        browser_roundtrip_s: (performance.now() - requestStarted) / 1000,
        ...data.timing,
      });
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
          raw_prediction: r.raw_prediction ?? r.prediction,
          analysis_session: sequenceSession,
          raw_confidence: r.raw_confidence ?? r.confidence,
          recovery_decision: null,
          state_probabilities: r.state_probabilities || null,
          occupancy_prediction: r.occupancy_prediction || null,
          occupancy_score: r.occupancy_score ?? null,
          occupancy_method: r.occupancy_method || null,
          confidence: r.confidence,
          decision_eligible: r.decision_eligible,
          abstention_reason: r.abstention_reason,
          inference_skipped: null,
          model: data.model,
          egg_count: r.egg_count ?? null,
        };
      });
      if (study === "ccrt") C.applyRecoveryPolicy(project);
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
$("inputPreview").onclick = async () => {
  try {
    if (!selected || !files[index])
      return status("Select an image and a well first.");
    const form = new FormData();
    form.append("image", files[index]);
    form.append(
      "wells",
      JSON.stringify([
        { id: selected, points: C.geometry(selectedWell(), index) },
      ]),
    );
    form.append("model", $("model").value);
    form.append("mode", $("inputPreviewMode").value);
    const result = await post("/api/input-preview", form);
    $("inputPreviewOriginal").src = result.original;
    $("inputPreviewImage").src = result.image;
    $("inputPreviewOriginal").hidden = $("inputPreviewImage").hidden = false;
    $("inputPreviewInfo").textContent =
      `${result.size} × ${result.size} · ${result.foreground_status.replaceAll("_", " ")}`;
  } catch (e) {
    status(e.message);
  }
};
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type })),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
$("json").onclick = () =>
  download(
    `flyscope-${study}-project.json`,
    JSON.stringify(C.projectBackup(project)),
    "application/json",
  );
$("csv").onclick = () =>
  download(
    `flyscope-${study}-results.csv`,
    C.tableCSV(C.humanReport(project).results),
    "text/csv",
  );
$("importButton").onclick = () => $("import").click();
async function importSaved(file) {
  try {
    if (!file) return;
    if (!files.length)
      throw Error(
        "Select the original image files first, in their saved order.",
      );
    const value = /\.zip$/i.test(file.name)
      ? await TrainingExport.readArchiveJSON(file)
      : JSON.parse(await file.text());
    if (value.kind?.endsWith("-results")) {
      const imported = C.importReadableReport(project, value);
      if (
        project.records &&
        Object.keys(project.records).length &&
        !confirm(
          "Replace matching image/well annotations with the saved report labels?",
        )
      )
        return;
      project = imported.project;
      lastLabelOperation = null;
      $("experiment").value = project.experiment_id;
      status(
        `Imported ${imported.imported} report rows onto matching wells. Current boundaries are preserved; older readable reports cannot restore omitted geometry or full history.`,
      );
    } else {
      const loaded = C.validateProject(value.project || value, project.frames);
      if ((loaded.task || "ccrt") !== study)
        throw Error(
          "Open the matching study page before restoring this project.",
        );
      if (
        project.wells.length &&
        !confirm("Replace this review with the saved project?")
      )
        return;
      stop();
      project = {
        ...loaded,
        task: study,
        experiment_id: loaded.experiment_id || crypto.randomUUID(),
        history: loaded.history || [],
      };
      $("experiment").value = project.experiment_id;
      lastLabelOperation = null;
      selected = project.wells[0]?.uid || null;
      status(
        "Project restored. Images matched by path, size and modification time.",
      );
    }
    $("layout").value =
      project.layout || (project.wells.length > 1 ? "grid" : "individual");
    $("mode").value = project.mode;
    $("interval").value = project.interval;
    $("order").value = project.order || "name";
    $("timeConfirmed").checked = project.time_origin?.confirmed ?? true;
    $("recoveryDelay").value = project.time_origin?.offset_seconds || 0;
    $("eggScope").value = project.wells.some((w) => !C.imageArea(w))
      ? "areas"
      : "image";
    $("useModelPolicy").checked =
      project.recovery_policy?.source !== "manual_override" &&
      project.recovery_policy?.source !== "manual";
    updatePolicyControls();
    workflowStage = project.wells.length ? "review" : "images";
    await renderFrame();
  } catch (error) {
    status(error.message);
  }
}
$("import").onchange = async (e) => {
  await importSaved(e.target.files[0]);
  e.target.value = "";
};
$("importReportButton").onclick = () => $("importReport").click();
$("importReport").onchange = async (e) => {
  await importSaved(e.target.files[0]);
  e.target.value = "";
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
    models = (await response.json()).filter(
      (m) =>
        m.id === "manual" ||
        (study === "eggs"
          ? m.task === "eggs"
          : ["coma", "ccrt"].includes(m.task)),
    );
    $("model").replaceChildren(
      ...models.map(
        (m) =>
          new Option(
            `${m.id === "manual" ? "Manual " + (study === "eggs" ? "egg counting" : "CCRT review") : m.name} · ${m.version.slice(0, 12)}${m.available === false ? (m.untrained ? " (train first)" : " (runtime unavailable)") : ""}`,
            m.id,
          ),
      ),
    );
    [...$("model").options].forEach((option) => {
      if (models.find((m) => m.id === option.value)?.available === false)
        option.disabled = true;
    });
    $("model").onchange = () => {
      const m = models.find((m) => m.id === $("model").value);
      if (m?.recovery_policy && $("useModelPolicy").checked) {
        project.recovery_policy = structuredClone(m.recovery_policy);
      }
      updatePolicyControls();
      $("modelInfo").textContent =
        m?.id === "manual"
          ? study === "eggs"
            ? "Manual egg annotations; no egg model is registered."
            : "Manual annotations only. Choose a model above to predict states."
          : m?.description || "";
      updateContext();
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
    (!entireSequence && index === C.analysisStart(project))
  )
    return status(
      "Define wells on the first frame, then select a later sequence frame or align the entire sequence.",
    );
  const reference = C.analysisStart(project),
    targets = entireSequence ? C.analysisIndices(project).slice(1) : [index];
  const baseWells = project.wells.map((w) => ({
    id: w.uid,
    points: C.geometry(w, reference),
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
        form.append("reference", files[reference]);
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
          reference_frame: reference,
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
          reference_frame: reference,
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
  if (study === "eggs") delete project.detection_regions?.[index];
  history("clear-detection-region", {});
  updateContext();
  draw();
};
$("confirmFrame").onclick = () => {
  if (!project.wells.length) return status("Select wells first.");
  if (
    !confirm(
      "Confirm the visible boundaries on this frame? Other frames still require their own review.",
    )
  )
    return;
  visibleWells().forEach((w) => {
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
function updateContext() {
  updateWorkflow();
  const sequence =
      study === "ccrt" && $("mode").value === "sequence" && files.length > 1,
    multiple = $("mode").value !== "single" && files.length > 1;
  document
    .querySelectorAll(
      "[data-ccrt],[data-eggs],[data-sequence],[data-multiple],[data-multiple-mode],[data-geometry],[data-loaded],[data-wells],[data-project-wells],[data-selected],[data-workflow-stages]",
    )
    .forEach((el) => {
      el.hidden =
        (el.hasAttribute("data-ccrt") && study !== "ccrt") ||
        (el.hasAttribute("data-eggs") && study !== "eggs") ||
        (el.hasAttribute("data-sequence") && !sequence) ||
        (el.hasAttribute("data-multiple") && !multiple) ||
        (el.hasAttribute("data-multiple-mode") &&
          $("mode").value === "single") ||
        (el.hasAttribute("data-loaded") && !files.length) ||
        (el.hasAttribute("data-wells") && !visibleWells().length) ||
        (el.hasAttribute("data-project-wells") && !project.wells.length) ||
        (el.hasAttribute("data-selected") && !selectedWell()) ||
        (el.hasAttribute("data-workflow-stages") &&
          !el.dataset.workflowStages.split(" ").includes(workflowStage)) ||
        (el.hasAttribute("data-geometry") &&
          (!files.length ||
            (study === "eggs" && $("eggScope").value === "image")));
    });
  document
    .querySelector(".previews")
    .classList.toggle("single-preview", !sequence);
  $("currentCaption").textContent = sequence
    ? "T — current frame"
    : "Current image";
  document.querySelector("#wellSelect").previousElementSibling.textContent =
    study === "ccrt"
      ? "Selected well"
      : $("eggScope").value === "image"
        ? "Selected image"
        : "Selected counting area";
  $("summaryTitle").textContent =
    study === "eggs"
      ? "Egg-count summary"
      : sequence
        ? "Sequence summary"
        : "Image summary";
  $("reviewGuide").textContent =
    study === "eggs"
      ? "Count each independent image. No well layout is assumed. Optional custom areas are available if future data need them; the annotation format is provisional."
      : "Draw or correct wells on the first frame. Keep their IDs across the sequence; align or edit later frames. Approve stable boundaries across the sequence, or confirm individual frames when needed.";
  $("experiment").value ||= project.experiment_id;
  $("analyze").textContent =
    study === "eggs"
      ? multiple
        ? "Analyze image collection"
        : "Analyze image"
      : sequence
        ? "Analyze sequence"
        : "Analyze image";
  if (study === "eggs") $("scope").value = "frame";
  $("align").disabled =
    busy ||
    !sequence ||
    index === C.analysisStart(project) ||
    C.analysisIndices(project).length < 2;
  $("alignSequence").disabled = busy || !sequence || files.length < 2;
  $("analyze").disabled = busy || $("model").value === "manual";
  $("order").disabled = busy;
  $("confirmSingles").hidden =
    study !== "ccrt" ||
    workflowStage !== "review" ||
    !project.wells.some((w) => !w.occupancy || w.occupancy === "unknown");
  $("nextBoundary").hidden =
    !sequence ||
    workflowStage !== "wells" ||
    !selectedWell() ||
    !C.analysisIndices(project).some((i) =>
      C.boundaryNeedsReview(selectedWell(), i),
    );
  $("nextLabel").hidden =
    !sequence ||
    workflowStage !== "review" ||
    !selectedWell() ||
    ["empty", "multiple"].includes(selectedWell().occupancy) ||
    !C.analysisIndices(project).some(
      (i) => !project.records[C.key(i, selected)]?.correction?.state,
    );
  $("predictionDetails").hidden =
    workflowStage !== "review" || !selectedWell() || !record().model;

  [...$("model").options].forEach((option) => {
    const m = models.find((m) => m.id === option.value);
    option.disabled =
      m?.available === false || (m?.requires_sequence && !sequence);
  });
  if (
    models.find((m) => m.id === $("model").value)?.requires_sequence &&
    !sequence
  )
    $("model").value = "manual";
  $("manualPolicyOptions").hidden =
    study !== "ccrt" ||
    $("useModelPolicy").checked ||
    $("model").value === "manual";
  $("policyInfo").hidden = study !== "ccrt" || $("model").value === "manual";
  $("useModelPolicy").closest("label").hidden =
    study !== "ccrt" || $("model").value === "manual";
  $("modelOptions").hidden = !models.some((model) => model.id !== "manual");
  $("clearRegion").hidden = !project.detection_region;
  $("cropControls").hidden = !$("cropDataset").checked;
  $("comparisonOptions").hidden = $("datasetTest").checked || study !== "ccrt";
  $("training").textContent = $("datasetTest").checked
    ? "Export test dataset ZIP"
    : `Export ${study === "ccrt" ? "CCRT" : "egg"} training dataset ZIP`;
  $("datasetHint").textContent = $("datasetTest").checked
    ? "Evaluation images and annotations only. Unannotated areas are not scored. Save a project backup separately to resume editing."
    : "One image per frame and shared boundaries; individual well crops are made during training. Partial annotations are supported. For large experiments, save directly to a folder. ZIP exports are split into bounded parts; extract every part into the same folder.";
  $("trainingFolder").hidden = typeof window.showDirectoryPicker !== "function";
  $("analyze").hidden = $("model").value === "manual";
  $("detectionHint").textContent =
    $("detectionMethod").value === "contours"
      ? "Propose non-circular boundaries, including simple concave shapes. Draw or correct boundaries if the rim is open or obscured."
      : $("layout").value === "individual"
        ? "Single-well detection fits one dominant rim in a close crop. Draw a boundary or select the full image if no rim is visible."
        : "Size and repeated-spacing filters reduce fixture proposals. Set a diameter when automatic estimation is wrong, or draw a detection region to exclude hardware.";
  $("timeline").min = workflowStage === "images" ? 0 : C.analysisStart(project);
  $("jump").min = Number($("timeline").min) + 1;
  $("analysisStart").max = files.length || 1;
  $("analysisWindowInfo").textContent =
    `${C.analysisIndices(project).length} analysis frames; frames before ${C.analysisStart(project) + 1} are setup only. Original image IDs and timer positions are retained.`;
  $("diameter").closest("label").hidden =
    $("detectionMethod").value === "contours";
  document
    .querySelectorAll('a[href="/ccrt"],a[href="/eggs"]')
    .forEach((a) => a.setAttribute("aria-disabled", String(busy)));
}
$("experiment").onchange = () => {
  const value = $("experiment").value.trim();
  if (!value) {
    $("experiment").value = project.experiment_id;
    return;
  }
  project.experiment_id = value;
  history("set-experiment", { experiment_id: value });
};
$("detectionMethod").onchange = updateContext;
function labelRange(state, first, last) {
  if (
    study === "ccrt" &&
    ["empty", "multiple"].includes(selectedWell()?.occupancy)
  )
    return status(
      "Empty or multiple-individual wells have no valid CCRT state. Check occupancy before assigning labels.",
    );
  if (busy || !selectedWell()) return status("Select a well first.");
  const count = last - first + 1;
  if (
    !Number.isInteger(first) ||
    !Number.isInteger(last) ||
    first < C.analysisStart(project) ||
    last < first ||
    last >= files.length
  )
    return status("Choose a valid inclusive frame range.");
  const replacements = Array.from(
    { length: count },
    (_, n) => project.records[C.key(first + n, selected)]?.correction?.state,
  ).filter(Boolean).length;
  if (
    replacements &&
    !confirm(
      `Label ${selectedWell().id} ${state} on frames ${first + 1}–${last + 1} inclusive? This replaces ${replacements} manual labels.`,
    )
  )
    return;
  lastLabelOperation = C.applyStateRange(
    project,
    selected,
    first,
    last,
    state,
    $("notes").value,
  );
  updateDetails();
  draw();
  status(
    `${selectedWell().id}: ${state} on ${count} frame${count === 1 ? "" : "s"} (${first + 1}–${last + 1}, inclusive).`,
  );
}
$("labelComa").onclick = () => labelRange("coma", index, index);
$("labelAwake").onclick = () => labelRange("awake", index, index);
$("labelUncertain").onclick = () => labelRange("unknown", index, index);
$("comaUntil").onclick = () =>
  labelRange("coma", C.analysisStart(project), index);
$("awakeFrom").onclick = () => labelRange("awake", index, files.length - 1);
$("applyRange").onclick = () =>
  labelRange(
    $("rangeState").value,
    Number($("rangeStart").value) - 1,
    Number($("rangeEnd").value) - 1,
  );
$("undoLabels").onclick = () => {
  if (!lastLabelOperation || busy) return;
  const restored = C.undoStateRange(project, lastLabelOperation);
  lastLabelOperation = null;
  updateDetails();
  draw();
  status(
    `Restored ${restored} previous frame labels. Later edits were preserved.`,
  );
};
$("report").onclick = async () => {
  if (!files.length || !project.wells.length)
    return status("Load images and define regions before exporting a report.");
  stop();
  setBusy(true);
  cancelled = false;
  try {
    const report = C.humanReport(project),
      entries = [
        {
          name: "project.json",
          text: JSON.stringify(C.projectBackup(project)),
        },
        { name: "report.json", text: JSON.stringify(report, null, 2) },
        { name: "results.csv", text: C.tableCSV(report.results) },
        { name: "summary.csv", text: C.tableCSV(report.summary) },
        {
          name: "well-locations.csv",
          text: C.tableCSV(
            study === "eggs"
              ? project.frames.flatMap((_, i) => C.wellMap(project, i))
              : C.wellMap(project),
          ),
        },
        { name: "report.html", text: humanHTML(report) },
      ];
    const frames =
      study === "eggs"
        ? project.frames
            .map((_, i) => i)
            .filter((i) =>
              project.wells.some((w) => C.activeWell(w, i) && !C.imageArea(w)),
            )
        : [C.analysisStart(project)];
    for (const frame of frames) {
      if (cancelled) throw Error("Report export cancelled.");
      entries.push({
        name: `maps/frame-${String(frame + 1).padStart(6, "0")}.png`,
        blob: await TrainingExport.locationMap(project, files, frame),
      });
    }
    entries.push({
      name: "README.txt",
      text: `FlyScope ${study} results\nOpen report.html for the human report. results.csv contains compact ${study === "eggs" ? "independent-image counts" : "one-row-per-well occupancy, recovery frame IDs and CCRT seconds only when the time reference is known"}. summary.csv contains two-column experimental measurements. Missing values are not zero. Machine analysis is a separate export with frame-level labels and provenance. well-locations.csv gives normalized percentage locations with matching numbered maps in maps/. CCRT maps refer to the first analysis frame; IDs stay stable through camera corrections. The included project.json restores editable boundaries, annotations, model provenance and history via Import report. Training datasets are a separate export.\n`,
    });
    download(
      `flyscope-${study}-report.zip`,
      await TrainingExport.zip(
        entries,
        () => {},
        () => cancelled,
      ),
      "application/zip",
    );
    status(
      `${study.toUpperCase()} report exported with readable results, summary and location maps.`,
    );
  } catch (error) {
    status(error.message);
  } finally {
    setBusy(false);
    renderFrame();
  }
};
let datasetDownloadURLs = [];
async function exportDataset(toFolder = false) {
  if (!files.length) return status("Load images first.");
  let directory;
  if (toFolder) {
    try {
      const parent = await window.showDirectoryPicker({ mode: "readwrite" });
      directory = await parent.getDirectoryHandle(
        `flyscope-${study}-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`,
        { create: true },
      );
    } catch (error) {
      return status(
        error.name === "AbortError"
          ? "Folder selection cancelled. Annotations are unchanged."
          : `Cannot open export folder: ${error.message}`,
      );
    }
  }
  stop();
  cancelled = false;
  setBusy(true);
  $("progress").value = 0;
  datasetDownloadURLs.forEach(URL.revokeObjectURL);
  datasetDownloadURLs = [];
  $("datasetDownloads").replaceChildren();
  $("datasetDownloads").hidden = true;
  try {
    const { archives, manifest } = await TrainingExport.training(
      project,
      files,
      (done, total, phase) => {
        $("progress").max = total;
        $("progress").value = done;
        status(
          `${phase === "archive" ? "Packing ZIP files" : phase === "metadata" ? "Writing annotations and shared selections" : phase === "saving-images" ? "Saving dataset images to folder" : "Preparing dataset images"}: ${done} / ${total}… Export is complete only when a saved-folder message or download links appear.`,
        );
      },
      () => cancelled,
      {
        directory,
        cropToWells: study === "ccrt" && $("cropDataset").checked,
        cropPadding: Number($("cropPadding").value),
        cropEncoding: $("cropEncoding").value,
        role: $("datasetTest").checked ? "test" : "train",
        comparisons: {
          seed: Number($("comparisonSeed").value),
          poor_well_id: $("poorWell").value,
          rich_well_id: $("richWell").value,
        },
      },
    );
    if (cancelled) throw Error("Dataset export cancelled.");
    if (archives) {
      const box = $("datasetDownloads"),
        intro = document.createElement("p");
      intro.textContent =
        archives.length === 1
          ? "Dataset ready. If downloading did not start, use this link:"
          : `Dataset ready in ${archives.length} ZIP parts. Save every part, then extract all of them into the same empty folder:`;
      box.append(intro);
      archives.forEach((blob, i) => {
        const a = document.createElement("a"),
          url = URL.createObjectURL(blob);
        datasetDownloadURLs.push(url);
        a.href = url;
        a.download = `flyscope-${study}-${manifest.role === "test" ? "test" : "training"}${archives.length > 1 ? `-part-${String(i + 1).padStart(3, "0")}` : ""}.zip`;
        a.textContent = `${a.download} (${(blob.size / 1048576).toFixed(1)} MB)`;
        box.append(a);
        if (archives.length === 1) a.click();
      });
      box.hidden = false;
    }
    status(
      `${directory ? `Dataset saved in ${directory.name} inside your selected folder.` : "Dataset prepared; download links are below."} ${manifest.annotated_sample_count} manual samples; ${manifest.eligible_sample_count} eligible, ${manifest.annotated_sample_count - manifest.eligible_sample_count} flagged for review. Dataset images and boundaries included; individual well crops are derived during training.`,
    );
  } catch (error) {
    status(
      `Dataset export failed: ${error.message}${directory ? " The new folder may be incomplete; export again to a new folder. Your annotations are unchanged." : " Your annotations are unchanged. Try Save dataset to folder for large experiments."}`,
    );
  } finally {
    setBusy(false);
    renderFrame();
  }
}
$("training").onclick = () => exportDataset(false);
$("trainingFolder").onclick = () => exportDataset(true);

$("layout").onchange = () => {
  if ($("layout").value === "individual" && project.wells.length > 1) {
    $("layout").value = "grid";
    status(
      "Single-well mode allows one well. Remove extra wells or start with new images first.",
    );
  }
  project.layout = $("layout").value;
  updateContext();
};
$("resizeWells").onclick = () => {
  if (busy || !project.wells.length) return;
  try {
    const factor = 1 + Number($("resizePercent").value) / 100;
    if (factor === 1) return status("Choose a nonzero size adjustment.");
    // Validate the entire edit before mutating any well.
    const changes = project.wells.map((w) => ({
      w,
      base: C.resizeBoundary(w.points, factor),
      overrides: Object.fromEntries(
        Object.entries(w.overrides || {}).map(([i, p]) => [
          i,
          C.resizeBoundary(p, factor),
        ]),
      ),
    }));
    history("resize-well-boundaries", {
      factor,
      previous: project.wells.map((w) => ({
        uid: w.uid,
        points: w.points,
        overrides: w.overrides,
      })),
    });
    for (const { w, base, overrides } of changes) {
      invalidate(w.uid, null);
      const previousPartial = !!w.detection.partial;
      w.points = base.points;
      w.detection.partial = !!w.detection.partial || base.partial;
      w.detection.review_required = true;
      w.overrides = Object.fromEntries(
        Object.entries(overrides).map(([i, p]) => [i, p.points]),
      );
      for (const [i, p] of Object.entries(overrides)) {
        w.partial_frames ||= {};
        w.partial_frames[i] =
          (w.partial_frames[i] ?? previousPartial) || p.partial;
      }
    }
    updateDetails();
    draw();
    previews();
    status(
      `Resized ${changes.length} wells across the sequence. Well IDs and manual annotations are preserved. Check the boundaries, then approve them; affected model predictions were cleared.`,
    );
  } catch (error) {
    status(error.message);
  }
};
function nextMissing(kind) {
  const well = selectedWell();
  if (!well || busy) return;
  for (let step = 1; step <= files.length; step++) {
    const frame = (index + step) % files.length;
    if (frame < C.analysisStart(project)) continue;
    if (
      kind === "boundary"
        ? C.boundaryNeedsReview(well, frame)
        : !project.records[C.key(frame, well.uid)]?.correction?.state
    ) {
      moveTo(frame);
      return;
    }
  }
  status(
    kind === "boundary"
      ? "Every boundary for this well is reviewed."
      : "Every frame for this well has a manual label.",
  );
}
$("nextBoundary").onclick = () => nextMissing("boundary");
$("nextLabel").onclick = () => nextMissing("label");
document.querySelector("nav").addEventListener("click", (e) => {
  if (busy) {
    e.preventDefault();
    status("Finish or cancel the current operation before changing studies.");
  }
});

function approveStable(uids) {
  if (busy || !uids.length) return;
  if (
    !confirm(
      "Approve these boundaries across the whole sequence based on stable well positions? Existing geometry and annotations are preserved. Frames with failed alignments remain pending.",
    )
  )
    return;
  const result = C.approveStableBoundaries(project, uids);
  updateDetails();
  draw();
  status(
    `Approved ${result.approved} well/frame boundaries across the sequence; ${result.skipped} failed-alignment boundaries remain pending. Later geometry edits will need review again.`,
  );
}
$("confirmSequence").onclick = () =>
  approveStable(project.wells.map((w) => w.uid));

function humanHTML(report) {
  const esc = (v) =>
    String(v ?? "").replace(
      /[&<>"']/g,
      (ch) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[ch],
    );
  const table = (rows) =>
    "<table><thead><tr>" +
    Object.keys(rows[0] || {})
      .map((k) => "<th>" + esc(k) + "</th>")
      .join("") +
    "</tr></thead><tbody>" +
    rows
      .map(
        (r) =>
          "<tr>" +
          Object.values(r)
            .map((v) => "<td>" + esc(v === "" ? "—" : v) + "</td>")
            .join("") +
          "</tr>",
      )
      .join("") +
    "</tbody></table>";
  return (
    '<!doctype html><meta charset="utf-8"><title>FlyScope report</title><style>body{font:16px system-ui;margin:2rem;color:#182332}table{border-collapse:collapse;margin:1rem 0}td,th{padding:.6rem;border:1px solid #aaa;text-align:left}th{background:#eee}</style><h1>' +
    esc(report.task.toUpperCase()) +
    " report</h1><p>Experiment: " +
    esc(report.experiment_id) +
    "</p>" +
    table(report.results) +
    table(report.summary) +
    report.definitions.map((d) => "<p>" + esc(d) + "</p>").join("") +
    "<p>Well positions: see well-locations.csv and numbered images in maps/. project.json is the editable backup; use Import report to restore it.</p>"
  );
}
$("machine").onclick = async () => {
  if (!files.length) return status("Load images first.");
  stop();
  setBusy(true);
  cancelled = false;
  try {
    const report = C.machineReport(project);
    const entries = [
      { name: "machine.json", text: JSON.stringify(report, null, 2) },
      {
        name: "project.json",
        text: JSON.stringify(C.exportProject(project), null, 2),
      },
      {
        name: "frames.csv",
        text: C.tableCSV(C.exportProject(project).results),
      },
      { name: "recovery.csv", text: C.tableCSV(report.recovery || []) },
      {
        name: "README.txt",
        text: "Machine analysis retains frame labels, geometry, model confidence/version, manual corrections, timing bounds and complete editable project/history. Use the separate human report for compact results. Training exports additionally contain original images and eligible manual targets; crops are derived by the loaders.",
      },
    ];
    download(
      `flyscope-${study}-machine.zip`,
      await TrainingExport.zip(
        entries,
        () => {},
        () => cancelled,
      ),
      "application/zip",
    );
    status("Machine analysis exported.");
  } catch (e) {
    status(e.message);
  } finally {
    setBusy(false);
    renderFrame();
  }
};
$("confirmSingles").onclick = () => {
  if (
    busy ||
    !confirm(
      "Confirm that every currently unchecked well contains exactly one individual? Empty, multiple and obscured wells remain unchanged.",
    )
  )
    return;
  const uids = [];
  project.wells.forEach((w) => {
    if (!w.occupancy || w.occupancy === "unknown") {
      w.occupancy = "single";
      uids.push(w.uid);
    }
  });
  history("confirm-single-occupancy", { uids });
  updateDetails();
  draw();
};
function setTimeOrigin() {
  const offset = Number($("recoveryDelay").value);
  if (!Number.isFinite(offset) || offset < 0)
    return status("Delay must be nonnegative.");
  project.time_origin = {
    confirmed: $("timeConfirmed").checked,
    offset_seconds: offset,
  };
  history("set-time-origin", project.time_origin);
  renderSummary();
}
$("timeConfirmed").onchange = $("recoveryDelay").onchange = setTimeOrigin;
$("eggScope").onchange = () => {
  if (
    $("eggScope").value === "image" &&
    project.wells.some((w) => !C.imageArea(w))
  ) {
    $("eggScope").value = "areas";
    status(
      "Keep custom areas visible while this project contains area annotations.",
    );
  }
  $("tool").value = "select";
  updateContext();
  draw();
};

$("datasetTest").onchange = updateContext;
$("cropDataset").onchange = updateContext;

document.querySelectorAll("[data-step],[data-go-step]").forEach((button) => {
  button.onclick = () => goToStep(button.dataset.step || button.dataset.goStep);
});
document.querySelectorAll("[data-next-step]").forEach((button) => {
  button.onclick = () => {
    const steps = workflowSteps();
    goToStep(steps[steps.indexOf(workflowStage) + 1]);
  };
});
$("workflowBack").onclick = () => {
  const steps = workflowSteps();
  goToStep(steps[steps.indexOf(workflowStage) - 1]);
};
$("saveDraft").onclick = () => $("json").onclick();
$("drawWell").onclick = () => {
  $("tool").value = "polygon";
  draft = [];
  canvas.focus();
  draw();
  status(
    "Click around the boundary, then double-click to finish. Escape cancels the unfinished polygon.",
  );
};
$("setRegion").onclick = () => {
  $("tool").value = "region";
  canvas.focus();
  status("Drag a rectangle around the area where wells should be detected.");
};

$("analysisStart").onchange = () => {
  const value = Number($("analysisStart").value) - 1;
  if (!Number.isInteger(value) || value < 0 || value >= files.length) {
    $("analysisStart").value = C.analysisStart(project) + 1;
    return;
  }
  const previous = C.analysisStart(project);
  project.analysis_start_index = value;
  if (value !== previous) {
    for (const w of project.wells) {
      if (
        Object.entries(project.records).some(
          ([k, r]) => k.endsWith(":" + w.uid) && r.model?.requires_sequence,
        )
      )
        invalidate(w.uid, null);
    }
  }
  history("set-analysis-start", {
    frame_index: value,
    frame_id: project.frames[value].id,
  });
  if (project.records && study === "ccrt") C.applyRecoveryPolicy(project);
  moveTo(value);
};
$("startHere").onclick = () => {
  $("analysisStart").value = index + 1;
  $("analysisStart").onchange();
};
$("inputKind").onchange = () => {
  project.input_kind = $("inputKind").value;
  if (project.input_kind === "trimmed") {
    $("timeConfirmed").checked = false;
    $("sourceFirstFrame").value = "";
    project.source_first_frame_number = null;
  }
  setTimeOrigin();
  updateDetails();
};
$("sourceFirstFrame").onchange = () => {
  const value = $("sourceFirstFrame").value
    ? Number($("sourceFirstFrame").value)
    : null;
  if (value !== null && (!Number.isInteger(value) || value < 1)) return;
  project.source_first_frame_number = value;
  history("set-source-first-frame", { value });
};
$("gridSpecies").onchange = () => {
  project.species = $("gridSpecies").value.trim();
  history("set-grid-species", { species: project.species });
  renderSummary();
};
$("wellSpecies").onchange = () => {
  const w = selectedWell();
  if (w) {
    w.species = $("wellSpecies").value.trim();
    history("set-well-species", { uid: w.uid, species: w.species });
    renderSummary();
  }
};
$("useModelPolicy").onchange =
  $("decisionThreshold").onchange =
  $("awakeFrames").onchange =
  $("requireComaBaseline").onchange =
  $("inferMatchingGaps").onchange =
    () => {
      try {
        const policy = selectedRecoveryPolicy();
        C.applyRecoveryPolicy(project, policy);
        project.recovery_policy = policy;
        history("set-recovery-policy", policy);
        updatePolicyControls();
        updateDetails();
        draw();
      } catch (e) {
        status(e.message);
      }
    };
