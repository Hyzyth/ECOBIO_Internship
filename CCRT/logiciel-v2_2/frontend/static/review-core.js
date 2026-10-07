/* Pure project operations shared by the UI and tests. */
(function (root) {
  "use strict";
  const natural = new Intl.Collator("en", {
    numeric: true,
    sensitivity: "base",
  });
  function timestamp(name) {
    const m = name.match(
      /(\d{4})-(\d{2})-(\d{2})[T_ ](\d{2})[-:](\d{2})[-:](\d{2})/,
    );
    return m ? `${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}` : null;
  }
  function orderFiles(files, order = "name") {
    const list = [...files];
    const allTimestamped = list.every((f) => timestamp(f.name));
    const filename = (a, b) =>
      natural.compare(a.name, b.name) ||
      natural.compare(
        a.webkitRelativePath || a.name,
        b.webkitRelativePath || b.name,
      );
    return list.sort((a, b) =>
      order === "modified"
        ? a.lastModified - b.lastModified || filename(a, b)
        : (allTimestamped
            ? timestamp(a.name).localeCompare(timestamp(b.name))
            : 0) || filename(a, b),
    );
  }
  function isSimplePolygon(points) {
    if (
      !Array.isArray(points) ||
      points.length < 3 ||
      !points.every(
        (p) =>
          Array.isArray(p) &&
          p.length === 2 &&
          p.every((v) => Number.isFinite(v) && v >= 0 && v <= 1),
      )
    )
      return false;
    const eps = 1e-10;
    const cross = (a, b, c) =>
      (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const on = (a, b, p) =>
      Math.abs(cross(a, b, p)) <= eps &&
      p[0] >= Math.min(a[0], b[0]) - eps &&
      p[0] <= Math.max(a[0], b[0]) + eps &&
      p[1] >= Math.min(a[1], b[1]) - eps &&
      p[1] <= Math.max(a[1], b[1]) + eps;
    const opposite = (a, b) => (a > eps && b < -eps) || (b > eps && a < -eps);
    const intersects = (a, b, c, d) =>
      (opposite(cross(a, b, c), cross(a, b, d)) &&
        opposite(cross(c, d, a), cross(c, d, b))) ||
      on(a, b, c) ||
      on(a, b, d) ||
      on(c, d, a) ||
      on(c, d, b);
    let area = 0;
    for (let i = 0; i < points.length; i++) {
      const a = points[i],
        b = points[(i + 1) % points.length],
        previous = points[(i + points.length - 1) % points.length];
      if (Math.hypot(a[0] - b[0], a[1] - b[1]) <= eps) return false;
      // Collinear forward edges are fine; retracing an adjacent edge is not.
      if (
        Math.abs(cross(previous, a, b)) <= eps &&
        (previous[0] - a[0]) * (b[0] - a[0]) +
          (previous[1] - a[1]) * (b[1] - a[1]) >
          eps
      )
        return false;
      area += a[0] * b[1] - b[0] * a[1];
      for (let j = i + 1; j < points.length; j++) {
        if (j === i + 1 || (i === 0 && j === points.length - 1)) continue;
        if (intersects(a, b, points[j], points[(j + 1) % points.length]))
          return false;
      }
    }
    return Math.abs(area) > eps;
  }
  function geometry(well, frame) {
    return well.overrides?.[frame] || well.points;
  }
  function effective(record, well) {
    if (well?.occupancy === "empty") return "empty";
    if (well?.occupancy === "multiple") return "invalid";
    return record?.correction?.state || record?.prediction || "unknown";
  }
  function key(frame, well) {
    return `${frame}:${well}`;
  }
  function summary(project) {
    return project.wells.map((well) => {
      let awake = 0,
        coma = 0,
        unknown = 0,
        empty = 0,
        invalid = 0,
        transitions = 0,
        previous = null,
        confidence = [];
      const states = project.frames.map((_, i) => {
        const r = project.records[key(i, well.uid)],
          s = effective(r, well);
        if (s === "awake") awake++;
        else if (s === "coma") coma++;
        else if (s === "empty") empty++;
        else if (s === "invalid") invalid++;
        else unknown++;
        if (previous === "coma" && s === "awake") transitions++;
        previous = s;
        if (!["empty", "invalid"].includes(s) && r?.confidence != null)
          confidence.push(r.confidence);
        return s;
      });
      return {
        well_id: well.id,
        total_frames: states.length,
        awake_frames: awake,
        coma_frames: coma,
        unknown_frames: unknown,
        empty_frames: empty,
        invalid_frames: invalid,
        occupancy: well.occupancy || "unknown",
        awake_seconds:
          project.mode === "sequence" ? awake * project.interval : null,
        coma_seconds:
          project.mode === "sequence" ? coma * project.interval : null,
        transitions: project.mode === "sequence" ? transitions : null,
        average_model_confidence: confidence.length
          ? confidence.reduce((a, b) => a + b, 0) / confidence.length
          : null,
        states,
      };
    });
  }
  function exportProject(project) {
    const rows = [];
    project.frames.forEach((frame, i) =>
      project.wells.forEach((well) => {
        if (!activeWell(well, i)) return;
        const record = project.records[key(i, well.uid)] || {};
        rows.push({
          frame_number: i + 1,
          frame_id: frame.id,
          filename: frame.name,
          timestamp: frame.timestamp,
          elapsed_seconds:
            project.mode === "sequence" ? i * project.interval : null,
          well_id: well.id,
          well_uid: well.uid,
          prediction: record.prediction || "unknown",
          confidence: record.confidence ?? null,
          model: record.model || null,
          state_probabilities: record.state_probabilities || null,
          occupancy_prediction: record.occupancy_prediction || null,
          occupancy_score: record.occupancy_score ?? null,
          occupancy_method: record.occupancy_method || null,
          effective_state: effective(
            record,
            taskOf(project) === "ccrt" ? well : null,
          ),
          user_correction: record.correction || null,
          egg_count: record.egg_count ?? null,
          effective_egg_count:
            record.correction?.egg_count ?? record.egg_count ?? null,
          well_boundary: geometry(well, i),
          detection: well.detection,
          boundary_partial:
            well.partial_frames?.[i] ?? !!well.detection.partial,
          occupancy: well.occupancy || "unknown",
          geometry_override: !!well.overrides?.[i],
          boundary_review_required:
            !!well.detection.review_required ||
            !!well.review_frames?.includes(i),
        });
      }),
    );
    return {
      ...project,
      schema_version: 1,
      exported_at: new Date().toISOString(),
      results: rows,
      task: taskOf(project),
      summary:
        taskOf(project) === "eggs" ? eggSummary(project) : summary(project),
    };
  }
  function csv(project) {
    const fields = [
      "frame_number",
      "frame_id",
      "filename",
      "timestamp",
      "elapsed_seconds",
      "well_id",
      "prediction",
      "confidence",
      "model",
      "effective_state",
      "user_correction",
      "egg_count",
      "effective_egg_count",
      "well_boundary",
      "detection",
      "occupancy",
      "geometry_override",
      "boundary_review_required",
      "boundary_partial",
    ];
    const escape = (v) =>
      `"${(v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v)).replaceAll('"', '""').replace(/^[=+@-]/, "'$&")}"`;
    return [
      fields.join(","),
      ...exportProject(project).results.map((r) =>
        fields.map((f) => escape(r[f])).join(","),
      ),
    ].join("\r\n");
  }
  function validateProject(p, frames) {
    if (
      p?.schema_version !== 1 ||
      (p.history != null && !Array.isArray(p.history)) ||
      (p.mode === "single" && p.frames?.length !== 1) ||
      !Array.isArray(p.wells) ||
      !Array.isArray(p.frames) ||
      !p.records ||
      typeof p.records !== "object" ||
      Array.isArray(p.records) ||
      !Number.isFinite(p.interval) ||
      p.interval <= 0 ||
      !["single", "sequence", "batch"].includes(p.mode) ||
      (p.task != null && !["ccrt", "eggs"].includes(p.task)) ||
      (p.mode === "batch" && p.task !== "eggs") ||
      (p.task === "eggs" && p.mode === "sequence")
    )
      throw Error("Unsupported or invalid project.");
    if (
      p.frames.length !== frames.length ||
      p.frames.some((f, i) => f.id !== frames[i].id)
    )
      throw Error(
        "Select the same image files in the saved order before restoring this project.",
      );
    if (
      p.detection_region != null &&
      (!Array.isArray(p.detection_region) ||
        p.detection_region.length !== 4 ||
        !p.detection_region.every(
          (v) => Number.isFinite(v) && v >= 0 && v <= 1,
        ) ||
        p.detection_region[0] >= p.detection_region[2] ||
        p.detection_region[1] >= p.detection_region[3])
    )
      throw Error("Invalid detection region.");
    if (
      p.time_origin != null &&
      (typeof p.time_origin.confirmed !== "boolean" ||
        !Number.isFinite(p.time_origin.offset_seconds) ||
        p.time_origin.offset_seconds < 0)
    )
      throw Error("Invalid recovery time origin.");
    if (p.detection_regions != null) {
      if (
        typeof p.detection_regions !== "object" ||
        Array.isArray(p.detection_regions)
      )
        throw Error("Invalid per-image detection regions.");
      for (const [i, region] of Object.entries(p.detection_regions)) {
        if (!/^\d+$/.test(i) || Number(i) >= frames.length)
          throw Error("Invalid per-image detection region index.");
        validateProject(
          { ...p, detection_regions: undefined, detection_region: region },
          frames,
        );
      }
    }
    if (p.alignments != null) {
      if (typeof p.alignments !== "object" || Array.isArray(p.alignments))
        throw Error("Invalid alignment metadata.");
      for (const [frame, a] of Object.entries(p.alignments))
        if (
          !/^\d+$/.test(frame) ||
          Number(frame) >= frames.length ||
          !a ||
          !["similarity", "affine", "perspective"].includes(a.method) ||
          !Number.isFinite(a.inlier_ratio) ||
          a.inlier_ratio < 0 ||
          a.inlier_ratio > 1 ||
          !Number.isFinite(a.median_error_px) ||
          a.median_error_px < 0
        )
          throw Error("Invalid alignment metadata.");
    }
    if (p.layout === "individual" && p.wells.length > 1)
      throw Error("Single-well layout permits at most one well.");
    const ids = new Set(),
      names = new Set();
    const validPoints = isSimplePolygon;
    p.wells.forEach((w) => {
      if (
        typeof w.uid !== "string" ||
        !w.uid ||
        w.uid.includes(":") ||
        ids.has(w.uid) ||
        typeof w.id !== "string" ||
        !w.id.trim() ||
        names.has(w.id) ||
        !validPoints(w.points) ||
        typeof w.overrides !== "object" ||
        !w.overrides ||
        Array.isArray(w.overrides) ||
        !w.detection ||
        typeof w.detection.source !== "string" ||
        (w.review_frames != null &&
          (!Array.isArray(w.review_frames) ||
            w.review_frames.some(
              (i) => !Number.isInteger(i) || i < 0 || i >= frames.length,
            )))
      )
        throw Error("Invalid well boundaries or duplicate identities.");
      if (
        w.occupancy != null &&
        !["unknown", "empty", "single", "multiple", "obscured"].includes(
          w.occupancy,
        )
      )
        throw Error("Invalid occupancy.");
      if (w.area_type != null && !["image", "region"].includes(w.area_type))
        throw Error("Invalid annotation area type.");
      ids.add(w.uid);
      names.add(w.id);
      if (
        w.partial_frames != null &&
        (typeof w.partial_frames !== "object" ||
          Array.isArray(w.partial_frames) ||
          Object.entries(w.partial_frames).some(
            ([i, value]) =>
              !/^\d+$/.test(i) ||
              Number(i) >= frames.length ||
              typeof value !== "boolean",
          ))
      )
        throw Error("Invalid partial boundary metadata.");
      if (
        w.active_frames != null &&
        (!Array.isArray(w.active_frames) ||
          !w.active_frames.length ||
          w.active_frames.some(
            (i) => !Number.isInteger(i) || i < 0 || i >= frames.length,
          ))
      )
        throw Error("Invalid region image membership.");
      Object.entries(w.overrides).forEach(([i, pts]) => {
        if (!/^\d+$/.test(i) || Number(i) >= frames.length || !validPoints(pts))
          throw Error("Invalid frame boundary.");
      });
    });
    Object.entries(p.records).forEach(([k, r]) => {
      const [i, uid] = k.split(":");
      if (
        !/^\d+$/.test(i) ||
        Number(i) >= frames.length ||
        !ids.has(uid) ||
        !r ||
        !["unknown", "coma", "awake", "empty", undefined].includes(
          r.prediction,
        ) ||
        (r.confidence != null &&
          (!Number.isFinite(r.confidence) ||
            r.confidence < 0 ||
            r.confidence > 1)) ||
        (r.correction &&
          (!["coma", "awake", "unknown", null].includes(r.correction.state) ||
            (r.correction.egg_count != null &&
              (!Number.isInteger(r.correction.egg_count) ||
                r.correction.egg_count < 0))))
      )
        throw Error("Invalid frame annotations.");
    });
    return p;
  }
  function activeWell(well, frame) {
    return !well.active_frames || well.active_frames.includes(frame);
  }
  function taskOf(project) {
    return project.task || "ccrt";
  }
  function boundaryNeedsReview(well, frame) {
    return (
      !!well.detection.review_required || !!well.review_frames?.includes(frame)
    );
  }
  function applyStateRange(project, uid, first, last, state, notes = "") {
    if (
      taskOf(project) !== "ccrt" ||
      !project.wells.some((w) => w.uid === uid) ||
      !Number.isInteger(first) ||
      !Number.isInteger(last) ||
      first < 0 ||
      last < first ||
      last >= project.frames.length ||
      !["coma", "awake", "unknown"].includes(state)
    )
      throw Error("Invalid CCRT annotation range.");
    const operation = {
      action: "label-range",
      operation_id: crypto.randomUUID(),
      at: new Date().toISOString(),
      uid,
      first_frame: first + 1,
      last_frame: last + 1,
      state,
      changes: [],
    };
    for (let i = first; i <= last; i++) {
      const k = key(i, uid),
        previous = project.records[k];
      operation.changes.push({
        key: k,
        previous: previous ? structuredClone(previous) : null,
      });
      project.records[k] = {
        ...previous,
        correction: {
          ...previous?.correction,
          state,
          notes,
          at: operation.at,
          operation_id: operation.operation_id,
          range: { first_frame: first + 1, last_frame: last + 1 },
        },
      };
    }
    project.history.push(operation);
    return operation;
  }
  function undoStateRange(project, operation) {
    let restored = 0;
    for (const change of operation.changes) {
      if (
        project.records[change.key]?.correction?.operation_id !==
        operation.operation_id
      )
        continue;
      const current = { ...project.records[change.key] };
      if (change.previous?.correction)
        current.correction = structuredClone(change.previous.correction);
      else delete current.correction;
      if (Object.keys(current).length) project.records[change.key] = current;
      else delete project.records[change.key];
      restored++;
    }
    project.history.push({
      action: "undo-label-range",
      at: new Date().toISOString(),
      operation_id: operation.operation_id,
      restored,
    });
    return restored;
  }
  function wellMap(project, frame = 0) {
    return project.wells
      .filter((w) => activeWell(w, frame))
      .map((w) => {
        const pts = geometry(w, frame),
          xs = pts.map((p) => p[0]),
          ys = pts.map((p) => p[1]);
        return {
          well_id: w.id,
          well_uid: w.uid,
          reference_frame: frame + 1,
          filename: project.frames[frame]?.name || null,
          x_percent: +((Math.min(...xs) + Math.max(...xs)) * 50).toFixed(2),
          y_percent: +((Math.min(...ys) + Math.max(...ys)) * 50).toFixed(2),
        };
      });
  }
  function eggSummary(project) {
    return project.frames.map((frame, i) => {
      const regions = project.wells.filter((w) => activeWell(w, i));
      const counts = regions
        .map((w) => {
          const r = project.records[key(i, w.uid)];
          return r?.correction?.egg_count ?? r?.egg_count ?? null;
        })
        .filter((c) => c != null);
      return {
        filename: frame.name,
        regions: regions.length,
        counted_regions: counts.length,
        uncounted_regions: regions.length - counts.length,
        total_counted_eggs: counts.length
          ? counts.reduce((a, c) => a + c, 0)
          : null,
      };
    });
  }
  function resultsReport(project) {
    const task = taskOf(project),
      locations =
        task === "eggs"
          ? project.frames.flatMap((_, i) => wellMap(project, i))
          : wellMap(project);
    const results = exportProject(project).results.map((r) =>
      task === "eggs"
        ? {
            image_number: r.frame_number,
            filename: r.filename,
            region_id: r.well_id,
            egg_count: r.effective_egg_count,
            source:
              r.user_correction?.egg_count != null
                ? "manual"
                : r.egg_count != null
                  ? "model"
                  : "unannotated",
            model: r.model,
            confidence: r.confidence,
            notes: r.user_correction?.notes || "",
            boundary_review_required: r.boundary_review_required,
            boundary_partial: r.boundary_partial,
          }
        : {
            frame_number: r.frame_number,
            filename: r.filename,
            timestamp: r.timestamp,
            elapsed_seconds: r.elapsed_seconds,
            well_id: r.well_id,
            state: r.effective_state,
            occupancy: r.occupancy,
            source: r.user_correction?.state
              ? "manual"
              : r.prediction !== "unknown"
                ? "model"
                : "unannotated",
            model_prediction: r.prediction,
            model: r.model,
            confidence: r.confidence,
            notes: r.user_correction?.notes || "",
            boundary_review_required: r.boundary_review_required,
            boundary_partial: r.boundary_partial,
          },
    );
    const summaries = task === "eggs" ? eggSummary(project) : summary(project);
    return {
      schema_version: 1,
      kind: `${task}-results`,
      experiment_id: project.experiment_id || null,
      task,
      mode: project.mode,
      frame_interval_seconds:
        project.mode === "sequence" ? project.interval : null,
      exported_at: new Date().toISOString(),
      well_map: locations,
      summary: summaries,
      results,
    };
  }
  function tableCSV(rows, fields = Object.keys(rows[0] || {})) {
    const escape = (v) =>
      `"${(v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v)).replaceAll('"', '""').replace(/^[=+@-]/, "'$&")}"`;
    return [
      fields.join(","),
      ...rows.map((row) => fields.map((f) => escape(row[f])).join(",")),
    ].join("\r\n");
  }
  function trainingAnnotations(project) {
    const task = taskOf(project),
      annotations = [];
    project.frames.forEach((frame, i) =>
      project.wells.forEach((w) => {
        if (!activeWell(w, i)) return;
        const correction = project.records[key(i, w.uid)]?.correction;
        const manual =
          task === "eggs"
            ? correction?.egg_count != null
            : w.occupancy === "empty" || !!correction?.state;
        if (!manual) return;
        const partial = w.partial_frames?.[i] ?? !!w.detection.partial,
          reviewed = !boundaryNeedsReview(w, i);
        const uncertain =
          task === "ccrt" &&
          w.occupancy !== "empty" &&
          correction?.state === "unknown";
        annotations.push({
          id: `f${i + 1}-${w.uid}`,
          frame_index: i,
          frame_number: i + 1,
          filename: frame.name,
          timestamp: frame.timestamp,
          elapsed_seconds:
            project.mode === "sequence" ? i * project.interval : null,
          well_id: w.id,
          well_uid: w.uid,
          split_group: project.experiment_id,
          task,
          label:
            task === "eggs"
              ? correction.egg_count
              : w.occupancy === "empty"
                ? "empty"
                : correction.state,
          annotation: structuredClone(correction || {}),
          label_source:
            task === "ccrt" && w.occupancy === "empty"
              ? "manual_occupancy"
              : "manual",
          boundary_normalized: geometry(w, i),
          boundary_reviewed: reviewed,
          boundary_partial: partial,
          occupancy: w.occupancy || "unknown",
          annotation_unit:
            task === "eggs" ? (imageArea(w) ? "image" : "region") : "well",
          annotation_type: task === "eggs" ? "count" : "state",
          exclusion_reasons: [
            !reviewed ? "boundary_unreviewed" : null,
            uncertain ? "uncertain_state" : null,
            partial ? "partial_boundary" : null,
            task === "ccrt" && !["single", "empty"].includes(w.occupancy)
              ? "occupancy_not_confirmed_single_or_empty"
              : null,
          ].filter(Boolean),
          eligible_for_training:
            reviewed &&
            !uncertain &&
            !partial &&
            !(task === "ccrt" && !["single", "empty"].includes(w.occupancy)),
          state_training_eligible:
            task === "ccrt" &&
            reviewed &&
            !uncertain &&
            !partial &&
            w.occupancy === "single",
        });
      }),
    );
    return annotations;
  }

  function approveStableBoundaries(project, uids) {
    let approved = 0,
      skipped = 0;
    for (const w of project.wells.filter((w) => uids.includes(w.uid))) {
      const failures = project.frames
        .map((_, i) => i)
        .filter(
          (i) =>
            activeWell(w, i) &&
            project.alignment_failures?.[i] &&
            boundaryNeedsReview(w, i),
        );
      w.detection.review_required = false;
      w.review_frames = failures;
      for (let i = 0; i < project.frames.length; i++)
        if (activeWell(w, i)) failures.includes(i) ? skipped++ : approved++;
    }
    project.history.push({
      action: "approve-stable-sequence-boundaries",
      at: new Date().toISOString(),
      well_uids: uids,
      approved,
      skipped,
    });
    return { approved, skipped };
  }
  function importReadableReport(project, report) {
    if (
      report?.kind !== `${taskOf(project)}-results` ||
      !Array.isArray(report.results) ||
      !Array.isArray(report.well_map)
    )
      throw Error("Select a matching CCRT or egg-count report.");
    if (!project.wells.length)
      throw Error(
        "This older report has annotations but no boundaries. Detect or draw the wells first, then import it again; the saved labels will be restored.",
      );
    const copy = structuredClone(project),
      seen = new Set();
    if (report.mode) copy.mode = report.mode;
    if (
      Number.isFinite(report.frame_interval_seconds) &&
      report.frame_interval_seconds > 0
    )
      copy.interval = report.frame_interval_seconds;
    if (typeof report.experiment_id === "string" && report.experiment_id)
      copy.experiment_id = report.experiment_id;
    const baseName = (name) =>
      String(name).replaceAll("\\", "/").split("/").at(-1);
    let imported = 0;
    for (const row of report.results) {
      const frame = (row.frame_number ?? row.image_number) - 1;
      if (
        !Number.isInteger(frame) ||
        frame < 0 ||
        frame >= copy.frames.length ||
        (row.filename !== copy.frames[frame].name &&
          (baseName(row.filename) !== baseName(copy.frames[frame].name) ||
            copy.frames.filter(
              (f) => baseName(f.name) === baseName(row.filename),
            ).length !== 1))
      )
        throw Error(
          "Report images do not match the selected images in their saved order.",
        );
      const id = row.well_id ?? row.region_id,
        location = report.well_map.find(
          (m) =>
            m.well_id === id &&
            (taskOf(copy) === "ccrt" || m.reference_frame === frame + 1),
        );
      const well = copy.wells.find((w) => w.id === id && activeWell(w, frame));
      if (!well)
        throw Error(
          `Define or rename the matching well ${id} before importing. No annotations were changed.`,
        );
      if (
        location &&
        Number.isFinite(location.x_percent) &&
        Number.isFinite(location.y_percent)
      ) {
        const pts = geometry(well, taskOf(copy) === "ccrt" ? 0 : frame),
          xs = pts.map((p) => p[0]),
          ys = pts.map((p) => p[1]);
        const cx = (Math.min(...xs) + Math.max(...xs)) / 2,
          cy = (Math.min(...ys) + Math.max(...ys)) / 2;
        if (
          Math.hypot(
            cx - location.x_percent / 100,
            cy - location.y_percent / 100,
          ) >
          0.75 *
            Math.max(
              Math.max(...xs) - Math.min(...xs),
              Math.max(...ys) - Math.min(...ys),
            )
        )
          throw Error(
            `Well ${id} is at a different position. Check its identity/boundary before importing. No annotations were changed.`,
          );
      }
      const k = key(frame, well.uid);
      if (seen.has(k)) throw Error("Duplicate report annotations.");
      seen.add(k);
      const record = {
        prediction: row.model_prediction || "unknown",
        confidence: row.confidence ?? null,
        model: row.model || null,
      };
      if (taskOf(copy) === "eggs") {
        if (row.source === "manual")
          record.correction = {
            state: null,
            egg_count: row.egg_count,
            notes: row.notes || "",
            imported_from: "readable-report",
            at: report.exported_at || new Date().toISOString(),
          };
        else record.egg_count = row.egg_count ?? null;
      } else {
        if (row.source === "manual" && row.state !== "empty")
          record.correction = {
            state: row.state === "uncertain" ? "unknown" : row.state,
            notes: row.notes || "",
            imported_from: "readable-report",
            at: report.exported_at || new Date().toISOString(),
          };
        if (row.source === "model" && row.state !== "empty")
          record.prediction = row.model_prediction || row.state;
      }
      if (row.occupancy) well.occupancy = row.occupancy;
      if (row.state === "empty") well.occupancy = "empty";
      copy.records[k] = record;
      imported++;
    }
    copy.history.push({
      action: "import-readable-report",
      at: new Date().toISOString(),
      imported,
      source_exported_at: report.exported_at,
    });
    validateProject(exportProject(copy), copy.frames);
    return { project: copy, imported };
  }

  function imageArea(well) {
    return (
      well.area_type === "image" ||
      (!well.area_type &&
        well.points.length === 4 &&
        well.points.every(
          (p) => (p[0] === 0 || p[0] === 1) && (p[1] === 0 || p[1] === 1),
        ))
    );
  }
  function recoveryMetrics(project, well) {
    const states = project.frames.map((_, i) =>
      effective(project.records[key(i, well.uid)], well),
    );
    const occupancy = well.occupancy || "unknown",
      origin = project.time_origin || { confirmed: true, offset_seconds: 0 },
      sequence = project.mode === "sequence",
      interval = project.interval;
    const result = {
      well_id: well.id,
      occupancy,
      state: states[0] || "unknown",
      coma_seconds: null,
      awake_seconds: null,
      uncertain_seconds: null,
      first_awake_frame: null,
      recovery_after_first_image_seconds: null,
      ccrt_seconds: null,
      recovery_lower_seconds: null,
      recovery_upper_seconds: null,
      status: "Occupancy unconfirmed",
    };
    if (occupancy === "empty") {
      result.status = "Empty";
      return result;
    }
    if (occupancy === "multiple") {
      result.status = "Invalid: more than one individual";
      return result;
    }
    if (occupancy === "obscured") {
      result.status = "Cannot verify occupancy";
      return result;
    }
    if (occupancy !== "single") return result;
    if (!sequence) {
      result.status = "Single image: no recovery timing";
      return result;
    }
    // Sample-and-hold integration over the observed window, never beyond the
    // final image. Unknown intervals remain unallocated to coma or awake.
    result.coma_seconds =
      states.slice(0, -1).filter((s) => s === "coma").length * interval;
    result.awake_seconds =
      states.slice(0, -1).filter((s) => s === "awake").length * interval;
    result.uncertain_seconds =
      states.slice(0, -1).filter((s) => s === "unknown").length * interval;
    const first = states.indexOf("awake");
    if (first < 0) {
      result.status = states.every((s) => s === "coma")
        ? "Not recovered during observation"
        : "Recovery not observed / incomplete labels";
      return result;
    }
    result.first_awake_frame = first + 1;
    result.recovery_after_first_image_seconds = first * interval;
    if (first === 0) {
      result.status = "Already awake at first image (left-censored)";
      return result;
    }
    let lastComa = -1;
    for (let i = 0; i < first; i++) if (states[i] === "coma") lastComa = i;
    if (lastComa < 0) {
      result.status = "No preceding coma observation";
      return result;
    }
    const delay = origin.confirmed ? origin.offset_seconds || 0 : 0;
    result.recovery_lower_seconds = delay + lastComa * interval;
    result.recovery_upper_seconds = delay + first * interval;
    if (states.slice(first + 1).includes("coma")) {
      result.status = "Review: coma after an awake label";
      return result;
    }
    if (!origin.confirmed) {
      result.status = "Confirm experimental time zero";
      return result;
    }
    if (lastComa !== first - 1) {
      result.status = "Uncertain recovery interval";
      return result;
    }
    result.ccrt_seconds = result.recovery_upper_seconds;
    result.status = "Observed (sampled)";
    return result;
  }
  function humanReport(project) {
    const task = taskOf(project),
      fmt = (v) => (v == null ? "" : Math.round(v * 1000) / 1000);
    let results, summary;
    if (task === "eggs") {
      results = eggSummary(project).map((s) => ({
        Image: s.filename,
        "Egg count": s.total_counted_eggs ?? "",
        "Counted areas": s.counted_regions,
        "Uncounted areas": s.uncounted_regions,
        Status: s.uncounted_regions ? "Incomplete counts" : "Counted",
      }));
      summary = [
        { Measure: "Images", Value: project.frames.length },
        {
          Measure: "Annotation format",
          Value: "Independent whole-image counts; optional custom areas",
        },
        {
          Measure: "Experimental layout",
          Value: "Not specified; no well layout assumed",
        },
      ];
    } else {
      const measures = project.wells.map((w) => recoveryMetrics(project, w));
      results = measures.map((m) => ({
        Well: m.well_id,
        Occupancy: {
          empty: "Empty",
          single: "Not empty (1 individual)",
          multiple: "INVALID (>1 individual)",
          obscured: "Obscured",
          unknown: "Not checked",
        }[m.occupancy],
        "Coma (s)": fmt(m.coma_seconds),
        "Awake (s)": fmt(m.awake_seconds),
        "Uncertain (s)": fmt(m.uncertain_seconds),
        "CCRT (s)": fmt(m.ccrt_seconds),
        Status: m.status,
      }));
      if (project.mode === "single")
        results = measures.map((m) => ({
          Well: m.well_id,
          Occupancy: {
            empty: "Empty",
            single: "Not empty (1 individual)",
            multiple: "INVALID (>1 individual)",
            obscured: "Obscured",
            unknown: "Not checked",
          }[m.occupancy],
          State: m.state,
          Status: m.status,
        }));
      const valid = measures
          .filter((m) => m.ccrt_seconds != null)
          .map((m) => m.ccrt_seconds)
          .sort((a, b) => a - b),
        n = valid.length;
      summary = [
        { Measure: "Wells", Value: project.wells.length },
        {
          Measure: "Empty",
          Value: measures.filter((m) => m.occupancy === "empty").length,
        },
        {
          Measure: "Confirmed single individual",
          Value: measures.filter((m) => m.occupancy === "single").length,
        },
        {
          Measure: "Invalid: multiple individuals",
          Value: measures.filter((m) => m.occupancy === "multiple").length,
        },
        {
          Measure: "Occupancy not verified",
          Value: measures.filter((m) =>
            ["unknown", "obscured"].includes(m.occupancy),
          ).length,
        },
        { Measure: "Images", Value: project.frames.length },
        {
          Measure: "Frame interval (s)",
          Value:
            project.mode === "sequence" ? project.interval : "Not applicable",
        },
        {
          Measure: "Time zero confirmed",
          Value: (project.time_origin?.confirmed ?? true) ? "Yes" : "No",
        },
        {
          Measure: "Cold end to first image (s)",
          Value:
            (project.time_origin?.confirmed ?? true)
              ? project.time_origin?.offset_seconds || 0
              : "Not confirmed",
        },
        { Measure: "Valid sampled CCRT measurements", Value: n },
        {
          Measure: "Median sampled CCRT (s)",
          Value: n
            ? fmt(
                n % 2
                  ? valid[(n - 1) / 2]
                  : (valid[n / 2 - 1] + valid[n / 2]) / 2,
              )
            : "",
        },
      ];
    }
    if (task === "ccrt" && project.mode !== "sequence")
      summary = summary.filter(
        (r) =>
          ![
            "Frame interval (s)",
            "Time zero confirmed",
            "Cold end to first image (s)",
            "Valid sampled CCRT measurements",
            "Median sampled CCRT (s)",
          ].includes(r.Measure),
      );
    return {
      schema_version: 1,
      kind: `${task}-human-report`,
      task,
      experiment_id: project.experiment_id || null,
      results,
      summary,
      definitions:
        task === "ccrt"
          ? [
              "One individual per well is required. Empty, multiple and unverified occupancy do not yield recovery measurements.",
              "CCRT is the first labelled awake time following a coma observation, relative to confirmed cold-exposure end. The image interval limits precision. Unknown transition gaps and later coma labels require review; a missing CCRT is not zero.",
              "Coma/awake durations use the state at the start of each sampling interval, from the first through last image; no duration is invented after the last image.",
              "Already-awake and unrecovered wells are censored, not exact CCRT measurements. Machine exports retain numeric bounds, frame-level labels and provenance.",
            ]
          : [
              "Egg counts are provisional independent-image annotations; no wells, temporal relation or final experimental layout are assumed.",
              "Count custom areas without overlap if using area counts. Zero is distinct from not counted.",
            ],
    };
  }
  function machineReport(project) {
    const report = resultsReport(project);
    return {
      ...report,
      kind: `${taskOf(project)}-machine-report`,
      summary:
        taskOf(project) === "ccrt"
          ? project.wells.map((w) => recoveryMetrics(project, w))
          : report.summary,
      duration_method:
        "Sample-and-hold over first-to-last observation, excluding time after the last image",
      project: exportProject(project),
      time_origin: project.time_origin || {
        confirmed: true,
        offset_seconds: 0,
      },
      recovery:
        taskOf(project) === "ccrt"
          ? project.wells.map((w) => recoveryMetrics(project, w))
          : null,
      annotation_format:
        taskOf(project) === "eggs"
          ? "count (provisional); image or optional region"
          : "well state; exactly one individual required",
    };
  }

  const api = {
    imageArea,
    recoveryMetrics,
    humanReport,
    machineReport,
    approveStableBoundaries,
    importReadableReport,
    activeWell,
    taskOf,
    boundaryNeedsReview,
    applyStateRange,
    undoStateRange,
    wellMap,
    resultsReport,
    tableCSV,
    trainingAnnotations,
    timestamp,
    orderFiles,
    geometry,
    isSimplePolygon,
    effective,
    key,
    summary,
    exportProject,
    csv,
    validateProject,
  };
  if (typeof module !== "undefined") module.exports = api;
  else root.ReviewCore = api;
})(globalThis);
