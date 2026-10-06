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
  function geometry(well, frame) {
    return well.overrides?.[frame] || well.points;
  }
  function effective(record) {
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
        transitions = 0,
        previous = null,
        confidence = [];
      const states = project.frames.map((_, i) => {
        const r = project.records[key(i, well.uid)],
          s = effective(r);
        if (s === "awake") awake++;
        else if (s === "coma") coma++;
        else unknown++;
        if (previous === "coma" && s === "awake") transitions++;
        previous = s;
        if (r?.confidence != null) confidence.push(r.confidence);
        return s;
      });
      return {
        well_id: well.id,
        total_frames: states.length,
        awake_frames: awake,
        coma_frames: coma,
        unknown_frames: unknown,
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
          effective_state: effective(record),
          user_correction: record.correction || null,
          egg_count: record.egg_count ?? null,
          effective_egg_count:
            record.correction?.egg_count ?? record.egg_count ?? null,
          well_boundary: geometry(well, i),
          detection: well.detection,
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
      summary: summary(project),
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
      !["single", "sequence"].includes(p.mode)
    )
      throw Error("Unsupported or invalid project.");
    if (
      p.frames.length !== frames.length ||
      p.frames.some((f, i) => f.id !== frames[i].id)
    )
      throw Error(
        "Select the same image files in the saved order before restoring this project.",
      );
    const ids = new Set(),
      names = new Set();
    const validPoints = (points) =>
      Array.isArray(points) &&
      points.length >= 3 &&
      points.every(
        (pt) =>
          Array.isArray(pt) &&
          pt.length === 2 &&
          pt.every((v) => Number.isFinite(v) && v >= 0 && v <= 1),
      );
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
      ids.add(w.uid);
      names.add(w.id);
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
        !["unknown", "coma", "awake", undefined].includes(r.prediction) ||
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
  const api = {
    timestamp,
    orderFiles,
    geometry,
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
