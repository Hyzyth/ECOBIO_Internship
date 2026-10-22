/* Local ZIP and dataset export. Original pixels never come from UI screenshots. */
(function (root) {
  "use strict";
  const encoder = new TextEncoder();
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    for (let i = 0; i < 8; i++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1;
    return n >>> 0;
  });
  function crc32(bytes) {
    let crc = 0xffffffff;
    for (const byte of bytes) crc = crcTable[(crc ^ byte) & 255] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
  }
  async function zip(entries, onProgress = () => {}, cancelled = () => false) {
    if (entries.length > 65535)
      throw Error(
        "Dataset exceeds ZIP entry limits. Export smaller image collections.",
      );
    const parts = [],
      central = [];
    let offset = 0,
      centralSize = 0;
    for (const [entryIndex, entry] of entries.entries()) {
      if (cancelled()) throw Error("Archive export cancelled.");
      if (
        !/^[\w./-]+$/.test(entry.name) ||
        entry.name.includes("..") ||
        entry.name.startsWith("/")
      )
        throw Error("Invalid archive path.");
      const name = encoder.encode(entry.name),
        blob = entry.blob || new Blob([entry.text]);
      const crc = await blobCRC(blob, cancelled),
        size = blob.size;
      if (size > 0xffffffff || offset + 30 + name.length + size > 0xffffffff)
        throw Error(
          "Dataset exceeds 4 GB ZIP limits. Export smaller image collections.",
        );
      const local = new Uint8Array(30),
        v = new DataView(local.buffer);
      v.setUint32(0, 0x04034b50, true);
      v.setUint16(4, 20, true);
      v.setUint16(6, 0x800, true);
      v.setUint16(12, 33, true);
      v.setUint32(14, crc, true);
      v.setUint32(18, size, true);
      v.setUint32(22, size, true);
      v.setUint16(26, name.length, true);
      parts.push(local, name, blob);
      const header = new Uint8Array(46),
        c = new DataView(header.buffer);
      c.setUint32(0, 0x02014b50, true);
      c.setUint16(4, 20, true);
      c.setUint16(6, 20, true);
      c.setUint16(8, 0x800, true);
      c.setUint16(14, 33, true);
      c.setUint32(16, crc, true);
      c.setUint32(20, size, true);
      c.setUint32(24, size, true);
      c.setUint16(28, name.length, true);
      c.setUint32(42, offset, true);
      central.push(header, name);
      centralSize += header.length + name.length;
      offset += local.length + name.length + size;
      onProgress(entryIndex + 1, entries.length);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    if (offset + centralSize + 22 > 0xffffffff)
      throw Error("Dataset exceeds ZIP limits.");
    const end = new Uint8Array(22),
      v = new DataView(end.buffer);
    v.setUint32(0, 0x06054b50, true);
    v.setUint16(8, entries.length, true);
    v.setUint16(10, entries.length, true);
    v.setUint32(12, centralSize, true);
    v.setUint32(16, offset, true);
    return new Blob([...parts, ...central, end], { type: "application/zip" });
  }
  async function blobCRC(blob, cancelled) {
    const reader = blob.stream().getReader();
    let crc = 0xffffffff,
      chunks = 0;
    try {
      for (;;) {
        if (cancelled()) throw Error("Archive export cancelled.");
        const { value, done } = await reader.read();
        if (done) return (crc ^ 0xffffffff) >>> 0;
        for (const byte of value)
          crc = crcTable[(crc ^ byte) & 255] ^ (crc >>> 8);
        if (++chunks % 16 === 0)
          await new Promise((resolve) => setTimeout(resolve, 0));
      }
    } finally {
      await reader.cancel();
    }
  }
  async function writeEntry(directory, entry, cancelled) {
    if (cancelled())
      throw Error(
        "Dataset export cancelled; the new folder is incomplete. Export again into a new folder.",
      );
    const segments = entry.name.split("/");
    let parent = directory;
    for (const segment of segments.slice(0, -1))
      parent = await parent.getDirectoryHandle(segment, { create: true });
    const handle = await parent.getFileHandle(segments.at(-1), {
      create: true,
    });
    const writer = await handle.createWritable();
    try {
      const blob = entry.blob || new Blob([entry.text]);
      const reader = blob.stream().getReader();
      try {
        for (;;) {
          if (cancelled())
            throw Error(
              "Dataset export cancelled; the new folder is incomplete.",
            );
          const { value, done } = await reader.read();
          if (done) break;
          await writer.write(value);
        }
      } finally {
        await reader.cancel();
      }
      await writer.close();
    } catch (error) {
      try {
        await writer.abort();
      } catch {
        /* Preserve the original write error. */
      }
      throw error;
    }
  }
  async function zipParts(
    entries,
    onProgress = () => {},
    cancelled = () => false,
    partBytes = 256 * 1024 * 1024,
  ) {
    if (
      !Number.isFinite(partBytes) ||
      partBytes < 128 ||
      partBytes > 512 * 1024 * 1024
    )
      throw Error("Invalid ZIP part size.");
    const parts = [];
    let group = [],
      bytes = 22,
      complete = 0;
    async function flush() {
      if (!group.length) return;
      const current = group;
      parts.push(
        await zip(
          current,
          (done) => onProgress(complete + done, entries.length),
          cancelled,
        ),
      );
      complete += current.length;
      group = [];
      bytes = 22;
    }
    for (const entry of entries) {
      const blob = entry.blob || new Blob([entry.text]),
        size = blob.size + 76 + 2 * encoder.encode(entry.name).length;
      if (size + 22 > 512 * 1024 * 1024)
        throw Error(
          "One metadata file exceeds the portable ZIP limit. Use Save dataset to folder for this experiment.",
        );
      if (bytes + size > partBytes || group.length >= 60000) await flush();
      group.push({ name: entry.name, blob });
      bytes += size;
    }
    await flush();
    return parts;
  }
  function png(canvas) {
    return new Promise((resolve, reject) =>
      canvas.toBlob(
        (blob) =>
          blob ? resolve(blob) : reject(Error("Could not encode image.")),
        "image/png",
      ),
    );
  }
  async function decode(file) {
    const url = URL.createObjectURL(file),
      image = new Image();
    image.src = url;
    try {
      await image.decode();
      return {
        image,
        release: () => {
          URL.revokeObjectURL(url);
          image.src = "";
        },
      };
    } catch (error) {
      URL.revokeObjectURL(url);
      throw error;
    }
  }
  async function locationMap(project, files, frame = 0) {
    const { image, release } = await decode(files[frame]);
    try {
      const canvas = document.createElement("canvas"),
        scale = Math.min(1, 1400 / image.width);
      canvas.width = Math.round(image.width * scale);
      canvas.height = Math.round(image.height * scale);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      for (const well of project.wells.filter((w) =>
        ReviewCore.activeWell(w, frame),
      )) {
        const points = ReviewCore.geometry(well, frame);
        ctx.beginPath();
        points.forEach((p, i) =>
          i
            ? ctx.lineTo(p[0] * canvas.width, p[1] * canvas.height)
            : ctx.moveTo(p[0] * canvas.width, p[1] * canvas.height),
        );
        ctx.closePath();
        ctx.strokeStyle = "#07141c";
        ctx.lineWidth = 6;
        ctx.stroke();
        ctx.strokeStyle = "#ffe000";
        ctx.lineWidth = 3;
        ctx.stroke();
        const x = Math.min(...points.map((p) => p[0])) * canvas.width,
          y = Math.min(...points.map((p) => p[1])) * canvas.height;
        ctx.font = "bold 16px sans-serif";
        ctx.fillStyle = "#07141c";
        ctx.fillRect(x, y, ctx.measureText(well.id).width + 10, 22);
        ctx.fillStyle = "#ffe000";
        ctx.fillText(well.id, x + 5, y + 17);
      }
      return await png(canvas);
    } finally {
      release();
    }
  }
  async function training(
    project,
    files,
    onProgress = () => {},
    cancelled = () => false,
    options = {},
  ) {
    const role = options.role || "train";
    if (!["train", "test"].includes(role)) throw Error("Invalid dataset role.");
    const annotations = ReviewCore.trainingAnnotations(project, {
      compact: true,
    });
    if (!annotations.length)
      throw Error(
        "Add manual state labels or egg counts before exporting training data.",
      );
    const entries = [],
      images = [],
      task = ReviewCore.taskOf(project),
      frames = ReviewCore.analysisIndices(project);
    let roi = null;
    const cropEnabled = task === "ccrt" && options.cropToWells;
    const cropEncoding = options.cropEncoding || "png";
    if (cropEnabled && !["png", "jpeg"].includes(cropEncoding))
      throw Error("Invalid crop encoding.");
    const hash = async (blob) =>
      Array.from(
        new Uint8Array(
          await crypto.subtle.digest("SHA-256", await blob.arrayBuffer()),
        ),
        (b) => b.toString(16).padStart(2, "0"),
      ).join("");
    // One image per frame. All model loaders derive masked crops lazily.
    // Work is linear in frames + annotations, never frames × annotations.
    for (const [n, i] of frames.entries()) {
      if (cancelled()) throw Error("Dataset export cancelled.");
      const file = files[i];
      let ext =
        file.name.match(/\.(png|jpe?g|bmp|webp)$/i)?.[1].toLowerCase() || "bin";
      const { image, release } = await decode(file);
      try {
        const sourceSha = await hash(file);
        let blob = file,
          width = image.width,
          height = image.height,
          spatial = null;
        if (cropEnabled) {
          roi ||= SpatialExport.bounds(
            project,
            width,
            height,
            options.cropPadding ?? 32,
          );
          const [x, y, w, h] = [
            roi[0] * width,
            roi[1] * height,
            roi[2] * width,
            roi[3] * height,
          ];
          const canvas = document.createElement("canvas");
          canvas.width = Math.max(1, Math.round(w));
          canvas.height = Math.max(1, Math.round(h));
          canvas
            .getContext("2d")
            .drawImage(image, x, y, w, h, 0, 0, canvas.width, canvas.height);
          blob = await new Promise((resolve, reject) =>
            canvas.toBlob(
              (v) =>
                v ? resolve(v) : reject(Error("Image crop encoding failed.")),
              `image/${cropEncoding}`,
              0.98,
            ),
          );
          ext = cropEncoding === "jpeg" ? "jpg" : "png";
          spatial = {
            source_x: x,
            source_y: y,
            source_width: w,
            source_height: h,
            original_width: width,
            original_height: height,
            encoding: cropEncoding,
            resampled: [x, y, w, h].some(
              (v) => Math.abs(v - Math.round(v)) > 1e-6,
            ),
          };
          width = canvas.width;
          height = canvas.height;
        }
        const imagePath = `images/frame-${String(i + 1).padStart(6, "0")}.${ext}`;
        images.push({
          ...ReviewCore.frameTiming(project, i),
          frame_index: i,
          original_filename: project.frames[i].name,
          timestamp: project.frames[i].timestamp,
          path: imagePath,
          width,
          height,
          sha256: spatial ? await hash(blob) : sourceSha,
          source_sha256: sourceSha,
          source_size_bytes: file.size,
          exported_size_bytes: blob.size,
          ...(spatial ? { spatial_crop: spatial } : {}),
        });
        if (options.directory)
          await writeEntry(
            options.directory,
            { name: imagePath, blob },
            cancelled,
          );
        else entries.push({ name: imagePath, blob });
      } finally {
        release();
      }
      onProgress(
        n + 1,
        frames.length,
        options.directory ? "saving-images" : "images",
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    onProgress(0, 1, "metadata");
    await new Promise((resolve) => setTimeout(resolve, 0));
    const manifest = {
      schema_version: 3,
      geometry_format: "shared-well-boundaries-v1",
      completion_marker: "dataset-complete.json",
      crop_storage: "derived-on-load",
      spatial_export: roi
        ? {
            normalized_region: roi,
            padding_pixels: options.cropPadding ?? 32,
            reference_width: images[0].spatial_crop.original_width,
            reference_height: images[0].spatial_crop.original_height,
            encoding: cropEncoding,
          }
        : null,
      role,
      kind: `${task}-${role === "test" ? "test" : "training"}`,
      task,
      mode: project.mode,
      frame_interval_seconds:
        project.mode === "sequence" ? project.interval : null,
      experiment_id: project.experiment_id,
      split_group: project.experiment_id,
      created_at: new Date().toISOString(),
      classes: task === "ccrt" ? { coma: 0, awake: 1, empty: 2 } : null,
      target:
        task === "ccrt"
          ? "state_classification"
          : annotations.every((a) => a.annotation_unit === "image")
            ? "image_egg_count"
            : "region_egg_count",
      expected_individuals_per_well: task === "ccrt" ? 1 : null,
      time_origin:
        task === "ccrt"
          ? project.time_origin || { confirmed: true, offset_seconds: 0 }
          : null,
      annotation_format:
        task === "eggs"
          ? "Provisional integer count; independent image or optional region"
          : "Per-frame well state",
      analysis_start_index: ReviewCore.analysisStart(project),
      input_kind: project.input_kind || "raw",
      source_first_frame_number:
        project.source_first_frame_number === null
          ? null
          : project.source_first_frame_number || 1,
      species: project.species || null,
      recovery_policy: project.recovery_policy || {
        confidence_threshold: 0.8,
        awake_consecutive_frames: 3,
      },
      images,
      annotations,
      wells: project.wells.map((w) => ({
        well_id: w.id,
        well_uid: w.uid,
        occupancy: w.occupancy || "unknown",
        species:
          w.occupancy === "empty" ? null : w.species || project.species || null,
        points: roi ? SpatialExport.transform(w.points, roi) : w.points,
        overrides: roi
          ? Object.fromEntries(
              Object.entries(w.overrides || {}).map(([i, points]) => [
                i,
                SpatialExport.transform(points, roi),
              ]),
            )
          : w.overrides || {},
        active_frames: w.active_frames || null,
        review_required: !!w.detection.review_required,
        review_frames: w.review_frames || [],
        partial: !!w.detection.partial,
        partial_frames: w.partial_frames || {},
        annotation_unit:
          task === "eggs"
            ? ReviewCore.imageArea(w)
              ? "image"
              : "region"
            : "well",
      })),
      coverage: {
        detected_wells: project.wells.length,
        expected_samples: project.frames.reduce(
          (n, _, i) =>
            n +
            (i < ReviewCore.analysisStart(project)
              ? 0
              : project.wells.filter((w) => ReviewCore.activeWell(w, i))
                  .length),
          0,
        ),
        annotated_samples: annotations.length,
        eligible_samples: annotations.filter((a) => a.eligible_for_training)
          .length,
        annotated_wells: new Set(annotations.map((a) => a.well_uid)).size,
        eligible_wells: new Set(
          annotations
            .filter((a) => a.eligible_for_training)
            .map((a) => a.well_uid),
        ).size,
      },
      all_frame_count: project.frames.length,
      analysis_frame_count: ReviewCore.analysisIndices(project).length,
      annotated_sample_count: annotations.length,
      eligible_sample_count: annotations.filter((a) => a.eligible_for_training)
        .length,
      notes: [
        "Spatial export metadata maps cropped coordinates back to the original source image. Source hashes preserve train/test overlap checks across different crops. Project backup coordinates remain original.",
        "Targets come exclusively from manual annotations. Unannotated frame/wells are omitted.",
        "Uncertain, partial, unreviewed, and CCRT multiple/obscured/unconfirmed occupancy samples are retained with eligibility=false. Explicit Empty occupancy is a manual occupancy target, not a coma/awake label.",
        "Split related experiments together. Do not randomly split adjacent frames or wells from the same experiment.",
        "Egg targets are provisional independent-image or optional region counts; no well layout is assumed.",
      ],
    };
    const manifestText = JSON.stringify(manifest);
    entries.push({ name: "annotations.json", text: manifestText });
    if (role === "train") {
      const response = await fetch("/static/dataset-loader.py");
      if (!response.ok)
        throw Error("Training loader unavailable; no archive exported.");
      entries.push(
        {
          name: "project.json",
          text: JSON.stringify(ReviewCore.projectBackup(project)),
        },
        { name: "dataset.py", text: await response.text() },
      );
      const format = await fetch("/api/dataset-format");
      if (!format.ok)
        throw Error("Dataset format loader unavailable; export stopped.");
      entries.push({ name: "dataset_format.py", text: await format.text() });
      if (task === "ccrt") {
        const plan = DatasetPlans.build(manifest, options.comparisons || {});
        plan.dataset_sha256 = Array.from(
          new Uint8Array(
            await crypto.subtle.digest("SHA-256", encoder.encode(manifestText)),
          ),
          (b) => b.toString(16).padStart(2, "0"),
        ).join("");
        entries.push({
          name: "comparison-plan.json",
          text: JSON.stringify(plan),
        });
      }
    }
    entries.push({
      name: "README.txt",
      text:
        role === "test"
          ? "FlyScope evaluation-only dataset. Dataset images and manual annotations/geometry only; no duplicate crops, project history or training extras. Use tools/model_experiments.py evaluate or compare. Score only eligible annotated wells; unannotated wells are never negatives. This is not an editable project backup. Keep this grid separate from training.\n"
          : "FlyScope training dataset: dataset images, shared well boundaries, manual targets, geometry, provenance and project backup. Partial annotations are supported; unannotated wells are omitted. Empty is a manual occupancy target; coma/awake requires one individual. Keep comparison-plan.json identical across models; each run selects entire wells, only their eligible labels, and fixed seeds. Poor/rich well IDs are experiment selections, not special annotation notes. Unzip then use tools/model_experiments.py; see repository README. Crops are derived lazily by dataset.py and every model trainer; no duplicate per-well PNGs are required. Extract ALL ZIP parts into the SAME empty directory before training. See WORKFLOW.md.\n",
    });
    entries.push({
      name: "dataset-complete.json",
      text: JSON.stringify({
        schema_version: 1,
        image_count: images.length,
        annotated_sample_count: annotations.length,
      }),
    });
    if (cancelled()) throw Error("Dataset export cancelled.");
    if (options.directory) {
      for (const [i, entry] of entries.entries()) {
        await writeEntry(options.directory, entry, cancelled);
        onProgress(i + 1, entries.length, "metadata");
      }
      return { manifest, directory: options.directory.name };
    }
    const archives = await zipParts(
      entries,
      (done, total) => onProgress(done, total, "archive"),
      cancelled,
      options.partBytes,
    );
    return {
      blob: archives.length === 1 ? archives[0] : null,
      archives,
      manifest,
    };
  }

  async function readArchiveJSON(file) {
    const tail = new Uint8Array(
        await file.slice(Math.max(0, file.size - 65557)).arrayBuffer(),
      ),
      view = new DataView(tail.buffer);
    let end = -1;
    for (let i = tail.length - 22; i >= 0; i--)
      if (
        view.getUint32(i, true) === 0x06054b50 &&
        i + 22 + view.getUint16(i + 20, true) === tail.length
      ) {
        end = i;
        break;
      }
    if (end < 0) throw Error("Not a supported ZIP archive.");
    const count = view.getUint16(end + 10, true),
      size = view.getUint32(end + 12, true),
      offset = view.getUint32(end + 16, true);
    if (
      view.getUint16(end + 4, true) ||
      view.getUint16(end + 6, true) ||
      count === 65535 ||
      size > 16 * 1024 * 1024 ||
      offset + size > file.size
    )
      throw Error("Unsupported ZIP archive directory.");
    const dir = new Uint8Array(
        await file.slice(offset, offset + size).arrayBuffer(),
      ),
      dv = new DataView(dir.buffer),
      decoder = new TextDecoder(),
      entries = [];
    let pos = 0;
    for (let i = 0; i < count; i++) {
      if (pos + 46 > dir.length || dv.getUint32(pos, true) !== 0x02014b50)
        throw Error("Invalid ZIP directory.");
      const nl = dv.getUint16(pos + 28, true),
        extra = dv.getUint16(pos + 30, true),
        comment = dv.getUint16(pos + 32, true);
      if (pos + 46 + nl + extra + comment > dir.length)
        throw Error("Invalid ZIP entry.");
      const name = decoder.decode(dir.slice(pos + 46, pos + 46 + nl));
      if (["project.json", "report.json"].includes(name))
        entries.push({
          name,
          flags: dv.getUint16(pos + 8, true),
          method: dv.getUint16(pos + 10, true),
          crc: dv.getUint32(pos + 16, true),
          compressed: dv.getUint32(pos + 20, true),
          size: dv.getUint32(pos + 24, true),
          offset: dv.getUint32(pos + 42, true),
        });
      pos += 46 + nl + extra + comment;
    }
    const entry =
      entries.find((e) => e.name === "project.json") ||
      entries.find((e) => e.name === "report.json");
    if (!entry)
      throw Error("Archive contains neither project.json nor report.json.");
    if (
      entries.filter((e) => e.name === entry.name).length !== 1 ||
      entry.flags & 1 ||
      entry.size > 512 * 1024 * 1024 ||
      entry.compressed > 512 * 1024 * 1024
    )
      throw Error("Unsupported or oversized project metadata.");
    const header = new DataView(
      await file.slice(entry.offset, entry.offset + 30).arrayBuffer(),
    );
    if (header.byteLength < 30 || header.getUint32(0, true) !== 0x04034b50)
      throw Error("Invalid ZIP local header.");
    const start =
      entry.offset +
      30 +
      header.getUint16(26, true) +
      header.getUint16(28, true);
    if (start + entry.compressed > offset)
      throw Error("Invalid ZIP data range.");
    let bytes = new Uint8Array(
      await file.slice(start, start + entry.compressed).arrayBuffer(),
    );
    if (entry.method === 8) {
      const reader = new Blob([bytes])
          .stream()
          .pipeThrough(new DecompressionStream("deflate-raw"))
          .getReader(),
        chunks = [];
      let total = 0;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        total += value.length;
        if (total > entry.size) {
          await reader.cancel();
          throw Error("Invalid expanded ZIP size.");
        }
        chunks.push(value);
      }
      bytes = new Uint8Array(total);
      let at = 0;
      for (const chunk of chunks) {
        bytes.set(chunk, at);
        at += chunk.length;
      }
    } else if (entry.method !== 0) throw Error("Unsupported ZIP compression.");
    if (bytes.length !== entry.size || crc32(bytes) !== entry.crc)
      throw Error("ZIP metadata checksum failed.");
    return JSON.parse(decoder.decode(bytes));
  }

  const api = {
    readArchiveJSON,
    zipParts,
    writeEntry,
    zip,
    crc32,
    png,
    locationMap,
    training,
  };
  if (typeof module !== "undefined") module.exports = api;
  else root.TrainingExport = api;
})(globalThis);
