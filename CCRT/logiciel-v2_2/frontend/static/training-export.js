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
      const bytes = new Uint8Array(await blob.arrayBuffer()),
        crc = crc32(bytes),
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
  ) {
    const annotations = ReviewCore.trainingAnnotations(project);
    if (!annotations.length)
      throw Error(
        "Add manual state labels or egg counts before exporting training data.",
      );
    const entries = [],
      images = [],
      task = ReviewCore.taskOf(project),
      frames = [...new Set(annotations.map((a) => a.frame_index))];
    for (const [n, i] of frames.entries()) {
      if (cancelled())
        throw Error(
          "Training export cancelled. No incomplete archive was downloaded.",
        );
      const file = files[i],
        ext =
          file.name.match(/\.(png|jpe?g|bmp|webp)$/i)?.[1].toLowerCase() ||
          "bin",
        imagePath = `images/frame-${String(i + 1).padStart(6, "0")}.${ext}`;
      const { image, release } = await decode(file);
      try {
        const sha = Array.from(
          new Uint8Array(
            await crypto.subtle.digest("SHA-256", await file.arrayBuffer()),
          ),
          (b) => b.toString(16).padStart(2, "0"),
        ).join("");
        images.push({
          frame_number: i + 1,
          original_filename: project.frames[i].name,
          path: imagePath,
          width: image.width,
          height: image.height,
          sha256: sha,
        });
        entries.push({ name: imagePath, blob: file });
        for (const [j, a] of annotations
          .filter((a) => a.frame_index === i)
          .entries()) {
          if (cancelled()) throw Error("Training export cancelled.");
          const points = a.boundary_normalized.map((p) => [
              p[0] * image.width,
              p[1] * image.height,
            ]),
            xs = points.map((p) => p[0]),
            ys = points.map((p) => p[1]);
          const x = Math.max(0, Math.floor(Math.min(...xs))),
            y = Math.max(0, Math.floor(Math.min(...ys))),
            right = Math.min(image.width, Math.ceil(Math.max(...xs))),
            bottom = Math.min(image.height, Math.ceil(Math.max(...ys)));
          const crop = document.createElement("canvas");
          crop.width = right - x;
          crop.height = bottom - y;
          const ctx = crop.getContext("2d");
          ctx.beginPath();
          points.forEach((p, k) =>
            k ? ctx.lineTo(p[0] - x, p[1] - y) : ctx.moveTo(p[0] - x, p[1] - y),
          );
          ctx.closePath();
          ctx.clip();
          ctx.drawImage(image, -x, -y);
          const cropPath = `crops/frame-${String(i + 1).padStart(6, "0")}-region-${String(j + 1).padStart(4, "0")}.png`;
          Object.assign(a, {
            image_path: imagePath,
            crop_path: cropPath,
            image_width: image.width,
            image_height: image.height,
            boundary_pixels: points,
            bbox_pixels: [x, y, right - x, bottom - y],
            mask_encoding: "PNG alpha channel: outside polygon transparent",
          });
          entries.push({ name: cropPath, blob: await png(crop) });
          crop.width = crop.height = 0;
        }
      } finally {
        release();
      }
      onProgress(n + 1, frames.length);
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
    const manifest = {
      schema_version: 1,
      kind: `${task}-training`,
      task,
      experiment_id: project.experiment_id,
      split_group: project.experiment_id,
      created_at: new Date().toISOString(),
      classes: task === "ccrt" ? { coma: 0, awake: 1 } : null,
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
      images,
      annotations,
      all_frame_count: project.frames.length,
      annotated_sample_count: annotations.length,
      eligible_sample_count: annotations.filter((a) => a.eligible_for_training)
        .length,
      notes: [
        "Targets come exclusively from manual annotations. Unannotated frame/wells are omitted.",
        "Uncertain, partial, unreviewed, and CCRT empty/multiple/obscured/unconfirmed occupancy regions are retained with eligibility=false.",
        "Split related experiments together. Do not randomly split adjacent frames or wells from the same experiment.",
        "Egg targets are provisional independent-image or optional region counts; no well layout is assumed.",
      ],
    };
    const response = await fetch("/static/dataset-loader.py");
    if (!response.ok)
      throw Error("Training loader unavailable; no archive exported.");
    const loader = await response.text();
    entries.push(
      { name: "annotations.json", text: JSON.stringify(manifest, null, 2) },
      {
        name: "project.json",
        text: JSON.stringify(ReviewCore.exportProject(project), null, 2),
      },
      { name: "annotations.csv", text: ReviewCore.tableCSV(annotations) },
      { name: "dataset.py", text: loader },
      {
        name: "README.txt",
        text: `FlyScope ${task} training dataset\n\nOriginal frames: images/\nMasked crops at original pixel resolution: crops/ (PNG alpha is the well mask).\nannotations.json contains frame/well IDs, SHA-256 image hashes, normalized/pixel polygons, crop boxes, manual label provenance, review status and eligibility. project.json retains the complete review and history.\n\nInstall Pillow, then: python dataset.py .\nImport samples(root) in your training code; it yields RGB PIL images, targets and annotation metadata. Only eligible manual targets are selected by default. Unknown is uncertainty, never a coma/awake class. Coma=0; Awake=1. Egg targets are integer counts, including zero. No model is supplied.\n\nKeep the experiment split_group together when creating train/validation/test splits. Pixel data are original image pixels, never UI screenshots. Cropped reference boundaries cannot recover unseen rims; review all frames before labeling training targets.\n`,
      },
    );
    if (cancelled()) throw Error("Training export cancelled.");
    return {
      blob: await zip(
        entries,
        (done, total) => onProgress(done, total, "archive"),
        cancelled,
      ),
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
      entry.size > 32 * 1024 * 1024 ||
      entry.compressed > 32 * 1024 * 1024
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
    zip,
    crc32,
    png,
    locationMap,
    training,
  };
  if (typeof module !== "undefined") module.exports = api;
  else root.TrainingExport = api;
})(globalThis);
