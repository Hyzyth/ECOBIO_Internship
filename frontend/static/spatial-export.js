/* A constant normalized region preserves well identity across camera shifts. */
(function (root) {
  "use strict";
  function bounds(project, width, height, padding = 32) {
    if (!Number.isInteger(padding) || padding < 0 || padding > 512)
      throw Error("Crop padding must be an integer from 0 to 512 pixels.");
    let xmin = 1,
      xmax = 0,
      ymin = 1,
      ymax = 0,
      count = 0;
    for (const w of project.wells)
      for (const polygon of [w.points, ...Object.values(w.overrides || {})])
        for (const p of polygon) {
          if (
            !Array.isArray(p) ||
            p.length !== 2 ||
            p.some((v) => !Number.isFinite(v) || v < 0 || v > 1)
          )
            throw Error("Cropping needs valid normalized well boundaries.");
          xmin = Math.min(xmin, p[0]);
          xmax = Math.max(xmax, p[0]);
          ymin = Math.min(ymin, p[1]);
          ymax = Math.max(ymax, p[1]);
          count++;
        }
    if (!count) throw Error("Cropping needs well boundaries.");
    const left = Math.max(0, Math.floor(xmin * width) - padding),
      top = Math.max(0, Math.floor(ymin * height) - padding),
      right = Math.min(width, Math.ceil(xmax * width) + padding),
      bottom = Math.min(height, Math.ceil(ymax * height) + padding);
    if (right <= left || bottom <= top) throw Error("Empty well crop.");
    return [
      left / width,
      top / height,
      (right - left) / width,
      (bottom - top) / height,
    ];
  }
  function transform(points, roi) {
    return points.map(([x, y]) => [
      Math.max(0, Math.min(1, (x - roi[0]) / roi[2])),
      Math.max(0, Math.min(1, (y - roi[1]) / roi[3])),
    ]);
  }
  const api = { bounds, transform };
  if (typeof module !== "undefined") module.exports = api;
  root.SpatialExport = api;
})(typeof window !== "undefined" ? window : globalThis);
