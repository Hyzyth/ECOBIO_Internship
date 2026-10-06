const test = require("node:test"),
  assert = require("node:assert/strict"),
  C = require("../frontend/static/review-core.js");
const points = [
  [0, 0],
  [1, 0],
  [1, 1],
];
function project() {
  return {
    schema_version: 1,
    mode: "sequence",
    interval: 2,
    frames: [0, 1, 2, 3].map((i) => ({
      id: String(i),
      name: `frame${i}.png`,
      timestamp: null,
    })),
    wells: [
      {
        id: "A3",
        uid: "a",
        points,
        overrides: {},
        detection: { source: "manual" },
      },
    ],
    records: {
      "0:a": {
        prediction: "coma",
        confidence: 0.94,
        model: { id: "m", version: "1" },
      },
      "1:a": {
        prediction: "coma",
        confidence: 0.8,
        correction: { state: "awake", egg_count: 3 },
      },
      "2:a": { prediction: "unknown" },
      "3:a": { prediction: "awake" },
    },
  };
}
test("natural order and filename timestamps", () => {
  assert.deepEqual(
    C.orderFiles([
      { name: "frame10.png" },
      { name: "frame2.png" },
      { name: "frame1.png" },
    ]).map((f) => f.name),
    ["frame1.png", "frame2.png", "frame10.png"],
  );
  assert.equal(C.timestamp("2026-10-06_09-12-03.jpg"), "2026-10-06T09:12:03");
});
test("corrections stay separate; unknown breaks transitions", () => {
  const p = project(),
    s = C.summary(p)[0];
  assert.equal(s.transitions, 1);
  assert.equal(s.awake_seconds, 4);
  assert.equal(s.coma_seconds, 2);
  assert.equal(s.unknown_frames, 1);
  assert.equal(p.records["1:a"].prediction, "coma");
  assert.equal(s.average_model_confidence, 0.87);
});
test("nonsequential reports no duration or transition", () => {
  const p = project();
  p.mode = "single";
  assert.equal(C.summary(p)[0].transitions, null);
  assert.equal(C.summary(p)[0].awake_seconds, null);
});
test("export records model provenance and frame-specific boundaries", () => {
  const p = project();
  p.wells[0].overrides[1] = [
    [0.1, 0.1],
    [0.5, 0.1],
    [0.5, 0.5],
  ];
  const row = C.exportProject(p).results[1];
  assert.equal(row.prediction, "coma");
  assert.equal(row.effective_state, "awake");
  assert.equal(row.user_correction.egg_count, 3);
  assert.equal(row.geometry_override, true);
  assert.deepEqual(row.well_boundary, p.wells[0].overrides[1]);
});
test("CSV escapes quoted filenames and spreadsheet formulas", () => {
  const p = project();
  p.frames[0].name = '=evil,"quoted"';
  const csv = C.csv(p);
  assert.ok(csv.includes('"\'=evil,""quoted"""'));
});
test("restore validates file identity, bounds, and annotations", () => {
  const p = project();
  assert.equal(C.validateProject(p, p.frames), p);
  assert.throws(() => C.validateProject(p, [{ id: "different" }]));
  p.wells[0].points = [
    [2, 0],
    [1, 0],
    [1, 1],
  ];
  assert.throws(() => C.validateProject(p, p.frames));
});

test("timestamp ordering is independent of camera filename prefix", () => {
  const files = [
    { name: "camA_2026-10-06_10-00-00.png" },
    { name: "camB_2026-10-06_09-00-00.png" },
  ];
  assert.equal(C.orderFiles(files)[0].name, files[1].name);
});

test("simple polygon validation rejects crossings, retracing, duplicate vertices and zero area", () => {
  for (const points of [
    [
      [0, 0],
      [1, 1],
      [0, 1],
      [1, 0],
    ],
    [
      [0, 0],
      [1, 0],
      [0.5, 0],
      [1, 1],
      [0, 1],
    ],
    [
      [0, 0],
      [1, 0],
      [1, 0],
      [0, 1],
    ],
    [
      [0, 0],
      [0.5, 0.5],
      [1, 1],
    ],
  ])
    assert.equal(C.isSimplePolygon(points), false);
  assert.equal(
    C.isSimplePolygon([
      [0, 0],
      [1, 0],
      [0.5, 0.5],
      [1, 1],
      [0, 1],
    ]),
    true,
  );
  const p = project();
  p.wells[0].overrides[1] = [
    [0, 0],
    [1, 1],
    [0, 1],
    [1, 0],
  ];
  assert.throws(() => C.validateProject(p, p.frames));
});

test("project restore preserves valid regions and rejects invalid registration metadata", () => {
  const p = project();
  p.detection_region = [0.1, 0.1, 0.9, 0.9];
  p.alignments = {
    1: { method: "perspective", inlier_ratio: 0.95, median_error_px: 0.5 },
  };
  assert.equal(C.validateProject(p, p.frames), p);
  p.detection_region = [0.9, 0.1, 0.1, 0.9];
  assert.throws(() => C.validateProject(p, p.frames));
  p.detection_region = null;
  p.alignments[1].inlier_ratio = 2;
  assert.throws(() => C.validateProject(p, p.frames));
});
test("partial boundary status survives export/restore and appears in CSV", () => {
  const p = project();
  p.wells[0].detection.partial = true;
  p.wells[0].partial_frames = { 1: false, 2: true };
  const saved = C.exportProject(p);
  assert.deepEqual(
    saved.results.map((r) => r.boundary_partial),
    [true, false, true, true],
  );
  assert.match(C.csv(p), /boundary_partial/);
  C.validateProject(saved, p.frames);
  saved.wells[0].partial_frames[2] = "false";
  assert.throws(() => C.validateProject(saved, p.frames), /partial boundary/);
});
