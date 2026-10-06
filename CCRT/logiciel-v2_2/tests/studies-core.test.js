const test = require("node:test"),
  assert = require("node:assert/strict"),
  C = require("../frontend/static/review-core.js"),
  Z = require("../frontend/static/training-export.js");
function project(task = "ccrt") {
  return {
    schema_version: 1,
    task,
    experiment_id: "experiment-1",
    mode: task === "eggs" ? "batch" : "sequence",
    interval: 2,
    frames: [0, 1, 2].map((i) => ({
      id: String(i),
      name: `frame${i}.png`,
      timestamp: null,
    })),
    wells: [
      {
        id: "W1",
        uid: "well1",
        points: [
          [0.1, 0.1],
          [0.8, 0.1],
          [0.8, 0.8],
          [0.1, 0.8],
        ],
        overrides: {},
        detection: { source: "manual", review_required: false },
        review_frames: [],
      },
    ],
    records: {},
    history: [],
  };
}
test("inclusive range labels preserve models, uncertain interruptions and undo prior labels", () => {
  const p = project();
  p.records["0:well1"] = {
    prediction: "awake",
    confidence: 0.8,
    model: { id: "m" },
  };
  C.applyStateRange(p, "well1", 0, 1, "coma");
  C.applyStateRange(p, "well1", 2, 2, "awake");
  assert.deepEqual(C.summary(p)[0].states, ["coma", "coma", "awake"]);
  assert.equal(p.records["0:well1"].prediction, "awake");
  const op = C.applyStateRange(p, "well1", 1, 1, "unknown");
  assert.equal(C.summary(p)[0].transitions, 0);
  assert.equal(C.undoStateRange(p, op), 1);
  assert.equal(C.summary(p)[0].transitions, 1);
  const overwritten = C.applyStateRange(p, "well1", 0, 2, "coma");
  C.applyStateRange(p, "well1", 2, 2, "awake");
  assert.equal(C.undoStateRange(p, overwritten), 2);
  assert.equal(C.effective(p.records["2:well1"]), "awake");
  assert.throws(() => C.applyStateRange(p, "well1", 2, 1, "coma"), /range/);
});
test("training targets are manual only, frame geometry and eligibility remain explicit", () => {
  const p = project();
  p.records["0:well1"] = { prediction: "awake", confidence: 0.99 };
  assert.deepEqual(C.trainingAnnotations(p), []);
  C.applyStateRange(p, "well1", 0, 2, "coma");
  p.wells[0].review_frames = [1];
  p.wells[0].overrides[1] = [
    [0.2, 0.2],
    [0.8, 0.2],
    [0.8, 0.8],
    [0.2, 0.8],
  ];
  p.wells[0].partial_frames = { 2: true };
  const rows = C.trainingAnnotations(p);
  assert.equal(rows.length, 3);
  assert.deepEqual(
    rows.map((r) => r.eligible_for_training),
    [true, false, false],
  );
  assert.equal(rows[1].boundary_normalized[0][0], 0.2);
  assert.equal(rows[0].split_group, "experiment-1");
  assert.equal(rows[0].label_source, "manual");
  C.applyStateRange(p, "well1", 0, 0, "unknown");
  assert.equal(C.trainingAnnotations(p)[0].eligible_for_training, false);
});
test("egg regions belong to independent images; zero is counted and missing is not zero", () => {
  const p = project("eggs");
  p.wells[0].active_frames = [0];
  p.wells.push({
    ...structuredClone(p.wells[0]),
    id: "R2",
    uid: "well2",
    active_frames: [1],
  });
  p.records["0:well1"] = { correction: { state: null, egg_count: 0 } };
  const report = C.resultsReport(p);
  assert.equal(report.results.length, 2);
  assert.equal(report.results[0].egg_count, 0);
  assert.equal(report.summary[0].total_counted_eggs, 0);
  assert.equal(report.summary[1].total_counted_eggs, null);
  assert.equal(report.summary[1].uncounted_regions, 1);
  assert.equal(report.frame_interval_seconds, null);
  assert.ok(!("state" in report.results[0]));
  assert.equal(C.trainingAnnotations(p).length, 1);
  assert.equal(C.trainingAnnotations(p)[0].label, 0);
  C.validateProject(C.exportProject(p), p.frames);
  p.wells[0].active_frames = [3];
  assert.throws(
    () => C.validateProject(C.exportProject(p), p.frames),
    /membership/,
  );
});
test("human CCRT reports omit polygon arrays and egg fields but retain well locations and uncertainty", () => {
  const p = project();
  C.applyStateRange(p, "well1", 0, 1, "coma");
  const r = C.resultsReport(p);
  assert.equal(r.kind, "ccrt-results");
  assert.equal(r.well_map[0].x_percent, 45);
  assert.ok(!("well_boundary" in r.results[0]));
  assert.ok(!("egg_count" in r.results[0]));
  assert.equal(r.results[0].source, "manual");
  assert.equal(r.summary[0].coma_seconds, 4);
});
test("ZIP CRC and binary entries follow standard ZIP structure and reject unsafe paths", async () => {
  assert.equal(Z.crc32(new TextEncoder().encode("123456789")), 0xcbf43926);
  const blob = await Z.zip([
    { name: "annotations.json", text: '{"label":"coma"}' },
    { name: "images/a.bin", blob: new Blob([new Uint8Array([0, 255, 128])]) },
  ]);
  const bytes = new Uint8Array(await blob.arrayBuffer()),
    v = new DataView(bytes.buffer);
  assert.equal(v.getUint32(0, true), 0x04034b50);
  assert.equal(v.getUint32(bytes.length - 22, true), 0x06054b50);
  assert.equal(v.getUint16(bytes.length - 14, true), 2);
  await assert.rejects(
    () => Z.zip([{ name: "../image.png", text: "bad" }]),
    /path/,
  );
});
test("undo labels preserves later model changes and prediction invalidation", () => {
  const p = project();
  p.records["0:well1"] = { prediction: "coma", confidence: 0.7 };
  const op = C.applyStateRange(p, "well1", 0, 0, "awake");
  p.records["0:well1"] = {
    ...p.records["0:well1"],
    prediction: "awake",
    confidence: 0.9,
  };
  C.undoStateRange(p, op);
  assert.equal(p.records["0:well1"].prediction, "awake");
  assert.equal(p.records["0:well1"].correction, undefined);
  const next = C.applyStateRange(p, "well1", 0, 0, "coma");
  p.records["0:well1"] = { correction: p.records["0:well1"].correction };
  C.undoStateRange(p, next);
  assert.equal(p.records["0:well1"], undefined);
});
test("egg project backups contain count summaries and ZIP cancellation stops packaging", async () => {
  const p = project("eggs");
  p.records["0:well1"] = { correction: { state: null, egg_count: 4 } };
  const saved = C.exportProject(p);
  assert.equal(saved.task, "eggs");
  assert.equal(saved.summary[0].total_counted_eggs, 4);
  assert.ok(!("coma_frames" in saved.summary[0]));
  await assert.rejects(
    () =>
      Z.zip(
        [{ name: "a.json", text: "{}" }],
        () => {},
        () => true,
      ),
    /cancelled/,
  );
});
test("whole-sequence approval keeps failed frames pending and preserves annotations/geometry", () => {
  const p = project();
  p.wells[0].detection.review_required = true;
  p.wells[0].review_frames = [1, 2];
  p.alignment_failures = { 2: { error: "weak match" } };
  C.applyStateRange(p, "well1", 0, 1, "coma");
  const before = structuredClone(p.records),
    points = structuredClone(p.wells[0].points);
  const result = C.approveStableBoundaries(p, ["well1"]);
  assert.deepEqual(result, { approved: 2, skipped: 1 });
  assert.deepEqual(p.wells[0].review_frames, [2]);
  assert.deepEqual(p.records, before);
  assert.deepEqual(p.wells[0].points, points);
});
test("empty occupancy overrides presentation and summary while preserving manual labels", () => {
  const p = project();
  C.applyStateRange(p, "well1", 0, 2, "coma");
  p.wells[0].occupancy = "empty";
  const s = C.summary(p)[0];
  assert.equal(s.empty_frames, 3);
  assert.equal(s.coma_frames, 0);
  assert.equal(s.unknown_frames, 0);
  assert.equal(C.resultsReport(p).results[0].state, "empty");
  assert.equal(C.trainingAnnotations(p)[0].eligible_for_training, false);
  assert.equal(p.records["0:well1"].correction.state, "coma");
  p.wells[0].occupancy = "single";
  assert.equal(C.summary(p)[0].coma_frames, 3);
});
test("legacy readable report restores labels onto matched geometry atomically", () => {
  const p = project();
  C.applyStateRange(p, "well1", 0, 1, "coma");
  C.applyStateRange(p, "well1", 2, 2, "awake");
  const report = C.resultsReport(p),
    blank = project();
  const restored = C.importReadableReport(blank, report);
  assert.equal(restored.imported, 3);
  assert.deepEqual(C.summary(restored.project)[0].states, [
    "coma",
    "coma",
    "awake",
  ]);
  assert.deepEqual(restored.project.wells[0].points, blank.wells[0].points);
  assert.deepEqual(blank.records, {});
  report.results[2].filename = "wrong.png";
  assert.throws(() => C.importReadableReport(blank, report), /images/);
  assert.deepEqual(blank.records, {});
  report.results[2].filename = "frame2.png";
  report.well_map[0].x_percent = 99;
  assert.throws(
    () => C.importReadableReport(blank, report),
    /different position/,
  );
});
test("ZIP report import prefers full project and checks data CRC", async () => {
  const blob = await Z.zip([
    { name: "report.json", text: '{"kind":"ccrt-results"}' },
    { name: "project.json", text: '{"wells":[],"history":["kept"]}' },
  ]);
  assert.deepEqual(await Z.readArchiveJSON(blob), {
    wells: [],
    history: ["kept"],
  });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  bytes[30 + "report.json".length + 2] ^= 1; // Non-selected report does not affect project.
  assert.equal((await Z.readArchiveJSON(new Blob([bytes]))).history[0], "kept");
  const view = new DataView(bytes.buffer);
  let pos = 30 + view.getUint16(26, true) + view.getUint32(18, true);
  bytes[pos + 30 + "project.json".length + 2] ^= 1;
  await assert.rejects(() => Z.readArchiveJSON(new Blob([bytes])), /checksum/);
  const legacy = await Z.zip([
    { name: "report.json", text: '{"kind":"ccrt-results"}' },
  ]);
  assert.equal((await Z.readArchiveJSON(legacy)).kind, "ccrt-results");
});
