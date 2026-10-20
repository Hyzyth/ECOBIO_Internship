const test = require("node:test"),
  assert = require("node:assert/strict"),
  C = require("../frontend/static/review-core.js"),
  P = require("../frontend/static/dataset-plans.js");
function project() {
  return {
    schema_version: 1,
    task: "ccrt",
    mode: "sequence",
    interval: 2,
    analysis_start_index: 2,
    time_origin: { confirmed: true, offset_seconds: 5 },
    frames: Array.from({ length: 7 }, (_, i) => ({
      id: `original-${i}`,
      name: `image-${i}.png`,
    })),
    wells: [
      {
        uid: "w",
        id: "W1",
        occupancy: "single",
        species: "D. simulans",
        points: [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
        ],
        overrides: {},
        detection: { source: "manual", review_required: false },
      },
    ],
    records: {},
    history: [],
  };
}
test("raw setup window retains source indices and timer; trimmed offset unknown still reports recovery frame identity", () => {
  const p = project();
  p.records = {
    "2:w": { correction: { state: "coma" } },
    "3:w": { correction: { state: "awake" } },
  };
  let m = C.recoveryMetrics(p, p.wells[0]);
  assert.equal(m.first_awake_frame, 4);
  assert.equal(m.first_awake_frame_id, "original-3");
  assert.equal(m.ccrt_seconds, 11);
  assert.equal(C.trainingAnnotations(p).length, 2);
  assert.equal(C.trainingAnnotations(p)[0].frame_index, 2);
  assert.equal(C.frameTiming(p, 0).excluded_setup_frame, true);
  assert.equal(C.frameTiming(p, 3).timer_seconds, 11);
  p.time_origin.confirmed = false;
  m = C.recoveryMetrics(p, p.wells[0]);
  assert.equal(m.ccrt_seconds, null);
  assert.equal(m.recovery_upper_seconds, null);
  assert.equal(m.first_awake_frame, 4);
  const report = C.humanReport(p);
  assert.equal(report.results[0]["First awake image"], "image-3.png");
  assert.equal(report.results[0].Species, "D. simulans");
  assert.ok(!Object.hasOwn(report.results[0], "CCRT (s)"));
  assert.ok(!Object.hasOwn(report.results[0], "Coma (s)"));
});
test("Unsure abstains until a sustained Awake run; recovery is backdated to its onset and retained through later adverse poses", () => {
  const p = project();
  p.analysis_start_index = 0;
  const raw = ["coma", "awake", "coma", "awake", "awake", "awake", "coma"];
  raw.forEach(
    (prediction, i) =>
      (p.records[C.key(i, "w")] = {
        prediction,
        confidence: 0.95,
        model: { version: "fixed" },
        correction: i === 6 ? { state: "unknown" } : undefined,
      }),
  );
  C.applyRecoveryPolicy(p, {
    confidence_threshold: 0.8,
    awake_consecutive_frames: 3,
  });
  assert.deepEqual(
    raw.map((_, i) => p.records[C.key(i, "w")].prediction),
    ["coma", "unsure", "coma", "awake", "awake", "awake", "awake"],
  );
  assert.equal(p.records["3:w"].recovery_decision.confirmed_frame_index, 5);
  assert.equal(p.records["3:w"].recovery_decision.onset_frame_index, 3);
  assert.equal(p.records["6:w"].raw_prediction, "coma");
  assert.equal(p.records["6:w"].confidence, null);
  assert.equal(C.effective(p.records["6:w"], p.wells[0]), "unknown");
  C.applyRecoveryPolicy(p, {
    confidence_threshold: 0.99,
    awake_consecutive_frames: 3,
  });
  assert.ok(
    raw.every((_, i) => p.records[C.key(i, "w")].prediction === "unsure"),
  );
});
test("random individual budgets exclude Empty but full grid includes it", () => {
  const wells = Array.from({ length: 26 }, (_, i) => ({
    well_uid: `w${i}`,
    well_id: `W${i}`,
    occupancy: i === 25 ? "empty" : "single",
  }));
  const m = {
    role: "train",
    wells,
    annotations: wells.map((w) => ({
      id: w.well_uid,
      well_uid: w.well_uid,
      eligible_for_training: true,
    })),
  };
  const plan = P.build(m, { poor_well_id: "W0", rich_well_id: "W1" });
  assert.equal(plan.runs.length, 10);
  assert.equal(plan.runs[0].annotation_ids.length, 26);
  assert.ok(plan.runs.slice(1).every((r) => !r.well_uids.includes("w25")));
  assert.equal(plan.eligible_individual_count, 25);
});
test("different model analysis sessions cannot supply each other's recovery evidence", () => {
  const p = project();
  p.analysis_start_index = 0;
  for (let i = 0; i < 7; i++)
    p.records[C.key(i, "w")] = {
      prediction: "awake",
      confidence: 0.95,
      model: { version: "same-weights" },
      analysis_session: i < 2 ? "new-run" : "old-run",
    };
  C.applyRecoveryPolicy(p);
  assert.equal(p.records["0:w"].prediction, "unsure");
  assert.equal(p.records["1:w"].prediction, "unsure");
  assert.equal(p.records["2:w"].prediction, "awake");
});
