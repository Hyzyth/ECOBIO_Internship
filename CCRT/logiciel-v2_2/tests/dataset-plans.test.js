const test = require("node:test"),
  assert = require("node:assert/strict"),
  P = require("../frontend/static/dataset-plans.js"),
  C = require("../frontend/static/review-core.js");
function manifest(n = 60) {
  return {
    role: "train",
    experiment_id: "grid1",
    coverage: { detected_wells: 78 },
    wells: Array.from({ length: n }, (_, i) => ({
      well_uid: `uid-${i}`,
      well_id: `W${i + 1}`,
      occupancy: "single",
    })),
    annotations: Array.from({ length: n * 3 }, (_, i) => ({
      id: `annotation-${i}`,
      well_uid: `uid-${Math.floor(i / 3)}`,
      eligible_for_training: true,
    })),
  };
}
test("exact budget/repetition counts, determinism and whole-well membership are shared", () => {
  const m = manifest(),
    p = P.build(m, { seed: 17, poor_well_id: "W1", rich_well_id: "W2" });
  assert.deepEqual(
    p,
    P.build(m, { seed: 17, poor_well_id: "W1", rich_well_id: "W2" }),
  );
  assert.equal(p.runs.length, 70);
  assert.equal(p.runs[0].well_uids.length, 60);
  for (const [count, repeats] of [
    [50, 2],
    [25, 5],
    [10, 10],
    [5, 20],
    [2, 30],
  ]) {
    const runs = p.runs.filter((r) => r.id.startsWith(`random-${count}-`));
    assert.equal(runs.length, repeats);
    for (const r of runs) {
      assert.equal(new Set(r.well_uids).size, count);
      assert.equal(r.annotation_ids.length, count * 3);
      assert.ok(
        r.annotation_ids.every((id) =>
          r.well_uids.includes(m.annotations.find((a) => a.id === id).well_uid),
        ),
      );
    }
  }
  assert.deepEqual(p.runs.at(-2).well_uids, ["uid-0"]);
  assert.deepEqual(p.runs.at(-1).well_uids, ["uid-1"]);
  assert.notDeepEqual(
    p.runs[1].well_uids,
    P.build(m, { seed: 18 }).runs[1].well_uids,
  );
});
test("partial annotations exclude unlabeled/ineligible wells; oversized budgets stay explicit", () => {
  const m = manifest(3);
  m.annotations[0].eligible_for_training = false;
  const p = P.build(m);
  assert.equal(p.skipped.length, 4);
  assert.equal(p.runs[0].annotation_ids.length, 8);
  assert.throws(() => P.build(m, { poor_well_id: "missing" }), /well ID/);
  assert.throws(() => P.build({ ...m, role: "test" }), /Test datasets/);
});
test("manual Empty occupancy is a separate supervised target without inventing coma labels", () => {
  const p = {
    task: "ccrt",
    mode: "sequence",
    interval: 1,
    experiment_id: "x",
    frames: [{ name: "f1" }, { name: "f2" }],
    wells: [
      {
        uid: "w1",
        id: "W1",
        occupancy: "empty",
        points: [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
        ],
        detection: { review_required: false },
        overrides: {},
      },
    ],
    records: {},
    history: [],
  };
  const a = C.trainingAnnotations(p);
  assert.equal(a.length, 2);
  assert.ok(
    a.every(
      (x) =>
        x.label === "empty" &&
        x.label_source === "manual_occupancy" &&
        x.eligible_for_training &&
        !x.state_training_eligible,
    ),
  );
  assert.deepEqual(p.records, {});
  p.wells[0].review_frames = [1];
  assert.equal(C.trainingAnnotations(p)[1].eligible_for_training, false);
});
