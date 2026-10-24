const { test } = require("node:test"),
  assert = require("node:assert/strict");
const C = require("../frontend/static/review-core.js");

test("only confirmed wells stop; skipped frames have no invented raw confidence", () => {
  const p = {
    study: "ccrt",
    mode: "sequence",
    interval: 1,
    frames: Array.from({ length: 8 }, (_, i) => ({ name: String(i) })),
    wells: ["a", "b"].map((uid) => ({
      uid,
      occupancy: "single",
      detection: {},
    })),
    records: {},
    recovery_policy: {
      confidence_threshold: 0.8,
      awake_consecutive_frames: 2,
      require_coma_before_awake: true,
    },
  };
  const calls = { a: 0, b: 0 };
  p.records["4:a"] = { correction: { state: "coma" } };
  for (let i = 0; i < p.frames.length; i++) {
    for (const w of C.prepareInference(p, i, "session")) {
      calls[w.uid]++;
      p.records[C.key(i, w.uid)] = {
        model: { version: "v" },
        analysis_session: "session",
        prediction: w.uid === "a" && i >= 1 ? "awake" : "coma",
        confidence: 0.99,
      };
    }
    C.applyRecoveryPolicy(p);
  }
  assert.deepEqual(calls, { a: 3, b: 8 });
  const skipped = p.records["4:a"];
  assert.equal(skipped.prediction, "awake");
  assert.equal(skipped.raw_prediction, "unknown");
  assert.equal(skipped.raw_confidence, null);
  assert.equal(skipped.inference_skipped, "confirmed_awake");
  assert.equal(skipped.correction.state, "coma");
  assert.equal(C.prepareInference(p, 5, "new-session").length, 2);
  assert.equal(C.prepareInference(p, 5, "session", false).length, 2);
  p.recovery_policy.automatic_confirmation = false;
  C.applyRecoveryPolicy(p);
  assert.equal(p.records["5:a"].prediction, "unsure");
});
