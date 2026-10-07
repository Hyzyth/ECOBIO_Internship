/* One immutable well/sample selection plan is shared by every model. */
(function (root) {
  function random(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let v = state;
      v = Math.imul(v ^ (v >>> 15), v | 1);
      v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
      return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
    };
  }
  function build(manifest, options = {}) {
    if (manifest.role === "test")
      throw Error("Test datasets cannot create training selections.");
    const seed = Number(options.seed ?? 2026);
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295)
      throw Error("Use a seed between 0 and 4294967295.");
    const labels = manifest.annotations.filter((a) => a.eligible_for_training),
      pool = [...new Set(labels.map((a) => a.well_uid))].sort();
    if (!pool.length)
      throw Error(
        "No eligible manual annotations. Confirm occupancy and boundaries first.",
      );
    const runs = [],
      skipped = [],
      make = (id, wells, runSeed) => ({
        id,
        seed: runSeed >>> 0,
        well_uids: [...wells].sort(),
        annotation_ids: labels
          .filter((a) => wells.includes(a.well_uid))
          .map((a) => a.id)
          .sort(),
      });
    runs.push(make("all-annotated-wells", pool, seed));
    const budgets = options.budgets || [
      { wells: 50, repeats: 2 },
      { wells: 25, repeats: 5 },
      { wells: 10, repeats: 10 },
      { wells: 5, repeats: 20 },
      { wells: 2, repeats: 30 },
    ];
    for (const budget of budgets) {
      if (
        !Number.isInteger(budget.wells) ||
        budget.wells < 1 ||
        !Number.isInteger(budget.repeats) ||
        budget.repeats < 1 ||
        budget.repeats > 1000
      )
        throw Error("Invalid well budget/repeat count.");
      if (budget.wells > pool.length) {
        skipped.push({
          ...budget,
          reason: `Only ${pool.length} wells have eligible labels`,
        });
        continue;
      }
      for (let n = 0; n < budget.repeats; n++) {
        const runSeed = (seed + Math.imul(budget.wells, 100003) + n) >>> 0,
          rand = random(runSeed),
          wells = [...pool];
        for (let i = wells.length - 1; i > 0; i--) {
          const j = Math.floor(rand() * (i + 1));
          [wells[i], wells[j]] = [wells[j], wells[i]];
        }
        runs.push(
          make(
            `random-${budget.wells}-${n + 1}`,
            wells.slice(0, budget.wells),
            runSeed,
          ),
        );
      }
    }
    for (const kind of ["poor", "rich"]) {
      const name = options[kind + "_well_id"]?.trim();
      if (!name) continue;
      const matches = (manifest.wells || []).filter(
        (w) =>
          w.well_id === name &&
          pool.includes(w.well_uid) &&
          w.occupancy === "single",
      );
      if (matches.length !== 1)
        throw Error(
          `${kind} pose well ID must identify one annotated, single-individual well.`,
        );
      runs.push(make(kind + "-poses", [matches[0].well_uid], seed));
    }
    return {
      schema_version: 1,
      kind: "flyscope-comparison-plan",
      seed,
      experiment_id: manifest.experiment_id,
      selection_unit: "well",
      eligible_well_count: pool.length,
      detected_well_count: manifest.coverage?.detected_wells ?? pool.length,
      partial_annotations_supported: true,
      all_well_frames_annotated:
        Number.isInteger(manifest.coverage?.expected_samples) &&
        manifest.coverage.expected_samples > 0 &&
        manifest.coverage.annotated_samples ===
          manifest.coverage.expected_samples,
      all_well_frames_eligible:
        Number.isInteger(manifest.coverage?.expected_samples) &&
        manifest.coverage.expected_samples > 0 &&
        manifest.coverage.eligible_samples ===
          manifest.coverage.expected_samples,
      runs,
      skipped,
      notes: [
        "All models use the exact same annotation IDs and seeds. Each repetition restarts from its original model. Repetitions may select overlapping or identical subsets; overlap is explicit in the saved IDs. Test grids must be separate exports and are never selected for training.",
      ],
    };
  }
  const api = { build };
  if (typeof module !== "undefined") module.exports = api;
  root.DatasetPlans = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
