"""Recovery success includes missed events; only human truth defines eligibility."""

import unittest
from flyscope.reporting.metrics import first_recovery
from flyscope.reporting.recovery_metrics import (
    recovery_metrics,
    individual_timing_target,
)
from flyscope.data.format import hydrate_manifest


def rows(truth, predictions, uid="well"):
    return [
        dict(
            well_id=uid,
            well_name=uid,
            frame_index=i,
            frame_id=str(i),
            filename=f"{i}.png",
            ground_truth=t,
            scored=t is not None,
            prediction=p,
        )
        for i, (t, p) in enumerate(zip(truth, predictions))
    ]


def score(data):
    return recovery_metrics(
        data,
        {
            "mode": "sequence",
            "interval": 2,
            "time_origin": {"confirmed": True, "offset_seconds": 10},
        },
        first_recovery,
    )


class RecoveryTiming(unittest.TestCase):
    def test_individual_target_is_not_an_average_and_misses_fail(self):
        self.assertTrue(individual_timing_target([5] * 19 + [8])["passed"])
        self.assertFalse(individual_timing_target([0] * 19 + [9])["passed"])
        self.assertFalse(individual_timing_target([0] * 19 + [None])["passed"])
        self.assertFalse(individual_timing_target([5] * 18 + [6, 8])["passed"])
        self.assertTrue(individual_timing_target([-5] * 19 + [-8])["passed"])
        target = individual_timing_target([0] * 60)
        self.assertEqual(target["required_within_5"], 57)
        self.assertIsNone(individual_timing_target([])["passed"])

    def test_model_uncertainty_never_filters_bad_onset_and_misses_count(self):
        data = rows(
            ["coma"] * 2 + ["awake"] * 3, ["coma", "unsure", "unsure", "awake", "awake"]
        )
        data += rows(["coma"] * 2 + ["awake"] * 3, ["unsure"] * 5, "missed")
        m = score(data)
        self.assertEqual(m["paired_observed_wells"], 1)
        self.assertEqual(m["mae_frames"], 1)
        self.assertEqual(m["max_absolute_error_frames"], 1)
        self.assertFalse(m["all_exact_events_detected"])
        self.assertEqual(m["within_tolerance"]["8"]["fraction"], 0.5)
        self.assertEqual(
            m["within_tolerance"]["1"], dict(hits=1, total=2, fraction=0.5)
        )
        self.assertEqual(m["missed_recoveries"], 1)
        self.assertEqual(m["event_coverage"], 0.5)
        w = next(w for w in m["wells"] if w["well_uid"] == "well")
        self.assertEqual(w["ground_truth_ccrt_seconds"], 14)
        self.assertEqual(w["predicted_ccrt_seconds"], 16)

    def test_initial_awake_is_scored_as_early_not_discarded(self):
        m = score(rows(["coma"] * 3 + ["awake"] * 2, ["awake"] * 5))
        self.assertEqual(m["mae_frames"], 3)
        self.assertEqual(m["early_recoveries"], 1)
        self.assertEqual(m["within_tolerance"]["1"]["hits"], 0)

    def test_human_annotation_gap_is_interval_not_exact_truth(self):
        m = score(
            rows(
                ["coma", None, None, "awake", "awake"],
                ["coma", "unsure", "awake", "awake", "awake"],
            )
        )
        self.assertEqual(m["observed_truth_recoveries"], 0)
        self.assertEqual(m["interval_truth_recoveries"], 1)
        self.assertIsNone(m["mae_frames"])
        self.assertEqual(m["mean_distance_to_truth_interval_frames"], 0)
        self.assertEqual(m["interval_within_tolerance"]["1"]["total"], 1)
        mixture = rows(
            ["coma", None, "awake"], ["coma", "awake", "awake"], "interval"
        ) + rows(["coma", "awake", "awake"], ["coma", "awake", "awake"], "exact")
        mixed = score(mixture)
        self.assertEqual(mixed["within_tolerance"]["1"]["total"], 1)
        self.assertEqual(mixed["interval_within_tolerance"]["1"]["total"], 1)

    def test_censored_inconsistent_and_partial_truth_are_not_negatives(self):
        for truth in [
            ["awake"] * 4,
            ["coma", "awake", "coma", "awake"],
            ["coma", None, None, None],
        ]:
            m = score(rows(truth, ["coma", "awake", "awake", "awake"]))
            self.assertEqual(m["measurable_truth_recoveries"], 0)
            self.assertEqual(m["false_recoveries"], 0)
        m = score(rows(["coma"] * 4, ["coma", "awake", "awake", "awake"]))
        self.assertEqual(m["false_recoveries"], 1)
        invalid = rows(["coma", "awake", "awake"], ["coma", "awake", "awake"])
        invalid[1]["scored"] = invalid[2]["scored"] = False
        self.assertEqual(score(invalid)["measurable_truth_recoveries"], 0)

    def test_browser_interval_is_understood_without_mutating_raw_manifest_bytes(self):
        m = {"frame_interval_seconds": 1.5}
        hydrate_manifest(m)
        self.assertEqual(m["interval"], 1.5)
