"""Scientific contracts: individual isolation, balancing and honest recovery."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
import torch

from tests.suite_fixtures import grids
from flyscope.training.comparison import main
from flyscope.training.data import TrainingData
from flyscope.training.plans import plan_for, checked_plan
from flyscope.models.recovery import apply_policy
from flyscope.training.validation import sample_weights, BestCheckpoint
from flyscope.training.pose_quality import (
    split_examples,
    heatmap_loss,
    heatmap_targets,
    augment_example,
    evaluate,
)


def annotated_grid(root):
    train, test = grids(root)
    manifest = json.loads((train / "annotations.json").read_text())
    originals = [a for a in manifest["annotations"] if a["well_uid"] == "well-0"]
    for well in manifest["wells"][3:]:
        for original in originals:
            a = copy.deepcopy(original)
            a.update(
                id=f"{well['well_uid']}-{a['frame_index']}",
                well_uid=well["well_uid"],
                well_id=well["well_id"],
                boundary_normalized=well["frames"][a["frame_index"]]["points"],
            )
            manifest["annotations"].append(a)
    (train / "annotations.json").write_text(json.dumps(manifest))
    return train, test


class ValidationProtocolTests(unittest.TestCase):
    def test_sixty_individual_protocol_has_ten_selections_and_shared_holdout(self):
        dataset = SimpleNamespace(sha="a" * 64)
        data = SimpleNamespace(
            datasets=[dataset],
            annotations={},
            occupied={},
            well_names={},
            identity=lambda d, value: d.sha[:16] + "::" + value,
        )
        for n in range(60):
            well = f"well-{n}"
            uid = data.identity(dataset, well)
            data.occupied[uid], data.well_names[uid] = True, f"W{n+1}"
            for frame, label in enumerate(["coma", "awake"]):
                data.annotations[f"{uid}-{frame}"] = (
                    dataset,
                    {"well_uid": well, "label": label},
                )
        plan = plan_for(data, poor="W1", rich="W2", validation_fraction=0.2)
        self.assertEqual(len(plan["runs"]), 10)
        self.assertEqual(len(plan["validation"]["well_uids"]), 12)
        self.assertEqual(len(plan["runs"][0]["well_uids"]), 48)
        self.assertEqual(
            [len(r["well_uids"]) for r in plan["runs"]],
            [48, 25, 10, 10, 5, 5, 2, 2, 1, 1],
        )
        for run in plan["runs"]:
            self.assertFalse(
                set(run["well_uids"]) & set(plan["validation"]["well_uids"])
            )

    def test_plan_command_migrates_a_copy_without_replacing_old_selections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, _ = annotated_grid(root / "fixture")
            source = plan_for(TrainingData([train]), 79, "W1", "W2")
            old = root / "old-plan.json"
            old.write_text(json.dumps(source))
            original = old.read_bytes()
            main(
                [
                    "plan",
                    "--train",
                    str(train),
                    "--source-plan",
                    str(old),
                    "--validation-wells",
                    "W4",
                    "--output",
                    str(root / "new-plan.json"),
                ]
            )
            self.assertEqual(old.read_bytes(), original)
            new = checked_plan(TrainingData([train]), root / "new-plan.json")
            self.assertEqual(new["seed"], 79)
            self.assertEqual(new["runs"][-2:], source["runs"][-2:])
            self.assertEqual(
                new["source_plan_sha256"], hashlib.sha256(original).hexdigest()
            )

    def test_shared_validation_is_outside_every_training_budget_and_keeps_all_frames(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            train, _ = annotated_grid(Path(temporary) / "fixture")
            data = TrainingData([train])
            plan = plan_for(data, 2026, "W1", "W2", validation_wells=["W4"])
            validation = plan["validation"]
            self.assertEqual(len(validation["annotation_ids"]), 6)
            for run in plan["runs"]:
                self.assertFalse(set(run["well_uids"]) & set(validation["well_uids"]))
                self.assertFalse(
                    set(run["annotation_ids"]) & set(validation["annotation_ids"])
                )
            self.assertEqual(
                len(
                    next(r for r in plan["runs"] if r["id"] == "poor-poses")[
                        "annotation_ids"
                    ]
                ),
                6,
            )
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(checked_plan(data, path), plan)
            damaged = copy.deepcopy(plan)
            damaged["validation"]["annotation_ids"].pop()
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "exactly"):
                checked_plan(data, path)
            damaged = copy.deepcopy(plan)
            damaged["runs"][0]["well_uids"] += validation["well_uids"]
            damaged["runs"][0]["annotation_ids"] += validation["annotation_ids"]
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "also appears"):
                checked_plan(data, path)

    def test_well_folds_cover_unprotected_individuals_once_without_frame_splitting(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            train, _ = annotated_grid(Path(temporary) / "fixture")
            data = TrainingData([train])
            held_out = []
            for index in range(3):
                plan = plan_for(
                    data, 42, "W1", "W2", validation_folds=3, validation_fold=index
                )
                held_out.extend(plan["validation"]["well_uids"])
                self.assertEqual(len(plan["validation"]["annotation_ids"]), 6)
            self.assertEqual(len(set(held_out)), 3)
            self.assertEqual(len(held_out), 3)

    def test_no_validation_requires_an_explicit_opt_out_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, test = grids(root / "fixture")
            with self.assertRaisesRegex(ValueError, "no independent validation"):
                main(
                    [
                        "compare",
                        "--train",
                        str(train),
                        "--test",
                        str(test),
                        "--architectures",
                        "resnet18",
                        "--output",
                        str(root / "out"),
                    ]
                )
            self.assertFalse((root / "out").exists())

    def test_class_and_individual_mass_does_not_follow_long_awake_sequences(self):
        annotations = {}
        for well, coma, awake in [("short", 2, 3), ("long", 4, 50)]:
            for state, count in [("coma", coma), ("awake", awake)]:
                for i in range(count):
                    annotations[f"{well}-{state}-{i}"] = (
                        None,
                        {"well_uid": well, "label": state},
                    )
        data = SimpleNamespace(
            annotations=annotations, identity=lambda dataset, well: well
        )
        weights = sample_weights(data, list(annotations), "well-class", 2)
        mass = {}
        for identity, value in weights.items():
            a = annotations[identity][1]
            key = a["well_uid"], a["label"]
            mass[key] = mass.get(key, 0) + value
        self.assertAlmostEqual(mass["short", "coma"], mass["long", "coma"])
        self.assertAlmostEqual(mass["short", "awake"], mass["long", "awake"])
        self.assertAlmostEqual(mass["short", "coma"] / mass["short", "awake"], 2)
        self.assertAlmostEqual(np.mean(list(weights.values())), 1)

    def test_checkpoint_restores_the_best_validation_epoch_not_the_last(self):
        model = torch.nn.Linear(1, 1, bias=False)
        best = BestCheckpoint(patience=2)
        with torch.no_grad():
            model.weight.fill_(1)
        self.assertFalse(best.update(0.4, 1, model))
        with torch.no_grad():
            model.weight.fill_(99)
        self.assertFalse(best.update(0.6, 2, model))
        self.assertTrue(best.update(0.7, 3, model))
        best.restore(model)
        self.assertEqual(float(model.weight.detach()[0, 0]), 1)
        self.assertEqual(best.epoch, 1)

    def test_initial_false_awake_cannot_lock_the_whole_sequence(self):
        states = ["awake"] * 3 + ["coma"] + ["awake"] * 3 + ["coma"]
        rows = [
            {"frame_index": i, "prediction": s, "confidence": 0.99}
            for i, s in enumerate(states)
        ]
        policy = {
            "confidence_threshold": 0.8,
            "awake_consecutive_frames": 3,
            "require_coma_before_awake": True,
        }
        apply_policy(rows, policy)
        self.assertEqual(
            [r["prediction"] for r in rows], ["unsure"] * 3 + ["coma"] + ["awake"] * 4
        )
        self.assertEqual(rows[4]["recovery_onset_frame_index"], 4)
        self.assertEqual(rows[4]["recovery_confirmed_frame_index"], 6)
        self.assertEqual(rows[-1]["raw_prediction"], "coma")
        # Re-applying the same policy retains raw evidence and the same result.
        apply_policy(rows, policy)
        self.assertEqual(
            [r["prediction"] for r in rows], ["unsure"] * 3 + ["coma"] + ["awake"] * 4
        )

    def test_invalid_pose_or_partial_frame_cannot_supply_a_coma_baseline(self):
        rows = [
            {
                "frame_index": 0,
                "prediction": "unknown",
                "raw_prediction": "coma",
                "confidence": None,
                "raw_confidence": 0.99,
                "decision_eligible": False,
            }
        ]
        rows += [
            {"frame_index": i, "prediction": "awake", "confidence": 0.99}
            for i in range(1, 5)
        ]
        apply_policy(
            rows,
            {
                "confidence_threshold": 0.8,
                "awake_consecutive_frames": 3,
                "require_coma_before_awake": True,
            },
        )
        self.assertTrue(all(r["prediction"] == "unsure" for r in rows))


class PoseQualityTests(unittest.TestCase):
    def examples(self, videos=1):
        return [
            (
                np.zeros((32, 32, 3), np.uint8),
                np.array([[8.0, 10.0], [18.0, 20.0]]),
                np.array([True, True]),
                {"video": f"v{v}", "frame": f * 10, "instance": i},
            )
            for v in range(videos)
            for f in range(12)
            for i in range(2)
        ]

    def test_pose_validation_keeps_paired_animals_and_a_purged_time_block(self):
        fit, validation, split = split_examples(self.examples(), 0.25, 42, 5)
        held = {e[3]["frame"] for e in validation}
        self.assertEqual(len(validation), len(held) * 2)
        for e in fit:
            self.assertGreater(min(abs(e[3]["frame"] - frame) for frame in held), 5)
        fit, validation, split = split_examples(self.examples(2), 0.5, 42, 5)
        self.assertFalse(
            {e[3]["video"] for e in fit} & {e[3]["video"] for e in validation}
        )
        self.assertEqual(split["unit"], "whole video")

    def test_blank_heatmaps_are_not_rewarded_by_the_new_loss(self):
        example = (
            np.zeros((128, 128, 3), np.uint8),
            np.array([[64.0, 64.0]]),
            np.array([True]),
            {},
        )
        target, visible = heatmap_targets([example], 128, 3, torch.device("cpu"))
        predicted = torch.zeros_like(target, requires_grad=True)
        legacy = (predicted - target).square().mean()
        balanced = heatmap_loss(predicted, target, visible)
        self.assertLess(float(legacy.detach()), 0.002)
        self.assertGreater(float(balanced.detach()), 0.05)
        balanced.backward()
        self.assertGreater(abs(float(predicted.grad[0, 0, 64, 64])), 0.001)

    def test_pose_localization_and_coverage_expose_blank_predictions(self):
        examples = self.examples()[:2]
        config = {"nodes": ["head", "thorax"], "size": 32}

        class Correct(torch.nn.Module):
            def forward(self, x):
                maps = torch.zeros((len(x), 2, 32, 32), device=x.device)
                maps[:, 0, 10, 8] = 1
                maps[:, 1, 20, 18] = 1
                return maps

        class Blank(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((len(x), 2, 32, 32), device=x.device)

        good, _ = evaluate(Correct(), examples, config, torch.device("cpu"))
        bad, _ = evaluate(Blank(), examples, config, torch.device("cpu"))
        self.assertEqual(good["pck_005"], 1)
        self.assertEqual(good["mean_error_px"], 0)
        self.assertEqual(bad["pck_005"], 0)
        self.assertEqual(bad["visible_prediction_coverage"], 0)

    def test_rotation_moves_anatomical_targets_with_the_image(self):
        image = np.zeros((16, 16, 3), np.uint8)
        image[5, 2] = 255
        rng = SimpleNamespace(integers=lambda *args: 1)
        rotated, points, _, _ = augment_example(
            (image, np.array([[2.0, 5.0]]), np.array([True]), {}), rng
        )
        np.testing.assert_array_equal(points, [[5, 13]])
        np.testing.assert_array_equal(rotated[13, 5], [255, 255, 255])


class AutomaticRecoveryTests(unittest.TestCase):
    def fixtures(self):
        dataset = SimpleNamespace(manifest={"interval": 1})
        ids, annotations, probabilities = [], {}, []
        for well in range(3):
            for frame in range(30):
                identity = f"{well}-{frame}"
                state = "coma" if frame < 8 else "awake"
                ids.append(identity)
                annotations[identity] = (
                    dataset,
                    {"frame_index": frame, "well_uid": str(well), "label": state},
                )
                probabilities.append(
                    [0.99, 0.01, 0] if state == "coma" else [0.01, 0.99, 0]
                )
        return (
            SimpleNamespace(annotations=annotations, identity=lambda d, well: well),
            ids,
            np.array(probabilities),
        )

    def test_automatic_selection_uses_validation_only_and_rejects_false_early_events(
        self,
    ):
        from flyscope.training.policy_selection import select_policy

        data, ids, probabilities = self.fixtures()
        policy, candidates = select_policy(data, ids, probabilities)
        self.assertTrue(policy["automatic_confirmation"])
        self.assertEqual(policy["selection"]["coverage"], 1)
        self.assertEqual(policy["selection"]["false_early_wells"], 0)
        self.assertEqual(policy["selection"]["normalized_confirmation_delay"], 0)
        self.assertEqual(len(candidates), 50)
        unusable, _ = select_policy(
            data, ids, probabilities, eligible=[False] * len(ids)
        )
        self.assertFalse(unusable["automatic_confirmation"])
        # Awake during every Coma frame except a baseline; wrong confident
        # predictions cannot be fixed simply by raising a threshold.
        wrong = np.tile([0.0, 1.0, 0.0], (len(ids), 1))
        for i in [0, 30, 60]:
            wrong[i] = [1, 0, 0]
        bad, _ = select_policy(data, ids, wrong)
        self.assertFalse(bad["automatic_confirmation"])
        self.assertEqual(bad["selection"]["status"], "no_safe_candidate")

    def test_insufficient_validation_or_invalid_probabilities_never_silently_select(
        self,
    ):
        from flyscope.training.policy_selection import select_policy

        data, ids, probabilities = self.fixtures()
        policy, _ = select_policy(data, ids[:30], probabilities[:30])
        self.assertFalse(policy["automatic_confirmation"])
        self.assertEqual(
            policy["selection"]["status"], "insufficient_validation_events"
        )
        with self.assertRaises(ValueError):
            select_policy(data, ids, probabilities[:2])
        probabilities[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            select_policy(data, ids, probabilities)

    def test_auto_abstention_and_interval_conversion_preserve_raw_outputs(self):
        from flyscope.models.recovery import effective_policy

        records = [
            {"frame_index": i, "prediction": state, "confidence": 0.99}
            for i, state in enumerate(["coma", "awake", "awake", "awake"])
        ]
        policy = {
            "confidence_threshold": 0.8,
            "awake_consecutive_frames": 3,
            "automatic_confirmation": False,
        }
        apply_policy(records, policy)
        self.assertEqual(
            [r["prediction"] for r in records], ["coma", "unsure", "unsure", "unsure"]
        )
        self.assertEqual(records[-1]["raw_prediction"], "awake")
        self.assertEqual(
            effective_policy({**policy, "awake_confirmation_seconds": 4}, 2)[
                "awake_consecutive_frames"
            ],
            3,
        )
        self.assertEqual(
            effective_policy({**policy, "awake_confirmation_seconds": 4}, 1)[
                "awake_consecutive_frames"
            ],
            5,
        )


class BoundedInferenceTests(unittest.TestCase):
    def test_matching_coma_brackets_resolve_only_internal_eligible_gaps(self):
        records = [
            {"frame_index": i, "prediction": state, "confidence": confidence}
            for i, (state, confidence) in enumerate(
                [
                    ("coma", 0.99),
                    ("awake", 0.6),
                    ("coma", 0.99),
                    ("coma", 0.6),
                    ("awake", 0.99),
                    ("awake", 0.99),
                ]
            )
        ]
        policy = {
            "confidence_threshold": 0.8,
            "awake_consecutive_frames": 2,
            "infer_matching_gaps": True,
        }
        apply_policy(records, policy)
        self.assertEqual(records[1]["prediction"], "coma")
        self.assertIsNone(records[1]["confidence"])
        self.assertEqual(records[1]["raw_prediction"], "awake")
        self.assertEqual(records[1]["temporal_inference"]["right_frame_index"], 2)
        self.assertEqual(records[3]["prediction"], "unsure")
        previous = copy.deepcopy(records)
        apply_policy(records, policy)
        self.assertEqual(records, previous)
        records[1]["decision_eligible"] = False
        apply_policy(records, policy)
        self.assertEqual(records[1]["prediction"], "unsure")
        self.assertNotIn("temporal_inference", records[1])
        sparse = [
            {"frame_index": 0, "prediction": "coma", "confidence": 0.99},
            {"frame_index": 2, "prediction": "awake", "confidence": 0.6},
            {"frame_index": 3, "prediction": "coma", "confidence": 0.99},
        ]
        apply_policy(sparse, policy)
        self.assertEqual(sparse[1]["prediction"], "unsure")


if __name__ == "__main__":
    unittest.main()
