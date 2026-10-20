"""Whole-individual validation and explicit objectives for the offline trainers.

No function in this module reads test data. Validation wells are reserved before
the training budgets are drawn, including the one-individual experiments.
"""

from collections import Counter, defaultdict
import hashlib
import json
import math
import random

import numpy as np

CLASSES = ["coma", "awake", "empty"]


def resolve_wells(data, values):
    available = {data.identity(d, a["well_uid"]) for d, a in data.annotations.values()}
    result = []
    for value in values:
        matching = [w for w in available if w == value or data.well_names[w] == value]
        if len(matching) != 1:
            raise ValueError(
                f"Well {value!r} is absent or ambiguous; use its qualified ID."
            )
        result.append(matching[0])
    return sorted(set(result))


def make_validation(
    data, seed, fraction=0.2, protected=(), wells=None, folds=None, fold=0, unit="well"
):
    if not 0 <= fraction < 1 or unit not in ("well", "grid"):
        raise ValueError(
            "Use a validation fraction in [0,1) and a well/grid grouping unit."
        )
    if folds is not None and (folds < 2 or not 0 <= fold < folds):
        raise ValueError(
            "Use at least two validation folds and a valid zero-based fold index."
        )
    all_wells = sorted(
        {data.identity(d, a["well_uid"]) for d, a in data.annotations.values()}
    )
    if fraction == 0 and not wells and folds is None:
        return None
    groups = defaultdict(set)
    for dataset, a in data.annotations.values():
        well = data.identity(dataset, a["well_uid"])
        groups[dataset.sha if unit == "grid" else well].add(well)
    protected = set(protected)
    empty = {w for w in all_wells if not data.occupied[w]}
    # A sole Empty well cannot simultaneously support training and validation.
    # Keep it available to training and explicitly report absent validation support.
    if len(empty) == 1:
        protected |= empty
    if wells:
        chosen = set(resolve_wells(data, wells))
        if unit == "grid":
            chosen = set().union(*(g for g in groups.values() if g & chosen))
        if chosen & set(protected) - empty:
            raise ValueError(
                "Validation cannot contain the protected poor/rich training individuals."
            )
    else:
        candidates = sorted(k for k, group in groups.items() if not group & protected)
        random.Random(seed).shuffle(candidates)
        if not candidates:
            raise ValueError(
                "No independent validation group remains. Annotate more wells, use another validation grid, or explicitly opt out for a software smoke test."
            )
        if folds is not None:
            if len(candidates) < folds:
                raise ValueError(
                    "Not enough unprotected groups for the requested validation folds."
                )
            selected = [k for i, k in enumerate(candidates) if i % folds == fold]
        else:
            n = max(1, round(len(groups) * fraction))
            selected = candidates[: min(n, len(candidates))]
        chosen = set().union(*(groups[k] for k in selected))
    fit = set(all_wells) - chosen
    if not chosen or not fit or not any(data.occupied[w] for w in fit):
        raise ValueError(
            "Validation must leave at least one occupied training well and contain an independent group."
        )
    ids = sorted(
        k
        for k, (d, a) in data.annotations.items()
        if data.identity(d, a["well_uid"]) in chosen
    )
    support = dict(Counter(data.annotations[k][1]["label"] for k in ids))
    missing = [label for label in CLASSES if not support.get(label)]
    if not any(data.occupied[w] for w in chosen):
        raise ValueError(
            "Validation needs an occupied well; an Empty-only split cannot select a coma/awake model."
        )
    return {
        "kind": "whole-group-validation-v1",
        "unit": unit,
        "seed": seed,
        "fraction": fraction if folds is None and not wells else None,
        "folds": folds,
        "fold_index": fold if folds else None,
        "well_uids": sorted(chosen),
        "annotation_ids": ids,
        "annotation_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "class_support": support,
        "missing_classes": missing,
        "group_count": sum(bool(group & chosen) for group in groups.values()),
        "scope": (
            "held-out individuals in the training grid(s)"
            if unit == "well"
            else "held-out training experiment(s)"
        ),
        "test_data_used": False,
    }


def validate_split(data, plan):
    validation = plan.get("validation")
    if validation is None:
        return
    wells = validation.get("well_uids", [])
    available = {data.identity(d, a["well_uid"]) for d, a in data.annotations.values()}
    if (
        validation.get("unit") not in ("well", "grid")
        or len(set(wells)) != len(wells)
        or not set(wells) <= available
    ):
        raise ValueError(
            "Validation must reference unique known well IDs and a well/grid grouping unit."
        )
    expected = {
        k
        for k, (d, a) in data.annotations.items()
        if data.identity(d, a["well_uid"]) in wells
    }
    ids = validation.get("annotation_ids", [])
    if not expected or len(ids) != len(expected) or set(ids) != expected:
        raise ValueError(
            "Validation must contain exactly the eligible labels of its saved whole wells."
        )
    if (
        validation.get("annotation_ids_sha256")
        != hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()
    ):
        raise ValueError(
            "Validation annotation fingerprint differs from its saved split."
        )
    for run in plan["runs"]:
        if set(run["well_uids"]) & set(wells) or set(run["annotation_ids"]) & expected:
            raise ValueError(
                "A validation individual also appears in a training selection."
            )
    if validation.get("unit") == "grid":
        validation_grids = {data.annotations[k][0].sha for k in ids}
        if any(
            data.annotations[k][0].sha in validation_grids
            for r in plan["runs"]
            for k in r["annotation_ids"]
        ):
            raise ValueError("Grid-grouped validation shares a grid with training.")


def sample_weights(data, ids, mode="well-class", false_awake_cost=2, occupancy=False):
    labels = (
        [2 if data.annotations[k][1]["label"] == "empty" else 0 for k in ids]
        if occupancy
        else [CLASSES.index(data.annotations[k][1]["label"]) for k in ids]
    )
    groups = [
        data.identity(data.annotations[k][0], data.annotations[k][1]["well_uid"])
        for k in ids
    ]
    counts = Counter(labels)
    per_well = Counter(zip(labels, groups))
    wells_per_class = Counter(label for label, _ in per_well)
    values = []
    for label, well in zip(labels, groups):
        value = 1.0
        if mode == "class":
            value /= counts[label]
        elif mode == "well-class":
            value /= wells_per_class[label] * per_well[label, well]
        elif mode != "none":
            raise ValueError("Unknown balancing objective.")
        value *= false_awake_cost if label in ((2,) if occupancy else (0, 2)) else 1
        values.append(value)
    weights = np.asarray(values, np.float64)
    weights /= weights.mean()
    return dict(zip(ids, weights))


def validation_metrics(probabilities, labels, weights):
    probabilities = np.asarray(probabilities, np.float64)
    labels = np.asarray(labels)
    weights = np.asarray(weights)
    if probabilities.shape != (len(labels), 3) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid validation probabilities.")
    losses = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-9, 1))
    predictions = probabilities.argmax(1)
    recalls = {
        CLASSES[k]: float((predictions[labels == k] == k).mean())
        for k in np.unique(labels)
    }
    return {
        "balanced_nll": float(np.average(losses, weights=weights)),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "class_recall": recalls,
        "false_awake_rate_on_coma": (
            float((predictions[labels == 0] == 1).mean())
            if (labels == 0).any()
            else None
        ),
        "samples": len(labels),
    }


class BestCheckpoint:
    """One CPU snapshot, restored after a phase; never selected using test loss."""

    def __init__(self, patience=3, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.epoch = None
        self.state = None
        self.stale = 0

    def update(self, value, epoch, network):
        if not math.isfinite(value):
            raise ValueError("Nonfinite validation loss.")
        if value < self.best - self.min_delta:
            self.best, self.epoch, self.stale = value, epoch, 0
            self.state = {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }
        else:
            self.stale += 1
        return self.patience > 0 and self.stale >= self.patience

    def restore(self, network):
        if self.state is not None:
            network.load_state_dict(self.state)
