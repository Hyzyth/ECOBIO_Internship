"""Choose recovery settings on reserved validation individuals, never test grids.

Objective: meet empirical false-early/coverage constraints, then minimize missed
events and delay. No candidate passing the constraints means no automatic event
confirmation, not an invented Coma label or an optimistic accuracy claim.
"""

from collections import defaultdict
import math
import numpy as np
from backend.services.recovery_policy import apply_policy, effective_policy

LABELS = ["coma", "awake", "empty"]


def select_policy(
    data,
    ids,
    probabilities,
    threshold=0.8,
    frames=3,
    require_coma=True,
    false_early_limit=0.05,
    minimum_coverage=0.8,
    eligible=None,
):
    probabilities = np.asarray(probabilities, dtype=float)
    if (
        probabilities.shape != (len(ids), 3)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(1), 1, atol=1e-5)
    ):
        raise ValueError(
            "Policy selection requires one finite three-class probability vector per validation target."
        )
    if eligible is None:
        eligible = [True] * len(ids)
    if len(eligible) != len(ids):
        raise ValueError("Policy eligibility must match validation target count.")
    base = {
        "confidence_threshold": threshold,
        "awake_consecutive_frames": frames,
        "require_coma_before_awake": require_coma,
        "automatic_confirmation": False,
        "source": "validation",
        "infer_matching_gaps": True,
    }
    grouped = defaultdict(list)
    intervals = []
    for identity, p, usable in zip(ids, probabilities, eligible):
        dataset, a = data.annotations[identity]
        interval = dataset.manifest.get("interval")
        if interval and interval > 0:
            intervals.append(interval)
        grouped[data.identity(dataset, a["well_uid"])].append(
            {
                "frame_index": a["frame_index"],
                "truth": a["label"],
                "raw_prediction": LABELS[int(np.argmax(p))],
                "raw_confidence": float(max(p)),
                "interval": interval,
                "decision_eligible": bool(usable),
            }
        )
    truths = {}
    for well, rows in grouped.items():
        rows.sort(key=lambda r: r["frame_index"])
        awake = next((i for i, r in enumerate(rows) if r["truth"] == "awake"), None)
        coma = (
            [r["frame_index"] for r in rows[:awake] if r["truth"] == "coma"]
            if awake is not None
            else []
        )
        if awake is not None and any(r["truth"] == "coma" for r in rows[awake + 1 :]):
            base["selection"] = {
                "status": "inconsistent_validation_labels",
                "message": "Validation contains Coma after Awake; review those manual annotations.",
            }
            return base, []
        if awake is not None and coma:
            truths[well] = {
                "low": max(coma) + 1,
                "high": rows[awake]["frame_index"],
                "event": True,
            }
        elif awake is None and any(r["truth"] == "coma" for r in rows):
            truths[well] = {"event": False}
        elif all(r["truth"] == "empty" for r in rows):
            truths[well] = {"event": False}
    events = sum(t["event"] for t in truths.values())
    occupied = sum(any(r["truth"] == "coma" for r in grouped[w]) for w in truths)
    if occupied < 2 or not events:
        base["selection"] = {
            "status": "insufficient_validation_events",
            "validation_wells": len(grouped),
            "informative_occupied_wells": occupied,
            "observed_recoveries": events,
            "message": "Automatic policy selection needs at least two independent Coma-containing validation individuals and an annotated recovery.",
        }
        return base, []
    reference = min(intervals) if len(intervals) == len(ids) else None
    candidates = []
    for confidence in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.999]:
        for count in [1, 3, 5, 10, 20]:
            policy = {
                **base,
                "confidence_threshold": confidence,
                "awake_consecutive_frames": count,
                "automatic_confirmation": True,
            }
            if reference is not None:
                policy["awake_confirmation_seconds"] = (count - 1) * reference
            false_early = missed = detected = 0
            errors = []
            confirmation_delays = []
            for well, truth in truths.items():
                source = grouped[well]
                records = [
                    {
                        "frame_index": r["frame_index"],
                        "prediction": r["raw_prediction"],
                        "confidence": r["raw_confidence"],
                        "decision_eligible": r["decision_eligible"],
                    }
                    for r in source
                ]
                applied = effective_policy(policy, source[0]["interval"])
                apply_policy(records, applied)
                onset = next(
                    (
                        r["recovery_onset_frame_index"]
                        for r in records
                        if r["recovery_onset_frame_index"] is not None
                    ),
                    None,
                )
                if onset is not None and (not truth["event"] or onset < truth["low"]):
                    false_early += 1
                elif truth["event"]:
                    if onset is None:
                        missed += 1
                    else:
                        detected += 1
                        # Normalize delay by observed sequence length; no claim
                        # of an exact onset inside an uncertain annotation gap.
                        errors.append(
                            max(0, onset - truth["high"])
                            / max(
                                1, source[-1]["frame_index"] - source[0]["frame_index"]
                            )
                        )
                        confirmation = next(
                            r["recovery_confirmed_frame_index"]
                            for r in records
                            if r["recovery_confirmed_frame_index"] is not None
                        )
                        confirmation_delays.append(
                            max(0, confirmation - truth["high"])
                            / max(
                                1, source[-1]["frame_index"] - source[0]["frame_index"]
                            )
                        )
            coverage = detected / events
            rate = false_early / len(truths)
            candidates.append(
                {
                    "confidence_threshold": confidence,
                    "awake_consecutive_frames": count,
                    "awake_confirmation_seconds": (
                        (count - 1) * reference if reference is not None else None
                    ),
                    "false_early_wells": false_early,
                    "informative_wells": len(truths),
                    "false_early_rate": rate,
                    "detected_recoveries": detected,
                    "observed_recoveries": events,
                    "missed_recoveries": missed,
                    "coverage": coverage,
                    "normalized_mean_delay": float(np.mean(errors)) if errors else None,
                    "normalized_confirmation_delay": (
                        float(np.mean(confirmation_delays))
                        if confirmation_delays
                        else None
                    ),
                    "acceptable": rate <= false_early_limit
                    and coverage >= minimum_coverage,
                }
            )
    acceptable = [c for c in candidates if c["acceptable"]]
    # Minimize confirmation latency after onset accuracy, then prefer confidence.
    pool = acceptable or candidates
    chosen = min(
        pool,
        key=lambda c: (
            c["false_early_wells"],
            -c["detected_recoveries"],
            (
                c["normalized_mean_delay"]
                if c["normalized_mean_delay"] is not None
                else math.inf
            ),
            (
                c["normalized_confirmation_delay"]
                if c["normalized_confirmation_delay"] is not None
                else math.inf
            ),
            -c["awake_consecutive_frames"],
            -c["confidence_threshold"],
        ),
    )
    base.update(
        {k: chosen[k] for k in ("confidence_threshold", "awake_consecutive_frames")}
    )
    if chosen["awake_confirmation_seconds"] is not None:
        base["awake_confirmation_seconds"] = chosen["awake_confirmation_seconds"]
    base["automatic_confirmation"] = bool(acceptable)
    base["selection"] = {
        **chosen,
        "status": "selected" if acceptable else "no_safe_candidate",
        "false_early_limit": false_early_limit,
        "minimum_coverage": minimum_coverage,
        "candidate_count": len(candidates),
        "objective": "Validation individuals only: constrain false early events and recovery coverage, then minimize missed events and normalized delay.",
        "note": "Empirical validation rates on a small cohort are not statistical guarantees. Test data were not used.",
    }
    return base, candidates
