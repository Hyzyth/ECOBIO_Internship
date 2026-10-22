"""CCRT event scoring: truth determines measurability, not model confidence gaps."""

from collections import defaultdict
import numpy as np

DEFAULT_TOLERANCES = (1, 5, 10, 30, 60)


def recovery_metrics(rows, manifest, first_recovery, tolerances=DEFAULT_TOLERANCES):
    tolerances = sorted(set(tolerances))
    if not tolerances or any(
        isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in tolerances
    ):
        raise ValueError("CCRT tolerances must be nonnegative integer frame counts.")
    groups = defaultdict(list)
    if manifest.get("mode") == "sequence":
        for r in rows:
            groups[r["well_id"]].append(r)
    results = []
    interval = manifest.get("interval", manifest.get("frame_interval_seconds"))
    origin = manifest.get("time_origin") or {}
    for uid, group in sorted(groups.items()):
        group.sort(key=lambda r: r["frame_index"])
        if not any(r.get("scored") for r in group):
            continue
        truth_rows = [
            {**r, "ground_truth": r.get("ground_truth") if r.get("scored") else None}
            for r in group
        ]
        truth, predicted = first_recovery(truth_rows, True), first_recovery(group)
        awake = next((r for r in truth_rows if r.get("ground_truth") == "awake"), None)
        before = [
            r["frame_index"]
            for r in truth_rows
            if awake
            and r["frame_index"] < awake["frame_index"]
            and r.get("ground_truth") == "coma"
        ]
        bounds = None
        if (
            awake
            and before
            and not any(
                r.get("ground_truth") == "coma"
                and r["frame_index"] > awake["frame_index"]
                for r in truth_rows
            )
        ):
            bounds = [max(before) + 1, awake["frame_index"]]
        # An estimated onset remains scoreable even when the previous model
        # decision was Unsure. Do not select a flattering subset of predictions.
        estimate = predicted["frame_index"]
        confirmation = next(
            (
                r.get("recovery_confirmed_frame_index")
                for r in group
                if r.get("recovery_confirmed_frame_index") is not None
            ),
            None,
        )
        exact = bounds is not None and bounds[0] == bounds[1]
        error = estimate - bounds[0] if exact and estimate is not None else None
        distance = (
            max(bounds[0] - estimate, 0, estimate - bounds[1])
            if bounds and estimate is not None
            else None
        )

        def seconds(frame):
            return (
                origin.get("offset_seconds", 0) + frame * interval
                if frame is not None and interval and origin.get("confirmed")
                else None
            )

        results.append(
            {
                "well_uid": uid,
                "well_id": group[0]["well_name"],
                "species": group[0].get("species"),
                "ground_truth": truth,
                "predicted": predicted,
                "truth_bounds_frames": bounds,
                "truth_measurement": (
                    "exact" if exact else "interval" if bounds else truth["status"]
                ),
                "estimated_onset_frame_index": estimate,
                "confirmation_frame_index": confirmation,
                "error_frames": error,
                "distance_to_truth_interval_frames": distance,
                "ground_truth_ccrt_seconds": seconds(bounds[0]) if exact else None,
                "predicted_ccrt_seconds": seconds(estimate),
                "missed_recovery": bool(bounds and estimate is None),
                "early_recovery": bool(
                    bounds and estimate is not None and estimate < bounds[0]
                ),
                "false_recovery": estimate is not None
                and truth["status"] == "not_observed",
            }
        )
    exact = [r for r in results if r["truth_measurement"] == "exact"]
    bounded = [r for r in results if r["truth_bounds_frames"]]
    errors = [r["error_frames"] for r in exact if r["error_frames"] is not None]
    intervals = [r for r in bounded if r["truth_measurement"] == "interval"]
    distances = [
        r["distance_to_truth_interval_frames"]
        for r in intervals
        if r["distance_to_truth_interval_frames"] is not None
    ]
    detected = sum(r["estimated_onset_frame_index"] is not None for r in bounded)
    return {
        "definition_version": 2,
        "observed_truth_recoveries": len(exact),
        "interval_truth_recoveries": len(bounded) - len(exact),
        "measurable_truth_recoveries": len(bounded),
        "detected_recoveries": detected,
        "event_coverage": detected / len(bounded) if bounded else None,
        "paired_observed_wells": len(errors),
        "mae_frames": float(np.mean(np.abs(errors))) if errors else None,
        "median_absolute_error_frames": (
            float(np.median(np.abs(errors))) if errors else None
        ),
        "bias_frames": float(np.mean(errors)) if errors else None,
        "mae_seconds": (
            float(np.mean(np.abs(errors))) * interval if errors and interval else None
        ),
        "mean_distance_to_truth_interval_frames": (
            float(np.mean(distances)) if distances else None
        ),
        "within_tolerance": {
            str(t): {
                "hits": sum(abs(e) <= t for e in errors),
                "total": len(exact),
                "fraction": (
                    sum(abs(e) <= t for e in errors) / len(exact) if exact else None
                ),
            }
            for t in tolerances
        },
        "interval_within_tolerance": {
            str(t): {
                "hits": sum(e <= t for e in distances),
                "total": len(intervals),
                "fraction": (
                    sum(e <= t for e in distances) / len(intervals)
                    if intervals
                    else None
                ),
            }
            for t in tolerances
        },
        "missed_recoveries": sum(r["missed_recovery"] for r in results),
        "early_recoveries": sum(r["early_recovery"] for r in results),
        "false_recoveries": sum(r["false_recovery"] for r in results),
        "unscorable_truth_wells": len(results)
        - len(bounded)
        - sum(r["ground_truth"]["status"] == "not_observed" for r in results),
        "censored_predicted_recoveries": sum(
            r["predicted"]["status"] == "already_awake_left_censored" for r in results
        ),
        "uncertain_predicted_recoveries": sum(
            r["predicted"]["status"] == "uncertain_interval" for r in results
        ),
        "wells": results,
        "definition": "Success within ±X frames uses every exact human recovery as denominator, including missing model events as failures. MAE is conditional on detected events; prediction uncertainty never excludes an estimated onset. Interval-censored truth is reported separately as distance to a permissible interval, not exact error.",
    }
