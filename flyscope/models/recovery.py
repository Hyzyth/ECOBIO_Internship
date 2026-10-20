"""Sequence decisions are separate from raw class probabilities.

New validated models require observed Coma before an irreversible Awake event.
Missing fields retain legacy behavior for immutable older checkpoints.
"""

import math


def effective_policy(policy, interval=None):
    result = dict(policy)
    seconds = result.get("awake_confirmation_seconds")
    if seconds is not None and interval and interval > 0:
        result["awake_consecutive_frames"] = max(
            1, math.ceil(seconds / interval - 1e-9) + 1
        )
    return result


def apply_policy(records, policy):
    threshold = policy["confidence_threshold"]
    count = policy["awake_consecutive_frames"]
    streak = 0
    confirmed = None
    onset = None
    previous = None
    require_coma = policy.get("require_coma_before_awake", False)
    confirmation_enabled = policy.get("automatic_confirmation", True)
    observed_coma = False
    for index, r in enumerate(records):
        if previous is not None and r["frame_index"] != previous + 1:
            streak = 0
        previous = r["frame_index"]
        raw = r.get("raw_prediction", r["prediction"])
        eligible = r.setdefault(
            "decision_eligible",
            not r.get("boundary_partial", False)
            and (
                r["confidence"] is not None
                or r.get("recovery_confirmed_frame_index") is not None
            ),
        )
        confidence = r.get("raw_confidence", r["confidence"]) if eligible else None
        r.pop("recovery_status", None)
        r.pop("temporal_inference", None)
        r["raw_prediction"] = raw
        r.setdefault("raw_confidence", confidence)
        r["prediction"] = raw
        r["confidence"] = confidence
        if confirmed is not None:
            r["prediction"] = "awake"
            r["confidence"] = None
        elif confidence is None or confidence < threshold:
            r["prediction"] = "unsure"
            streak = 0
        elif raw == "awake":
            if not confirmation_enabled or (require_coma and not observed_coma):
                r["prediction"] = "unsure"
                r["recovery_status"] = (
                    "policy_not_validated"
                    if not confirmation_enabled
                    else "no_observed_coma_baseline"
                )
                r["recovery_confirmed_frame_index"] = None
                r["recovery_onset_frame_index"] = None
                streak = 0
                continue
            streak += 1
            if streak >= count:
                confirmed = r["frame_index"]
                onset = records[index - streak + 1]["frame_index"]
                for candidate in records[index - streak + 1 : index + 1]:
                    candidate["prediction"] = "awake"
                    candidate["recovery_confirmed_frame_index"] = confirmed
                    candidate["recovery_onset_frame_index"] = onset
            else:
                r["prediction"] = "unsure"
        else:
            if raw == "coma":
                observed_coma = True
            streak = 0
        r["recovery_confirmed_frame_index"] = confirmed
        r["recovery_onset_frame_index"] = onset
    if policy.get("infer_matching_gaps", False):
        infer_matching_gaps(records, threshold)


def infer_matching_gaps(records, threshold):
    """Offline inference between matching confident anchors; no new confidence.

    Only continuous, eligible predictions can be bridged. This never trains a
    model or resolves a Coma-to-Awake transition; source probabilities stay raw.
    """
    anchor = None
    previous = None
    pending = []
    for record in records:
        frame = record["frame_index"]
        if previous is not None and frame != previous + 1:
            anchor, pending = None, []
        previous = frame
        if record.get("decision_eligible") is False or record.get("boundary_partial"):
            anchor, pending = None, []
            continue
        state = record["prediction"]
        score = record.get("confidence")
        confident = state in ("coma", "awake") and (
            score is not None
            and score >= threshold
            or record.get("recovery_confirmed_frame_index") is not None
        )
        if confident:
            if anchor and anchor["prediction"] == state:
                for item in pending:
                    item["prediction"] = state
                    item["confidence"] = None
                    item["temporal_inference"] = {
                        "method": "matching_brackets",
                        "state": state,
                        "left_frame_index": anchor["frame_index"],
                        "right_frame_index": frame,
                        "assumption": "single_irreversible_recovery",
                    }
            anchor, pending = record, []
        elif state == "unsure" and anchor:
            pending.append(record)
        else:
            anchor, pending = None, []
