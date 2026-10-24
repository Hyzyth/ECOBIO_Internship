"""Optional, deterministic fly isolation. Source images are never modified.

Contrast is a heuristic: dust overlapping the fly and faint limbs cannot be
reliably separated without anatomical labels. Ambiguous/empty crops retain the
well rather than silently erasing a possible animal.
"""

import cv2
import numpy as np

MASKED = "masked-rgb-square-white-0-1-v1"
ISOLATED = "masked-fly-contrast-rgb-square-white-0-1-v1"
PREPROCESSINGS = (MASKED, ISOLATED)


def isolate(crop, mask):
    interior = mask > 0
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    core = distance > max(2, min(mask.shape) * 0.06)
    if core.sum() < 25:
        return crop.copy(), "insufficient_interior"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    median = np.median(gray[core])
    mad = np.median(np.abs(gray[core] - median))
    dark = gray < median - max(18, 4 * mad)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (dark & core).astype(np.uint8)
    )
    candidates = sorted(
        range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True
    )
    minimum = max(4, int(core.sum() * 0.0005))
    if not candidates or stats[candidates[0], 4] < minimum:
        return crop.copy(), "no_reliable_foreground"
    best = candidates[0]
    area = stats[best, 4]
    if area > core.sum() * 0.3 or (
        len(candidates) > 1 and stats[candidates[1], 4] >= area * 0.75
    ):
        return crop.copy(), "ambiguous_foreground"
    # Retain nearby weak contrast and disconnected legs; never use opening,
    # which would remove thin appendages. Growth is bounded to avoid a rim flood.
    radius = max(3, round(min(mask.shape) * 0.07))
    neighborhood = (
        cv2.dilate(
            (labels == best).astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            ),
        )
        > 0
    )
    keep = neighborhood & interior & (gray < median - max(6, 1.5 * mad))
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    result = np.full_like(crop, 255)
    result[keep & interior] = crop[keep & interior]
    return result, "isolated"
