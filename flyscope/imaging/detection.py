"""Editable boundary proposals; repeated-ring detection and arbitrary contours.

Grid mode uses repeated size/spacing evidence, not a fixed well count or layout.
It cannot semantically distinguish a mounting fixture with identical geometry.
"""

import json
import cv2
from flyscope.imaging.geometry import is_simple_polygon, clip_polygon
import numpy as np

DETECTOR_VERSION = "native-outer-rims-v3"


def _circles(gray, diameter=None, region=None):
    h, w = gray.shape
    contrast = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    contrast = cv2.GaussianBlur(contrast, (5, 5), 1)
    if diameter:
        low, high = max(5, int(diameter * 0.4)), max(6, int(diameter * 0.6))
    else:
        seeds = [p for p in _contours(gray) if min(p["size"]) / max(p["size"]) > 0.8]
        if len(seeds) >= 6:
            radii = np.array([(p["size"][0] + p["size"][1]) / 4 for p in seeds])
            typical = max(radii, key=lambda r: np.sum(np.abs(np.log(radii / r)) < 0.14))
            low, high = max(6, int(typical * 0.8)), max(8, int(typical * 1.2))
        else:
            low, high = max(6, int(min(h, w) * 0.012)), max(12, int(min(h, w) * 0.12))
    found = cv2.HoughCircles(
        contrast,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=max(10, low * 1.5),
        param1=250,
        param2=0.8,
        minRadius=low,
        maxRadius=high,
    )
    # Weak interior rims need the relaxed pass too. Every added candidate must
    # still pass rim coverage, dominant size and repeated-neighbour checks.
    border = cv2.HoughCircles(
        contrast,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=max(10, low * 1.5),
        param1=250,
        param2=0.5,
        minRadius=low,
        maxRadius=high,
    )
    candidates_found = [] if found is None else list(found[0])
    if border is not None:
        candidates_found.extend(
            c
            for c in border[0]
            if c[0] - c[2] < 0
            or c[1] - c[2] < 0
            or c[0] + c[2] >= w
            or c[1] + c[2] >= h
        )
    weak = cv2.HoughCircles(
        contrast,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,
        minDist=max(10, low * 1.5),
        param1=200,
        param2=0.65,
        minRadius=low,
        maxRadius=high,
    )
    if weak is not None:
        candidates_found.extend(weak[0])
    if not candidates_found:
        return []
    edges = cv2.Canny(contrast, 40, 80)
    angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    candidates = []
    for x, y, radius in candidates_found:
        # Ring coverage suppresses fly bodies/scratches, while allowing gaps.
        covered = np.zeros(len(angles), bool)
        for offset in [-3, -2, -1, 0, 1, 2, 3]:
            xs = np.rint(x + (radius + offset) * np.cos(angles)).astype(int)
            ys = np.rint(y + (radius + offset) * np.sin(angles)).astype(int)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            covered[valid] |= edges[ys[valid], xs[valid]] > 0
        support = float(covered.mean())
        if support < 0.45:
            continue
        if region is not None and not (
            region[0] * w <= x - radius
            and x + radius <= region[2] * w
            and region[1] * h <= y - radius
            and y + radius <= region[3] * h
        ):
            continue
        candidates.append(
            {"x": float(x), "y": float(y), "radius": float(radius), "support": support}
        )
    # Suppress concentric duplicates before computing the dominant size.
    unique = []
    for c in sorted(candidates, key=lambda c: c["support"], reverse=True):
        if all(
            np.hypot(c["x"] - d["x"], c["y"] - d["y"])
            > 0.75 * min(c["radius"], d["radius"])
            for d in unique
        ):
            unique.append(c)
    if len(unique) < 6:
        return unique if diameter else []
    radii = np.array([c["radius"] for c in unique])
    dominant = max(radii, key=lambda r: np.sum(np.abs(np.log(radii / r)) < 0.16))
    group = [c for c in unique if 0.82 * dominant <= c["radius"] <= 1.18 * dominant]
    if len(group) < 6:
        return []
    centers = np.array([[c["x"], c["y"]] for c in group])
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(distances.min(axis=1)))
    # Repeated directions may have DIFFERENT spacings (staggered/rectangular
    # supports, perspective). A single nearest-distance band loses corner wells.
    near = (distances > 0.65 * spacing) & (distances < 2.05 * spacing)
    delta = centers[None, :, :] - centers[:, None, :]
    angles = np.mod(np.arctan2(delta[:, :, 1], delta[:, :, 0]), np.pi)
    bins = np.rint(angles / np.pi * 36).astype(int) % 36
    consistent = np.zeros_like(near)
    for direction in range(36):
        angular_distance = np.minimum((bins - direction) % 36, (direction - bins) % 36)
        samples = distances[near & (angular_distance <= 2)]
        if len(samples) < max(4, len(group) * 0.12):
            continue
        # Infer the most repeated step in this direction; do not impose a
        # row count, hexagonal layout, or equal horizontal/vertical spacing.
        step = max(samples, key=lambda d: np.sum(np.abs(np.log(samples / d)) < 0.1))
        repeated = np.abs(np.log(samples / step)) < 0.1
        if np.sum(repeated) < max(4, len(group) * 0.12):
            continue
        consistent |= (
            near
            & (bins == direction)
            & (distances > 0.87 * step)
            & (distances < 1.13 * step)
        )
    neighbours = np.sum(consistent, axis=1)
    return [c for c, n in zip(group, neighbours) if n >= 2]


def _ellipse_radii(center, axes, rotation, origin, directions):
    """Positive ray/ellipse intersections measured from a seed center."""
    cosine, sine = np.cos(rotation), np.sin(rotation)
    transform = np.array([[cosine, sine], [-sine, cosine]])
    delta = (np.asarray(origin) - center) @ transform.T / axes
    rays = directions @ transform.T / axes
    a = np.sum(rays * rays, axis=1)
    b = 2 * np.sum(rays * delta, axis=1)
    c = np.sum(delta * delta) - 1
    return (-b + np.sqrt(np.maximum(0, b * b - 4 * a * c))) / (2 * a)


def _rim_polygon(gray, candidate, margin, neighbours=()):
    """Fit observed outer lips at native resolution, independent of seed size.

    Hough often locks onto a smaller internal reflection. Search several
    coherent concentric models instead of selecting one outer pixel per ray.
    Neighbours constrain extra context only; they never shrink an observed rim.
    """
    h, w = gray.shape
    x, y, radius = candidate["x"], candidate["y"], candidate["radius"]
    reach = radius * 1.75 + 4
    left, top = max(0, int(x - reach)), max(0, int(y - reach))
    right, bottom = min(w, int(x + reach) + 1), min(h, int(y + reach) + 1)
    patch = gray[top:bottom, left:right]
    smooth = cv2.GaussianBlur(patch, (5, 5), 0.9)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    angles = np.linspace(0, 2 * np.pi, 160, endpoint=False)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    radii = np.linspace(
        0.76 * radius, 1.68 * radius, max(40, min(240, round(radius * 1.8)))
    )
    xs = x + directions[:, 0, None] * radii - left
    ys = y + directions[:, 1, None] * radii - top
    valid = (
        (xs >= 1) & (xs < patch.shape[1] - 1) & (ys >= 1) & (ys < patch.shape[0] - 1)
    )
    dx = cv2.remap(gx, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)
    dy = cv2.remap(gy, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR)
    magnitude = np.hypot(dx, dy)
    radial = np.abs(dx * directions[:, 0, None] + dy * directions[:, 1, None])
    noise = np.median(magnitude[valid]) if valid.any() else 0
    edge = valid & (magnitude > max(6.0, noise * 2.5)) & (radial > 0.72 * magnitude)
    band = max(1.5, radius * 0.065)
    models = []
    origin = np.array([x, y])
    for scale in np.arange(0.9, 1.61, 0.1):
        center, axes, rotation = origin.copy(), np.array([radius * scale] * 2), 0.0
        for _ in range(4):
            expected = _ellipse_radii(center, axes, rotation, origin, directions)
            distance = np.abs(radii[None, :] - expected[:, None])
            cost = np.where(edge & (distance <= band), distance, np.inf)
            chosen = cost.argmin(axis=1)
            covered = np.isfinite(cost[np.arange(len(angles)), chosen])
            if covered.sum() < 40:
                break
            points = origin + directions[covered] * radii[chosen[covered], None]
            fitted = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
            (cx, cy), (a, b), degrees = fitted
            if not (
                0.72 * radius < min(a, b) / 2
                and max(a, b) / 2 < 1.72 * radius
                and np.hypot(cx - x, cy - y) < 0.24 * radius
            ):
                break
            center, axes, rotation = (
                np.array([cx, cy]),
                np.array([a / 2, b / 2]),
                np.deg2rad(degrees),
            )
        expected = _ellipse_radii(center, axes, rotation, origin, directions)
        distance = np.abs(radii[None, :] - expected[:, None])
        covered = (edge & (distance <= max(1.25, radius * 0.045))).any(axis=1)
        visible = (
            (x + directions[:, 0] * expected >= 1)
            & (x + directions[:, 0] * expected < w - 1)
            & (y + directions[:, 1] * expected >= 1)
            & (y + directions[:, 1] * expected < h - 1)
        )
        support = float(covered.mean())
        # Evidence must span the visible quadrants, not just one fly edge.
        distributed = all(
            covered[q : q + 40].sum() >= min(8, visible[q : q + 40].sum() * 0.35)
            for q in range(0, 160, 40)
        )
        if (
            distributed
            and support >= 0.52
            and covered.sum() >= 0.62 * max(1, visible.sum())
        ):
            models.append((float(np.prod(axes)), center, axes, rotation, support))
    if models:
        _, center, axes, rotation, support = max(models, key=lambda m: m[0])
    else:
        center, axes, rotation, support = origin, np.array([radius, radius]), 0.0, 0.0
    candidate["outer_rim_support"] = support
    candidate["outer_radius"] = float(np.sqrt(np.prod(axes)))
    candidate["refined_center"] = center.tolist()
    # Add context outside the fitted rim, limiting only that added context
    # along each neighbouring direction. A close diagonal neighbour cannot
    # reduce both ellipse axes as the previous isotropic cap did.
    a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    unit = np.c_[np.cos(a), np.sin(a)]
    observed = _ellipse_radii(center, axes, rotation, center, unit)
    # A 5x5 Gaussian followed by a 3x3 derivative spans three source pixels.
    # At small resolutions a percentage alone can be less than one pixel and
    # clip wings/legs after resampling. Zero still requests the fitted rim only.
    padding = np.maximum(observed * margin / 100, 3.0) if margin else 0.0
    grown = observed + padding
    for neighbour in neighbours:
        delta = np.asarray(neighbour) - center
        dot = unit @ delta
        limit = np.divide(
            np.dot(delta, delta), 2 * dot, out=np.full(len(unit), np.inf), where=dot > 0
        )
        grown = np.minimum(grown, np.maximum(observed, limit))
    return ((center + unit * grown[:, None]) / [w, h]).tolist()


def _recover_rims(gray, circles, region=None):
    """Search repeated local displacements, then require actual rim evidence.

    Two non-collinear neighbouring wells must support a location. Unlike pair
    midpoints, repeated translations do not introduce half-row phantom sites
    in staggered grids. Only one pass is made; recovered wells cannot propagate
    an invented grid beyond the observed field.
    """
    if len(circles) < 8:
        return circles
    centers = np.array([[c["x"], c["y"]] for c in circles])
    delta = centers[None, :, :] - centers[:, None, :]
    distances = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(distances.min(axis=1)))
    radius = float(np.median([c["radius"] for c in circles]))
    vectors = delta[(distances > 0.72 * spacing) & (distances < 1.85 * spacing)]
    if not len(vectors):
        return circles
    from scipy.spatial import cKDTree

    tree = cKDTree(vectors)
    counts = tree.query_ball_point(vectors, 0.1 * spacing, return_length=True)
    steps = []
    for index in np.argsort(-counts):
        if counts[index] < max(4, len(circles) * 0.06):
            break
        if any(
            np.linalg.norm(vectors[index] - other) <= 0.22 * spacing for other in steps
        ):
            continue
        step = np.median(
            vectors[tree.query_ball_point(vectors[index], 0.1 * spacing)], axis=0
        )
        steps.append(step)
    votes = []
    for i, center in enumerate(centers):
        for step in steps:
            point = center + step
            if np.linalg.norm(centers - point, axis=1).min() < radius * 0.9:
                continue
            group = next(
                (
                    g
                    for g in votes
                    if np.linalg.norm(g["point"] - point) < radius * 0.25
                ),
                None,
            )
            if group is None:
                votes.append({"point": point, "anchors": {i}})
            else:
                group["anchors"].add(i)
    hull = cv2.convexHull(centers.astype(np.float32))
    result = list(circles)
    for vote in votes:
        point = vote["point"]
        angles = [
            np.arctan2(*(centers[i] - point)[::-1]) % np.pi for i in vote["anchors"]
        ]
        if not any(0.4 < abs(a - b) < np.pi - 0.4 for a in angles for b in angles):
            continue
        if (
            np.linalg.norm(
                np.array([[c["x"], c["y"]] for c in result]) - point, axis=1
            ).min()
            < radius * 0.9
        ):
            continue
        x, y = point
        if not (0 < x < gray.shape[1] and 0 < y < gray.shape[0]):
            continue
        candidate = {
            "x": float(x),
            "y": float(y),
            "radius": radius,
            "support": None,
            "recovered": True,
        }
        points = _rim_polygon(gray, candidate, 0, centers)
        outside = cv2.pointPolygonTest(hull, tuple(point), False) < 0
        minimum_support = 0.6 if outside or len(vote["anchors"]) >= 3 else 0.72
        if candidate["outer_rim_support"] < minimum_support:
            continue
        if not 0.82 * radius < candidate["outer_radius"] < 1.6 * radius:
            continue
        if region is not None and not all(
            region[0] <= a <= region[2] and region[1] <= b <= region[3]
            for a, b in points
        ):
            continue
        cx, cy = candidate["refined_center"]
        candidate.update(
            x=cx,
            y=cy,
            radius=candidate["outer_radius"],
            support=candidate["outer_rim_support"],
        )
        result.append(candidate)
    return result


def _single_well(gray, diameter=None, region=None):
    """Find one dominant rim in a close crop, independently of grid repetition.

    Require a large rim near the crop center and evidence on at least half
    its circumference. No whole-image rectangle is substituted on failure.
    """
    h, w = gray.shape
    contrast = cv2.GaussianBlur(cv2.createCLAHE(2, (8, 8)).apply(gray), (5, 5), 1)
    low, high = (
        (max(5, int(diameter * 0.4)), max(6, int(diameter * 0.6)))
        if diameter
        else (int(min(h, w) * 0.3), int(max(h, w) * 0.68))
    )
    found = cv2.HoughCircles(
        contrast,
        cv2.HOUGH_GRADIENT_ALT,
        1.5,
        max(10, min(h, w) * 0.3),
        param1=150,
        param2=0.65,
        minRadius=low,
        maxRadius=high,
    )
    if found is None:
        return []
    edges = cv2.Canny(contrast, 40, 80)
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    candidates = []
    for x, y, r in found[0]:
        if abs(x - w / 2) > 0.25 * w or abs(y - h / 2) > 0.25 * h:
            continue
        covered = np.zeros(len(angles), bool)
        for offset in range(-3, 4):
            xs = np.rint(x + (r + offset) * np.cos(angles)).astype(int)
            ys = np.rint(y + (r + offset) * np.sin(angles)).astype(int)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            covered[valid] |= edges[ys[valid], xs[valid]] > 0
        support = float(covered.mean())
        if support < 0.5:
            continue
        if region and not (
            region[0] * w <= x - r
            and x + r <= region[2] * w
            and region[1] * h <= y - r
            and y + r <= region[3] * h
        ):
            continue
        candidates.append(
            {"x": float(x), "y": float(y), "radius": float(r), "support": support}
        )
    # Prefer the outer rim over a concentric internal reflection.
    return sorted(candidates, key=lambda c: c["radius"], reverse=True)[:1]


def _clean_single_polygon(points, width, height):
    """Reject spikes and near-duplicate vertices relative to boundary size.

    Only automatic single-well proposals use these quality rules. Manual and
    grid boundaries retain the existing arbitrary simple-polygon contract.
    """
    polygon = np.asarray(points, float) * [width, height]
    span = np.ptp(polygon, axis=0)
    distance = max(1.0, min(span) * 0.015)
    area = lambda p: abs(float(cv2.contourArea(p.astype(np.float32))))
    original = area(polygon)
    if original < 0.12 * width * height or min(span) < 0.45 * min(width, height):
        return []
    while len(polygon) > 3:
        edges = np.linalg.norm(polygon - np.roll(polygon, -1, axis=0), axis=1)
        if edges.min() >= distance:
            break
        candidate = np.delete(polygon, (int(edges.argmin()) + 1) % len(polygon), axis=0)
        if abs(area(candidate) - original) > 0.05 * original:
            return []
        polygon = candidate
    for i, p in enumerate(polygon):
        a = polygon[i - 1] - p
        b = polygon[(i + 1) % len(polygon)] - p
        cosine = np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))
        if np.degrees(np.arccos(np.clip(cosine, -1, 1))) < 25:
            return []
    result = (polygon / [width, height]).tolist()
    return result if is_simple_polygon(result) else []


def _contours(gray, single=False):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    proposals = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if (
            not (0.12 if single else 0.0005) * w * h
            < area
            < (0.95 if single else 0.12) * w * h
        ):
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if not hull_area or area / hull_area < 0.45:
            continue
        # Preserve concave shapes when the contour is simple. Repair a
        # retraced edge contour with a hull only when necessary, and label it.
        polygon = cv2.approxPolyDP(contour, 0.005 * cv2.arcLength(contour, True), True)
        normalized = (polygon[:, 0, :] / [w, h]).tolist()
        repaired = not is_simple_polygon(normalized)
        if repaired:
            polygon = cv2.approxPolyDP(hull, 0.005 * cv2.arcLength(hull, True), True)
        if len(polygon) < 3:
            continue
        x, y, bw, bh = cv2.boundingRect(polygon)
        if min(bw, bh) / max(bw, bh) < 0.35:
            continue
        if any(
            abs(x + bw / 2 - p["center"][0]) < 0.25 * min(bw, p["size"][0])
            and abs(y + bh / 2 - p["center"][1]) < 0.25 * min(bh, p["size"][1])
            for p in proposals
        ):
            continue
        proposals.append(
            {
                "points": polygon[:, 0, :].astype(float).tolist(),
                "center": (x + bw / 2, y + bh / 2),
                "size": (bw, bh),
                "repaired": repaired,
            }
        )
    return proposals


def detect_wells(
    content, method="auto", diameter=None, region=None, layout="grid", margin=4
):
    if layout not in ("grid", "individual"):
        raise ValueError("Unknown image layout.")
    try:
        margin = float(margin)
        if not np.isfinite(margin) or not 0 <= margin <= 20:
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError("Rim margin must be between 0 and 20 percent.") from None
    if region not in (None, ""):
        try:
            region = json.loads(region) if isinstance(region, str) else region
            if (
                not isinstance(region, list)
                or len(region) != 4
                or not all(
                    isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and np.isfinite(v)
                    and 0 <= v <= 1
                    for v in region
                )
                or region[0] >= region[2]
                or region[1] >= region[3]
            ):
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError(
                "Detection region must be normalized [left, top, right, bottom]."
            ) from None
    else:
        region = None
    if method not in ("auto", "grid", "contours"):
        raise ValueError("Unknown detection method.")
    if diameter not in (None, ""):
        try:
            diameter = float(diameter)
        except (ValueError, TypeError):
            raise ValueError("Well diameter must be a positive pixel value.") from None
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("Well diameter must be a positive pixel value.")
    else:
        diameter = None
    try:
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None:
        raise ValueError("The selected file is not a readable image.")
    h, w = image.shape[:2]
    scale = min(1.0, 1200 / max(h, w))
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    circles = (
        (
            _single_well(gray, diameter * scale if diameter else None, region)
            if layout == "individual"
            else _circles(gray, diameter * scale if diameter else None, region)
        )
        if method != "contours"
        else []
    )
    wells = []
    if circles:
        if layout == "grid":
            circles = _recover_rims(gray, circles, region)
        native = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sx, sy = w / sw, h / sh
        for c in circles:
            seed = {
                **c,
                "x": c["x"] * sx,
                "y": c["y"] * sy,
                "radius": c["radius"] * (sx + sy) / 2,
            }
            neighbours = [[d["x"] * sx, d["y"] * sy] for d in circles if d is not c]
            points = _rim_polygon(native, seed, margin, neighbours)
            if region is not None and not all(
                region[0] <= x <= region[2] and region[1] <= y <= region[3]
                for x, y in points
            ):
                # Never silently extend a user-defined detection area.
                continue
            partial = not is_simple_polygon(points)
            points = clip_polygon(points)
            if not points:
                continue
            wells.append(
                {
                    "points": points,
                    "partial": partial,
                    "source": (
                        "single-rim-proposal"
                        if layout == "individual"
                        else "repeated-ring-proposal"
                    ),
                    "review_required": True,
                    "confidence": None,
                    "ring_coverage": c["support"],
                    "outer_rim_coverage": seed["outer_rim_support"],
                    "recovered_from_neighbours": bool(c.get("recovered")),
                }
            )
        used = "single-rim" if layout == "individual" else "grid"
    elif method == "grid":
        used = "grid"
    else:
        proposals = _contours(gray, single=layout == "individual")
        if layout == "individual":
            proposals = [
                p
                for p in proposals
                if min(p["size"]) > 0.5 * min(sh, sw)
                and abs(p["center"][0] - sw / 2) < 0.25 * sw
                and abs(p["center"][1] - sh / 2) < 0.25 * sh
            ]
            proposals = sorted(
                proposals, key=lambda p: p["size"][0] * p["size"][1], reverse=True
            )[:1]
        for proposal in proposals:
            if region is not None and not all(
                region[0] <= x / sw <= region[2] and region[1] <= y / sh <= region[3]
                for x, y in proposal["points"]
            ):
                continue
            wells.append(
                {
                    "points": [[x / sw, y / sh] for x, y in proposal["points"]],
                    "source": (
                        "repaired-convex-contour"
                        if proposal["repaired"]
                        else "simple-contour-proposal"
                    ),
                    "review_required": True,
                    "confidence": None,
                }
            )
        used = "contours"
    if layout == "individual":
        valid = []
        for well in wells:
            points = _clean_single_polygon(well["points"], sw, sh)
            if points:
                well["points"] = points
                valid.append(well)
        wells = valid[:1]
    # Cluster rows by centers, then assign IDs left-to-right, without fixed rows.
    center = lambda well: np.mean(well["points"], axis=0)
    wells.sort(key=lambda well: center(well)[1])
    rows = []
    for well in wells:
        cy = float(center(well)[1])
        height = max(p[1] for p in well["points"]) - min(p[1] for p in well["points"])
        if not rows or abs(cy - rows[-1]["y"]) > 0.45 * height:
            rows.append({"y": cy, "wells": []})
        rows[-1]["wells"].append(well)
    wells = [
        well
        for row in rows
        for well in sorted(row["wells"], key=lambda well: center(well)[0])
    ]
    return {
        "wells": wells,
        "width": w,
        "height": h,
        "method": used,
        "settings": {
            "method": method,
            "diameter_px": diameter,
            "region": region,
            "layout": layout,
            "rim_margin_percent": margin,
            "algorithm_version": DETECTOR_VERSION,
        },
        "warning": "Review proposals; no fixed layout or well count is assumed. Partially visible rims are clipped to the image edge and flagged; their complete shape remains unknown. Restrict the detection region to exclude mounting hardware when geometry alone is ambiguous; contour mode preserves simple non-circular/concave boundaries.",
    }
