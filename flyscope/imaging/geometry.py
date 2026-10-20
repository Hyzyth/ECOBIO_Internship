"""Validate and parse normalized polygons without restricting concave shapes."""

import json
import math


def is_simple_polygon(points):
    eps = 1e-10
    if not isinstance(points, list) or len(points) < 3:
        return False
    if not all(
        isinstance(p, list)
        and len(p) == 2
        and all(
            not isinstance(v, bool)
            and isinstance(v, (int, float))
            and math.isfinite(v)
            and 0 <= v <= 1
            for v in p
        )
        for p in points
    ):
        return False

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on(a, b, p):
        return (
            abs(cross(a, b, p)) <= eps
            and min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
        )

    def opposite(a, b):
        return a > eps and b < -eps or b > eps and a < -eps

    def intersects(a, b, c, d):
        return (
            opposite(cross(a, b, c), cross(a, b, d))
            and opposite(cross(c, d, a), cross(c, d, b))
            or on(a, b, c)
            or on(a, b, d)
            or on(c, d, a)
            or on(c, d, b)
        )

    area = 0
    n = len(points)
    for i, a in enumerate(points):
        b, previous = points[(i + 1) % n], points[(i - 1) % n]
        if math.dist(a, b) <= eps:
            return False
        if (
            abs(cross(previous, a, b)) <= eps
            and sum((previous[k] - a[k]) * (b[k] - a[k]) for k in (0, 1)) > eps
        ):
            return False
        area += a[0] * b[1] - b[0] * a[1]
        for j in range(i + 1, n):
            if j == i + 1 or i == 0 and j == n - 1:
                continue
            if intersects(a, b, points[j], points[(j + 1) % n]):
                return False
    return abs(area) > eps


def clip_polygon(points):
    """Intersect a convex boundary with the normalized image rectangle.

    Concave polygons entirely inside remain untouched; ambiguous clipped
    concave shapes require manual correction rather than joining components.
    """
    if is_simple_polygon(points):
        return points
    import cv2
    import numpy as np

    polygon = np.asarray(points, dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
        return []
    if not cv2.isContourConvex(polygon.reshape(-1, 1, 2)):
        return []
    # OpenCV float32 intersection can return FLT_MAX vertices for near-tangent
    # edges. Clamping those values creates a polygon spanning the whole image.
    # Sutherland–Hodgman in double precision preserves the actual intersection.
    result = [[float(x), float(y)] for x, y in points]
    for axis, edge, sign in [(0, 0, 1), (0, 1, -1), (1, 0, 1), (1, 1, -1)]:
        output = []
        for i, a in enumerate(result):
            b = result[(i + 1) % len(result)]
            inside_a, inside_b = (
                sign * (a[axis] - edge) >= 0,
                sign * (b[axis] - edge) >= 0,
            )
            if inside_a:
                output.append(a)
            if inside_a != inside_b:
                t = (edge - a[axis]) / (b[axis] - a[axis])
                crossing = [a[k] + t * (b[k] - a[k]) for k in (0, 1)]
                crossing[axis] = edge
                output.append(crossing)
        result = output
    result = [
        p
        for i, p in enumerate(result)
        if math.dist(p, result[(i + 1) % len(result)]) > 1e-10
    ]
    return result if is_simple_polygon(result) else []


def parse_wells(wells_json):
    try:
        wells = json.loads(wells_json)
        if not isinstance(wells, list) or not wells:
            raise ValueError()
        ids = set()
        for well in wells:
            if not isinstance(well, dict):
                raise ValueError()
            identity = well.get("id")
            points = well.get("points")
            if not isinstance(identity, str) or not identity or identity in ids:
                raise ValueError()
            ids.add(identity)
            if not is_simple_polygon(points):
                raise ValueError()
    except (TypeError, ValueError):
        raise ValueError("Invalid well geometry.") from None
    return wells
