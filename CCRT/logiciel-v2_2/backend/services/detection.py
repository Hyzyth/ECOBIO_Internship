"""Contour proposals independent of circular or fixed-grid layouts."""
import cv2
import numpy as np

def detect_wells(content):
    try:
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        raise ValueError("The selected file is not a readable image.") from None
    if image is None:
        raise ValueError('The selected file is not a readable image.')
    h, w = image.shape[:2]
    scale = min(1., 1200 / max(h, w))
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = small.shape[0] * small.shape[1]
    wells = []
    for contour in contours:
        a = cv2.contourArea(contour)
        if not .002 * area < a < .8 * area:
            continue
        polygon = cv2.approxPolyDP(contour, .008 * cv2.arcLength(contour, True), True)
        if len(polygon) < 3:
            continue
        points = [[float(x) / scale / w, float(y) / scale / h] for [[x, y]] in polygon]
        wells.append({'points': points, 'source': 'contour-proposal', 'review_required': True, 'confidence': None})
    wells.sort(key=lambda well: (round(min(p[1] for p in well['points']) * 10), min(p[0] for p in well['points'])))
    return {'wells': wells, 'width': w, 'height': h,
            'warning': 'Boundary proposals only. Confirm identities and boundaries; detection does not establish occupancy or track camera movement.'}
