"""Feature-based camera alignment proposals; fail explicitly on weak matches."""
import cv2
import numpy as np
from backend.services.model_registry import parse_wells

def align_wells(reference_bytes, current_bytes, wells_json):
    wells = parse_wells(wells_json)
    def decode(content):
        try:
            img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
        except cv2.error:
            raise ValueError("Unreadable alignment image.") from None
        if img is None:
            raise ValueError('Unreadable alignment image.')
        h, w = img.shape
        scale = min(1., 1200 / max(h, w))
        return cv2.resize(img, (max(1, round(w*scale)), max(1, round(h*scale))))
    reference, current = decode(reference_bytes), decode(current_bytes)
    orb = cv2.ORB_create(nfeatures=2500)
    kr, dr = orb.detectAndCompute(reference, None)
    kc, dc = orb.detectAndCompute(current, None)
    if dr is None or dc is None:
        raise ValueError('Insufficient visual features. Correct boundaries manually.')
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dr, dc, k=2)
    good = [pair[0] for pair in matches if len(pair)==2 and pair[0].distance < .7 * pair[1].distance]
    if len(good) < 12:
        raise ValueError('Too few reliable matches. Correct boundaries manually.')
    src = np.float32([kr[m.queryIdx].pt for m in good])
    dst = np.float32([kc[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
    ratio = float(np.mean(inliers)) if inliers is not None else 0
    if matrix is None or ratio < .6 or int(np.sum(inliers)) < 12:
        raise ValueError('Alignment is unreliable. Correct boundaries manually.')
    # Require spatially distributed evidence, not features from one tiny region.
    accepted = src[inliers.ravel().astype(bool)]
    extent = np.ptp(accepted, axis=0)
    if np.any(extent < np.array([reference.shape[1], reference.shape[0]]) * .15):
        raise ValueError('Matches cover too little of the image. Correct boundaries manually.')
    result = []
    for well in wells:
        points = np.float64(well['points']) * [reference.shape[1], reference.shape[0]]
        transformed = points @ matrix[:, :2].T + matrix[:, 2]
        normalized = transformed / [current.shape[1], current.shape[0]]
        if not np.isfinite(normalized).all() or np.any(normalized < 0) or np.any(normalized > 1):
            raise ValueError('A transformed well falls outside the image. Correct boundaries manually.')
        result.append({'id':well['id'], 'points':normalized.tolist()})
    return {'wells':result, 'inlier_ratio':ratio, 'warning':'Alignment proposal only. Review and confirm all affected boundaries.'}
