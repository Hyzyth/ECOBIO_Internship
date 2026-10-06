"""Camera alignment proposals: select the simplest well-supported transform."""
import cv2
import numpy as np
from backend.services.model_registry import parse_wells
from backend.services.geometry import is_simple_polygon, clip_polygon


def align_wells(reference_bytes, current_bytes, wells_json):
    wells = parse_wells(wells_json)
    def decode(content):
        try:
            img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
        except cv2.error:
            img = None
        if img is None:
            raise ValueError('Unreadable alignment image.')
        h,w = img.shape
        scale = min(1.,1200/max(h,w))
        return cv2.resize(img,(max(1,round(w*scale)),max(1,round(h*scale))))
    reference,current = decode(reference_bytes),decode(current_bytes)
    orb = cv2.ORB_create(nfeatures=3500)
    kr,dr = orb.detectAndCompute(reference,None)
    kc,dc = orb.detectAndCompute(current,None)
    if dr is None or dc is None:
        raise ValueError('Insufficient visual features. Correct boundaries manually.')
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dr,dc,k=2)
    matches = [p[0] for p in pairs if len(p)==2 and p[0].distance < .7*p[1].distance]
    if len(matches)<12:
        raise ValueError('Too few reliable matches. Correct boundaries manually.')
    src = np.float32([kr[m.queryIdx].pt for m in matches])
    dst = np.float32([kc[m.trainIdx].pt for m in matches])
    candidates=[]
    partial,mask = cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3)
    if partial is not None:
        candidates.append(('similarity',np.vstack([partial,[0,0,1]]),mask))
    affine,mask = cv2.estimateAffine2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=3)
    if affine is not None:
        candidates.append(('affine',np.vstack([affine,[0,0,1]]),mask))
    homography,mask = cv2.findHomography(src,dst,cv2.RANSAC,3)
    if homography is not None:
        candidates.append(('perspective',homography,mask))
    accepted=[]
    for kind,matrix,mask in candidates:
        inliers=mask.ravel().astype(bool);ratio=float(inliers.mean())
        if ratio < .6 or inliers.sum()<12 or not np.isfinite(matrix).all():
            continue
        extent=np.ptp(src[inliers],axis=0)/[reference.shape[1],reference.shape[0]]
        if np.any(extent < .15):
            continue
        error=np.linalg.norm(cv2.perspectiveTransform(src.reshape(-1,1,2),matrix)[:,0]-dst,axis=1)
        median=float(np.median(error[inliers]));p95=float(np.percentile(error[inliers],95))
        if p95>3 or median>1.5:
            continue
        accepted.append({'method':kind,'matrix':matrix,'inliers':inliers,'inlier_ratio':ratio,
                         'median_error_px':median,'p95_error_px':p95})
    if not accepted:
        raise ValueError('Alignment is unreliable or matches cover too little of the image. Correct boundaries manually.')
    chosen=accepted[0]
    for candidate in accepted[1:]:
        # Upgrade only for a measurable gain, rather than warping every frame
        # with a more flexible transform when simple camera motion suffices.
        if candidate['inlier_ratio']>=chosen['inlier_ratio']-.02 and candidate['median_error_px']<chosen['median_error_px']*.75 and chosen['median_error_px']-candidate['median_error_px']>.15:
            chosen=candidate
    matrix=chosen['matrix']
    corners=np.float32([[[0,0],[reference.shape[1],0],[reference.shape[1],reference.shape[0]],[0,reference.shape[0]]]])
    projected=cv2.perspectiveTransform(corners,matrix)[0]
    if not cv2.isContourConvex(projected.reshape(-1,1,2)) or cv2.contourArea(projected,oriented=True)<=0:
        raise ValueError('Alignment folds or reverses the image. Correct boundaries manually.')
    area=cv2.contourArea(projected)/(current.shape[0]*current.shape[1])
    if not .25 < area < 4:
        raise ValueError('Alignment scale is implausible. Correct boundaries manually.')
    result=[]
    for well in wells:
        points=np.float32(well['points'])*[reference.shape[1],reference.shape[0]]
        transformed=cv2.perspectiveTransform(points.astype(np.float32).reshape(-1,1,2),matrix)[:,0]
        normalized=(transformed/[current.shape[1],current.shape[0]]).tolist()
        partial = not is_simple_polygon(normalized)
        normalized = clip_polygon(normalized)
        if not normalized:
            raise ValueError('A transformed boundary is entirely out of view, invalid, or cannot be clipped unambiguously. Correct boundaries manually; no alignment was saved.')
        result.append({'id':well['id'],'points':normalized,'partial':partial})
    metrics={k:v for k,v in chosen.items() if k not in ('matrix','inliers')}
    return {'wells':result,**metrics,'working_image_size':[current.shape[1],current.shape[0]],'warning':'Alignment proposals retain identities; review boundaries before analysis. Error metrics are measured on matched image features, not biological accuracy.'}
