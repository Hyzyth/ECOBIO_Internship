"""Editable boundary proposals; repeated-ring detection and arbitrary contours.

Grid mode uses repeated size/spacing evidence, not a fixed well count or layout.
It cannot semantically distinguish a mounting fixture with identical geometry.
"""
import json
import cv2
from backend.services.geometry import is_simple_polygon
import numpy as np


def _circles(gray, diameter=None, region=None):
    h, w = gray.shape
    contrast = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    contrast = cv2.GaussianBlur(contrast, (5, 5), 1)
    if diameter:
        low, high = max(5, int(diameter * .4)), max(6, int(diameter * .6))
    else:
        seeds = [p for p in _contours(gray) if min(p['size']) / max(p['size']) > .8]
        if len(seeds) >= 6:
            radii = np.array([(p['size'][0]+p['size'][1])/4 for p in seeds])
            typical = max(radii,key=lambda r:np.sum(np.abs(np.log(radii/r)) < .14))
            low, high = max(6,int(typical*.8)),max(8,int(typical*1.2))
        else:
            low, high = max(6, int(min(h, w) * .012)), max(12, int(min(h, w) * .12))
    found = cv2.HoughCircles(contrast, cv2.HOUGH_GRADIENT_ALT, dp=1.5,
                            minDist=max(10, low * 1.5), param1=250, param2=.8,
                            minRadius=low, maxRadius=high)
    if found is None:
        return []
    edges = cv2.Canny(contrast, 40, 80)
    angles = np.linspace(0, 2*np.pi, 90, endpoint=False)
    candidates = []
    for x, y, radius in found[0]:
        # Ring coverage suppresses fly bodies/scratches, while allowing gaps.
        covered = np.zeros(len(angles), bool)
        for offset in [-3, -2, -1, 0, 1, 2, 3]:
            xs = np.rint(x + (radius + offset) * np.cos(angles)).astype(int)
            ys = np.rint(y + (radius + offset) * np.sin(angles)).astype(int)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            covered[valid] |= edges[ys[valid], xs[valid]] > 0
        support = float(covered.mean())
        if support < .45 or x-radius < 0 or y-radius < 0 or x+radius >= w or y+radius >= h:
            continue
        if region is not None and not (region[0]*w <= x-radius and x+radius <= region[2]*w and region[1]*h <= y-radius and y+radius <= region[3]*h):
            continue
        candidates.append({'x':float(x), 'y':float(y), 'radius':float(radius), 'support':support})
    # Suppress concentric duplicates before computing the dominant size.
    unique = []
    for c in sorted(candidates, key=lambda c:c['support'], reverse=True):
        if all(np.hypot(c['x']-d['x'],c['y']-d['y']) > .75*min(c['radius'],d['radius']) for d in unique):
            unique.append(c)
    if len(unique) < 6:
        return unique if diameter else []
    radii = np.array([c['radius'] for c in unique])
    dominant = max(radii, key=lambda r:np.sum(np.abs(np.log(radii/r)) < .16))
    group = [c for c in unique if .82*dominant <= c['radius'] <= 1.18*dominant]
    if len(group) < 6:
        return []
    centers = np.array([[c['x'],c['y']] for c in group])
    distances = np.linalg.norm(centers[:,None,:]-centers[None,:,:],axis=2)
    np.fill_diagonal(distances, np.inf)
    spacing = float(np.median(distances.min(axis=1)))
    # Repeated directions may have DIFFERENT spacings (staggered/rectangular
    # supports, perspective). A single nearest-distance band loses corner wells.
    near = (distances > .65*spacing)&(distances < 2.05*spacing)
    delta = centers[None,:,:]-centers[:,None,:]
    angles = np.mod(np.arctan2(delta[:,:,1],delta[:,:,0]),np.pi)
    bins = np.rint(angles/np.pi*36).astype(int)%36
    consistent = np.zeros_like(near)
    for direction in range(36):
        angular_distance = np.minimum((bins-direction)%36,(direction-bins)%36)
        samples = distances[near & (angular_distance<=2)]
        if len(samples)<max(4,len(group)*.12):
            continue
        # Infer the most repeated step in this direction; do not impose a
        # row count, hexagonal layout, or equal horizontal/vertical spacing.
        step = max(samples,key=lambda d:np.sum(np.abs(np.log(samples/d))<.1))
        repeated = np.abs(np.log(samples/step))<.1
        if np.sum(repeated)<max(4,len(group)*.12):
            continue
        consistent |= near & (bins==direction) & (distances>.87*step) & (distances<1.13*step)
    neighbours = np.sum(consistent,axis=1)
    return [c for c,n in zip(group,neighbours) if n >= 2]


def _contours(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    proposals = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not .0005*w*h < area < .12*w*h:
            continue
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if not hull_area or area/hull_area < .45:
            continue
        # Preserve concave shapes when the contour is simple. Repair a
        # retraced edge contour with a hull only when necessary, and label it.
        polygon = cv2.approxPolyDP(contour,.005*cv2.arcLength(contour,True),True)
        normalized=(polygon[:,0,:]/[w,h]).tolist()
        repaired = not is_simple_polygon(normalized)
        if repaired:
            polygon = cv2.approxPolyDP(hull,.005*cv2.arcLength(hull,True),True)
        if len(polygon) < 3:
            continue
        x,y,bw,bh = cv2.boundingRect(polygon)
        if min(bw,bh)/max(bw,bh) < .35:
            continue
        if any(abs(x+ bw/2 - p['center'][0]) < .25*min(bw,p['size'][0]) and
               abs(y+ bh/2 - p['center'][1]) < .25*min(bh,p['size'][1]) for p in proposals):
            continue
        proposals.append({'points':polygon[:,0,:].astype(float).tolist(),
                          'center':(x+bw/2,y+bh/2),'size':(bw,bh),'repaired':repaired})
    return proposals


def detect_wells(content, method='auto', diameter=None, region=None):
    if region not in (None,''):
        try:
            region=json.loads(region) if isinstance(region,str) else region
            if not isinstance(region,list) or len(region)!=4 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and np.isfinite(v) and 0<=v<=1 for v in region) or region[0]>=region[2] or region[1]>=region[3]:
                raise ValueError()
        except (ValueError,TypeError):
            raise ValueError('Detection region must be normalized [left, top, right, bottom].') from None
    else:
        region=None
    if method not in ('auto','grid','contours'):
        raise ValueError('Unknown detection method.')
    if diameter not in (None, ''):
        try:
            diameter = float(diameter)
        except (ValueError, TypeError):
            raise ValueError('Well diameter must be a positive pixel value.') from None
        if not np.isfinite(diameter) or diameter <= 0:
            raise ValueError('Well diameter must be a positive pixel value.')
    else:
        diameter = None
    try:
        image = cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None:
        raise ValueError('The selected file is not a readable image.')
    h,w = image.shape[:2]
    scale = min(1.,1200/max(h,w))
    small = cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
    sh,sw = small.shape[:2]
    gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    circles = _circles(gray,diameter*scale if diameter else None,region) if method!='contours' else []
    wells = []
    if circles:
        for c in circles:
            angles = np.linspace(0,2*np.pi,40,endpoint=False)
            points = [[(c['x']+c['radius']*np.cos(a))/sw,(c['y']+c['radius']*np.sin(a))/sh] for a in angles]
            wells.append({'points':points,'source':'repeated-ring-proposal','review_required':True,
                          'confidence':None,'ring_coverage':c['support']})
        used = 'grid'
    elif method=='grid':
        used = 'grid'
    else:
        for proposal in _contours(gray):
            if region is not None and not all(region[0]<=x/sw<=region[2] and region[1]<=y/sh<=region[3] for x,y in proposal['points']):
                continue
            wells.append({'points':[[x/sw,y/sh] for x,y in proposal['points']],
                          'source':'repaired-convex-contour' if proposal['repaired'] else 'simple-contour-proposal','review_required':True,'confidence':None})
        used = 'contours'
    # Cluster rows by centers, then assign IDs left-to-right, without fixed rows.
    center = lambda well:np.mean(well['points'],axis=0)
    wells.sort(key=lambda well:center(well)[1])
    rows=[]
    for well in wells:
        cy=float(center(well)[1])
        height=max(p[1] for p in well['points'])-min(p[1] for p in well['points'])
        if not rows or abs(cy-rows[-1]['y']) > .45*height:
            rows.append({'y':cy,'wells':[]})
        rows[-1]['wells'].append(well)
    wells=[well for row in rows for well in sorted(row['wells'],key=lambda well:center(well)[0])]
    return {'wells':wells,'width':w,'height':h,'method':used,
            'settings':{'method':method,'diameter_px':diameter,'region':region},
            'warning':'Review proposals; no fixed layout or well count is assumed. Complete boundaries only: clipped wells require manual selection. Restrict the detection region to exclude mounting hardware when geometry alone is ambiguous; contour mode preserves simple non-circular/concave boundaries.'}
