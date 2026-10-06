"""Editable boundary proposals; repeated-ring detection and arbitrary contours.

Grid mode uses repeated size/spacing evidence, not a fixed well count or layout.
It cannot semantically distinguish a mounting fixture with identical geometry.
"""
import cv2
import numpy as np


def _circles(gray, diameter=None):
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
    found = cv2.HoughCircles(contrast, cv2.HOUGH_GRADIENT, dp=1.2,
                            minDist=max(10, low * 1.5), param1=80, param2=20,
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
    # Isolated circles and mounts away from the repeated lattice are excluded.
    near = (distances > .82*spacing)&(distances < 1.18*spacing)
    # Real lattice neighbours share repeated directions. This rejects mounting
    # rings that happen to be one spacing from a few wells but are off-lattice.
    delta = centers[None,:,:]-centers[:,None,:]
    angles = np.mod(np.arctan2(delta[:,:,1],delta[:,:,0]),np.pi)
    bins = np.rint(angles/np.pi*36).astype(int)%36
    histogram = np.bincount(bins[near],minlength=36)
    support = sum(np.roll(histogram,offset) for offset in [-2,-1,0,1,2])
    consistent = support[bins] >= max(4,len(group)*.12)
    neighbours = np.sum(near & consistent,axis=1)
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
        if not hull_area or area/hull_area < .8:
            continue
        # A hull has consistent winding and never retraces/crosses its edges.
        polygon = cv2.approxPolyDP(hull, .005*cv2.arcLength(hull,True), True)
        if len(polygon) < 3:
            continue
        x,y,bw,bh = cv2.boundingRect(polygon)
        if min(bw,bh)/max(bw,bh) < .35:
            continue
        if any(abs(x+ bw/2 - p['center'][0]) < .25*min(bw,p['size'][0]) and
               abs(y+ bh/2 - p['center'][1]) < .25*min(bh,p['size'][1]) for p in proposals):
            continue
        proposals.append({'points':polygon[:,0,:].astype(float).tolist(),
                          'center':(x+bw/2,y+bh/2),'size':(bw,bh)})
    return proposals


def detect_wells(content, method='auto', diameter=None):
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
    scale = min(1.,1400/max(h,w))
    small = cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
    sh,sw = small.shape[:2]
    gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    circles = _circles(gray,diameter*scale if diameter else None) if method!='contours' else []
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
            wells.append({'points':[[x/sw,y/sh] for x,y in proposal['points']],
                          'source':'convex-contour-proposal','review_required':True,'confidence':None})
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
            'warning':'Review proposals; repeated size and spacing suppress isolated fixtures but cannot identify every mounting feature. Adjust diameter, remove false proposals, or use contour mode for non-circular wells.'}
