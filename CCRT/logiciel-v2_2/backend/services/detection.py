"""Editable boundary proposals; repeated-ring detection and arbitrary contours.

Grid mode uses repeated size/spacing evidence, not a fixed well count or layout.
It cannot semantically distinguish a mounting fixture with identical geometry.
"""
import json
import cv2
from backend.services.geometry import is_simple_polygon, clip_polygon
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
    # Weak interior rims need the relaxed pass too. Every added candidate must
    # still pass rim coverage, dominant size and repeated-neighbour checks.
    border = cv2.HoughCircles(contrast, cv2.HOUGH_GRADIENT_ALT, dp=1.5,
                             minDist=max(10, low*1.5), param1=250, param2=.5,
                             minRadius=low, maxRadius=high)
    candidates_found = [] if found is None else list(found[0])
    if border is not None:
        candidates_found.extend(c for c in border[0] if c[0]-c[2]<0 or c[1]-c[2]<0 or c[0]+c[2]>=w or c[1]+c[2]>=h)
    weak = cv2.HoughCircles(contrast, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=max(10, low*1.5), param1=200, param2=.65, minRadius=low, maxRadius=high)
    if weak is not None:
        candidates_found.extend(weak[0])
    if not candidates_found:
        return []
    edges = cv2.Canny(contrast, 40, 80)
    angles = np.linspace(0, 2*np.pi, 90, endpoint=False)
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
        if support < .45:
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


def _rim_polygon(gray, candidate, margin, neighbour_distance=None, edges=None):
    """Refine a circular seed to the outer observed elliptical rim.

    Radial sampling excludes interior fly edges. Robust ellipse fitting allows
    perspective changes; poor support retains the original circular proposal.
    Growth is bounded by neighbouring centers, not by a fixed tray geometry.
    """
    h, w = gray.shape
    x, y, radius = candidate['x'], candidate['y'], candidate['radius']
    if edges is None:
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 25, 65)
    angles = np.linspace(0, 2*np.pi, 120, endpoint=False)
    points = []
    for angle in angles:
        radii = np.linspace(.88*radius, 1.22*radius, max(12, round(radius*.7)))
        xs = np.rint(x+radii*np.cos(angle)).astype(int)
        ys = np.rint(y+radii*np.sin(angle)).astype(int)
        valid = (xs>=0)&(xs<w)&(ys>=0)&(ys<h)
        hits = np.flatnonzero(valid & (edges[np.clip(ys,0,h-1), np.clip(xs,0,w-1)]>0))
        if len(hits):
            r = radii[hits[-1]]
            points.append([x+r*np.cos(angle), y+r*np.sin(angle)])
    center, axes, rotation = (x,y), (radius,radius), 0.
    if len(points) >= 65:
        data = np.float32(points)
        fitted = None
        for _ in range(3):
            fitted = cv2.fitEllipse(data.reshape(-1,1,2))
            (cx,cy), (major,minor), degrees = fitted
            theta = np.deg2rad(degrees)
            delta = data-[cx,cy]
            u = delta[:,0]*np.cos(theta)+delta[:,1]*np.sin(theta)
            v = -delta[:,0]*np.sin(theta)+delta[:,1]*np.cos(theta)
            residual = np.abs(np.hypot(u/(major/2),v/(minor/2))-1)
            keep = residual < .08
            if keep.sum()<60: break
            data = data[keep]
        if fitted:
            (cx,cy),(major,minor),degrees=fitted
            if np.hypot(cx-x,cy-y)<.15*radius and .85*radius<min(major,minor)/2 and max(major,minor)/2<1.26*radius:
                center, axes, rotation = (cx,cy), (major/2,minor/2), np.deg2rad(degrees)
    axes = np.array(axes)*(1+margin/100)
    if neighbour_distance is not None:
        axes *= min(1., .49*neighbour_distance/max(axes))
    a = np.linspace(0,2*np.pi,40,endpoint=False)
    u,v=axes[0]*np.cos(a),axes[1]*np.sin(a)
    return [[(center[0]+u0*np.cos(rotation)-v0*np.sin(rotation))/w,
             (center[1]+u0*np.sin(rotation)+v0*np.cos(rotation))/h] for u0,v0 in zip(u,v)]


def _single_well(gray, diameter=None, region=None):
    """Find one dominant rim in a close crop, independently of grid repetition.

    Require a large rim near the crop center and evidence on at least half
    its circumference. No whole-image rectangle is substituted on failure.
    """
    h,w = gray.shape
    contrast=cv2.GaussianBlur(cv2.createCLAHE(2,(8,8)).apply(gray),(5,5),1)
    low,high=(max(5,int(diameter*.4)),max(6,int(diameter*.6))) if diameter else (int(min(h,w)*.3),int(max(h,w)*.68))
    found=cv2.HoughCircles(contrast,cv2.HOUGH_GRADIENT_ALT,1.5,max(10,min(h,w)*.3),param1=150,param2=.65,minRadius=low,maxRadius=high)
    if found is None:
        return []
    edges=cv2.Canny(contrast,40,80)
    angles=np.linspace(0,2*np.pi,180,endpoint=False)
    candidates=[]
    for x,y,r in found[0]:
        if abs(x-w/2)>.25*w or abs(y-h/2)>.25*h:
            continue
        covered=np.zeros(len(angles),bool)
        for offset in range(-3,4):
            xs=np.rint(x+(r+offset)*np.cos(angles)).astype(int)
            ys=np.rint(y+(r+offset)*np.sin(angles)).astype(int)
            valid=(xs>=0)&(xs<w)&(ys>=0)&(ys<h)
            covered[valid]|=edges[ys[valid],xs[valid]]>0
        support=float(covered.mean())
        if support<.5:
            continue
        if region and not (region[0]*w<=x-r and x+r<=region[2]*w and region[1]*h<=y-r and y+r<=region[3]*h):
            continue
        candidates.append({'x':float(x),'y':float(y),'radius':float(r),'support':support})
    # Prefer the outer rim over a concentric internal reflection.
    return sorted(candidates,key=lambda c:c['radius'],reverse=True)[:1]


def _clean_single_polygon(points, width, height):
    """Reject spikes and near-duplicate vertices relative to boundary size.

    Only automatic single-well proposals use these quality rules. Manual and
    grid boundaries retain the existing arbitrary simple-polygon contract.
    """
    polygon=np.asarray(points,float)*[width,height]
    span=np.ptp(polygon,axis=0)
    distance=max(1., min(span)*.015)
    area=lambda p:abs(float(cv2.contourArea(p.astype(np.float32))))
    original=area(polygon)
    if original < .12*width*height or min(span)<.45*min(width,height): return []
    while len(polygon)>3:
        edges=np.linalg.norm(polygon-np.roll(polygon,-1,axis=0),axis=1)
        if edges.min()>=distance:break
        candidate=np.delete(polygon,(int(edges.argmin())+1)%len(polygon),axis=0)
        if abs(area(candidate)-original)>.05*original:return []
        polygon=candidate
    for i,p in enumerate(polygon):
        a=polygon[i-1]-p;b=polygon[(i+1)%len(polygon)]-p
        cosine=np.dot(a,b)/max(1e-12,np.linalg.norm(a)*np.linalg.norm(b))
        if np.degrees(np.arccos(np.clip(cosine,-1,1))) < 25:return []
    result=(polygon/[width,height]).tolist()
    return result if is_simple_polygon(result) else []


def _contours(gray, single=False):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    proposals = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (.12 if single else .0005)*w*h < area < (.95 if single else .12)*w*h:
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


def detect_wells(content, method='auto', diameter=None, region=None, layout='grid', margin=4):
    if layout not in ('grid', 'individual'):
        raise ValueError('Unknown image layout.')
    try:
        margin = float(margin)
        if not np.isfinite(margin) or not 0 <= margin <= 20: raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('Rim margin must be between 0 and 20 percent.') from None
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
    circles = (_single_well(gray, diameter*scale if diameter else None, region) if layout=='individual' else _circles(gray,diameter*scale if diameter else None,region)) if method!='contours' else []
    wells = []
    if circles:
        rim_edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 25, 65)
        for c in circles:
            distance = min((np.hypot(c['x']-d['x'],c['y']-d['y']) for d in circles if d is not c), default=None)
            points = _rim_polygon(gray, c, margin, distance, rim_edges)
            if region is not None and not all(region[0]<=x<=region[2] and region[1]<=y<=region[3] for x,y in points):
                # Never silently extend a user-defined detection area.
                continue
            partial = not is_simple_polygon(points)
            points = clip_polygon(points)
            if not points:
                continue
            wells.append({'points':points,'partial':partial,'source':'single-rim-proposal' if layout=='individual' else 'repeated-ring-proposal','review_required':True,
                          'confidence':None,'ring_coverage':c['support']})
        used = 'single-rim' if layout=='individual' else 'grid'
    elif method=='grid':
        used = 'grid'
    else:
        proposals = _contours(gray, single=layout=='individual')
        if layout=='individual':
            proposals = [p for p in proposals if min(p['size'])>.5*min(sh,sw) and abs(p['center'][0]-sw/2)<.25*sw and abs(p['center'][1]-sh/2)<.25*sh]
            proposals = sorted(proposals,key=lambda p:p['size'][0]*p['size'][1],reverse=True)[:1]
        for proposal in proposals:
            if region is not None and not all(region[0]<=x/sw<=region[2] and region[1]<=y/sh<=region[3] for x,y in proposal['points']):
                continue
            wells.append({'points':[[x/sw,y/sh] for x,y in proposal['points']],
                          'source':'repaired-convex-contour' if proposal['repaired'] else 'simple-contour-proposal','review_required':True,'confidence':None})
        used = 'contours'
    if layout == "individual":
        valid=[]
        for well in wells:
            points=_clean_single_polygon(well["points"],sw,sh)
            if points:
                well["points"]=points
                valid.append(well)
        wells=valid[:1]
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
            'settings':{'method':method,'diameter_px':diameter,'region':region,'layout':layout,'rim_margin_percent':margin},
            'warning':'Review proposals; no fixed layout or well count is assumed. Partially visible rims are clipped to the image edge and flagged; their complete shape remains unknown. Restrict the detection region to exclude mounting hardware when geometry alone is ambiguous; contour mode preserves simple non-circular/concave boundaries.'}
