"""Validate simple normalized polygons without restricting concave shapes."""
import math


def is_simple_polygon(points):
    eps = 1e-10
    if not isinstance(points, list) or len(points) < 3:
        return False
    if not all(isinstance(p,list) and len(p)==2 and all(
        not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in p) for p in points):
        return False
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a,b,p):
        return abs(cross(a,b,p))<=eps and min(a[0],b[0])-eps<=p[0]<=max(a[0],b[0])+eps and min(a[1],b[1])-eps<=p[1]<=max(a[1],b[1])+eps
    def opposite(a,b):
        return a>eps and b < -eps or b>eps and a < -eps
    def intersects(a,b,c,d):
        return opposite(cross(a,b,c),cross(a,b,d)) and opposite(cross(c,d,a),cross(c,d,b)) or on(a,b,c) or on(a,b,d) or on(c,d,a) or on(c,d,b)
    area=0
    n=len(points)
    for i,a in enumerate(points):
        b,previous=points[(i+1)%n],points[(i-1)%n]
        if math.dist(a,b)<=eps:
            return False
        if abs(cross(previous,a,b))<=eps and sum((previous[k]-a[k])*(b[k]-a[k]) for k in (0,1))>eps:
            return False
        area+=a[0]*b[1]-b[0]*a[1]
        for j in range(i+1,n):
            if j==i+1 or i==0 and j==n-1:
                continue
            if intersects(a,b,points[j],points[(j+1)%n]):
                return False
    return abs(area)>eps
