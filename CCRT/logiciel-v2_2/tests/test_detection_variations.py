import unittest
import cv2
import numpy as np
from backend.services.detection import detect_wells


def scene():
    rng = np.random.default_rng(52)
    h,w = 800,1000
    base = np.tile(np.linspace(100,170,w,dtype=np.float32),(h,1)) + rng.normal(0,2,(h,w))
    image = np.uint8(np.clip(base,0,255)); centers=[]
    for row in range(8):
        for col in range(8):
            if (row,col) in {(2,2),(3,5),(6,3)}:
                continue
            dx,dy=rng.integers(-2,3,2)
            x,y=140+col*90+(row%2)*45+dx,100+row*80+dy
            centers.append(np.array([x,y]))
            cv2.circle(image,(int(x),int(y)),28,205,2)
            cv2.ellipse(image,(int(x),int(y)),(12,5),30,0,360,40,-1)
    for x,y in [(70,50),(500,35),(920,730)]:
        for radius in [40,33,24]:cv2.circle(image,(x,y),radius,205,2)
    return image,centers


class DetectionVariations(unittest.TestCase):
    def test_anisotropic_staggered_spacing_keeps_corner_wells(self):
        image=np.zeros((650,900),np.uint8)
        expected=[]
        for row in range(8):
            for col in range(7 if row%2==0 else 6):
                x,y=80+col*110+(row%2)*55,90+row*65
                expected.append(np.array([x,y]));cv2.circle(image,(x,y),26,220,2)
        self.check(image,expected,52)

    def test_concave_non_circular_boundary_is_preserved(self):
        image=np.zeros((500,700),np.uint8)
        polygon=np.int32([[60,70],[230,70],[230,130],[130,130],[130,260],[60,260]])
        cv2.polylines(image,[polygon],True,255,4)
        result=detect_wells(cv2.imencode('.png',image)[1].tobytes(),method='contours')
        self.assertEqual(len(result['wells']),1)
        self.assertEqual(result['wells'][0]['source'],'simple-contour-proposal')
        boundary=np.float32(result['wells'][0]['points'])*[700,500]
        self.assertFalse(cv2.isContourConvex(boundary.astype(np.float32).reshape(-1,1,2)))

    def test_detection_region_excludes_boundary_proposals(self):
        image,_=scene()
        result=detect_wells(cv2.imencode('.png',image)[1].tobytes(),method='grid',diameter=56,region=[.1,.2,.8,.8])
        self.assertGreater(len(result['wells']),10)
        self.assertTrue(all(.1<=p[0]<=.8 and .2<=p[1]<=.8 for well in result['wells'] for p in well['points']))
        with self.assertRaises(ValueError): detect_wells(b'',region=[1,0,0,1])

    def check(self,image,expected,diameter=None):
        h,w=image.shape[:2]
        result=detect_wells(cv2.imencode('.png',image)[1].tobytes(),method='grid',diameter=diameter)
        actual=[np.mean(well['points'],axis=0)*[w,h] for well in result['wells']]
        self.assertEqual(len(actual),len(expected))
        self.assertTrue(all(any(np.linalg.norm(center-point)<8 for center in actual) for point in expected))
    def test_missing_wells_and_small_position_variations(self):
        image,expected=scene();self.check(image,expected)
    def test_diameter_override(self):
        image,expected=scene();self.check(image,expected,56)
    def test_resolution_change(self):
        image,expected=scene()
        image=cv2.resize(image,None,fx=2,fy=2)
        self.check(image,[p*2 for p in expected],112)
