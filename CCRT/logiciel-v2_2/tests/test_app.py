import io
import json
import unittest
import cv2
import numpy as np
from app import app
from backend.services.model_registry import ADAPTERS

class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        image = np.zeros((500, 700, 3), np.uint8)
        cv2.rectangle(image, (50, 70), (200, 230), (255, 255, 255), 4)
        cv2.ellipse(image, (420, 260), (90, 60), 20, 0, 360, (255, 255, 255), 4)
        self.image = cv2.imencode('.png', image)[1].tobytes()
    def test_app_without_model(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        response = self.client.get('/static/review.js')
        self.assertEqual(response.status_code, 200)
        response.close()
        models = self.client.get('/api/models').json
        self.assertEqual(models[0]['id'], 'manual')
    def test_non_circular_detection(self):
        response = self.client.post('/api/detect', data={'image':(io.BytesIO(self.image),'test.png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json['wells']), 2)
        for well in response.json['wells']:
            self.assertTrue(well['review_required'])
            self.assertIsNone(well['confidence'])
            self.assertTrue(all(0 <= v <= 1 for pt in well['points'] for v in pt))
    def test_alignment_and_weak_features(self):
        rng = np.random.default_rng(123)
        reference = np.zeros((500, 700, 3), np.uint8)
        for x, y in rng.integers([20,20],[650,450],size=(200,2)):
            cv2.circle(reference,(int(x),int(y)),4,(255,255,255),-1)
        current = cv2.warpAffine(reference,np.float32([[1,0,15],[0,1,10]]),(700,500))
        encode = lambda img: cv2.imencode('.png',img)[1].tobytes()
        wells = [{'id':'stable-id','points':[[.2,.2],[.4,.2],[.4,.4],[.2,.4]]}]
        response = self.client.post('/api/align',data={
            'reference':(io.BytesIO(encode(reference)),'ref.png'),
            'image':(io.BytesIO(encode(current)),'current.png'), 'wells':json.dumps(wells)})
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(response.json['wells'][0]['id'],'stable-id')
        self.assertAlmostEqual(response.json['wells'][0]['points'][0][0],.2+15/700,delta=.005)
        blank=encode(np.zeros_like(reference))
        response=self.client.post('/api/align',data={'reference':(io.BytesIO(blank),'ref.png'),
            'image':(io.BytesIO(blank),'current.png'),'wells':json.dumps(wells)})
        self.assertEqual(response.status_code,400)

    def test_invalid_image(self):
        self.assertEqual(self.client.post('/api/detect').status_code, 400)
        self.assertEqual(self.client.post('/api/detect',data={'image':(io.BytesIO(b''),'empty.png')}).status_code,400)
        self.assertEqual(self.client.post('/api/detect',data={'image':(io.BytesIO(b'bad'),'x.png')}).status_code,400)
    def test_no_fake_predictions(self):
        response=self.client.post('/api/analyze',data={'model':'manual','image':(io.BytesIO(self.image),'x.png')})
        self.assertEqual(response.status_code,400)
    def test_adapter_contract(self):
        class Adapter:
            metadata={'name':'Test classifier','version':'test-only','task':'coma'}
            def predict(self,content,wells):
                return [{'well_id':w['id'],'prediction':'coma','confidence':.94} for w in wells]
        ADAPTERS['test']=Adapter()
        try:
            wells=[{'id':'a','points':[[0,0],[1,0],[1,1]]}]
            def send(wells):
                return self.client.post('/api/analyze',data={'model':'test','wells':json.dumps(wells),'image':(io.BytesIO(self.image),'x.png')})
            response=send(wells)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json['model']['version'],'test-only')
            self.assertEqual(response.json['records'][0]['confidence'],.94)
            self.assertEqual(send([{'id':'a','points':[]}]).status_code,400)
            ADAPTERS['test'].predict=lambda *_:[{'well_id':'a','prediction':'coma','confidence':float('nan')}]
            self.assertEqual(send(wells).status_code,400)
        finally:
            ADAPTERS.clear()

if __name__=='__main__': unittest.main()
