"""Local research UI; trained model adapters are optional."""
from pathlib import Path
import cv2

# Bound OpenCV workers; large host CPU counts can slow interactive Hough calls.
cv2.setNumThreads(2)
from flask import Flask, jsonify, render_template, request
from backend.services.detection import detect_wells
from backend.services.alignment import align_wells
from backend.services.model_registry import available_models, analyze
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / 'frontend/templates'), static_folder=str(BASE_DIR / 'frontend/static'))
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
@app.get('/')
def index():
    return render_template('index.html')
@app.get('/api/models')
def models():
    return jsonify(available_models())
@app.post('/api/detect')
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify(error='Select an image first.'), 400
    try:
        return jsonify(detect_wells(image.read(), request.form.get('method','auto'), request.form.get('diameter'), request.form.get('region'), request.form.get('layout','grid')))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
@app.post('/api/align')
def align():
    reference, current = request.files.get('reference'), request.files.get('image')
    if not reference or not current:
        return jsonify(error='Select reference and current images.'), 400
    try:
        return jsonify(align_wells(reference.read(), current.read(), request.form.get('wells', '[]')))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

@app.post('/api/analyze')
def predict():
    image = request.files.get('image')
    if not image:
        return jsonify(error='Select an image first.'), 400
    try:
        return jsonify(analyze(request.form.get('model'), image.read(), request.form.get('wells', '[]')))
    except (ValueError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 400
@app.errorhandler(413)
def too_large(error):
    return jsonify(error='Image exceeds the 32 MB upload limit.'), 413
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
