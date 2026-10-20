"""Local research UI; trained model adapters are optional."""

from flyscope.paths import FRONTEND, DATASET_FORMAT
import cv2

# Bound OpenCV workers; large host CPU counts can slow interactive Hough calls.
cv2.setNumThreads(2)
from flask import Flask, jsonify, render_template, request, send_file
from flyscope.imaging.detection import detect_wells
from flyscope.imaging.alignment import align_wells
from flyscope.models.registry import available_models, analyze

app = Flask(
    __name__,
    template_folder=str(FRONTEND / "templates"),
    static_folder=str(FRONTEND / "static"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html", study="ccrt")


@app.get("/ccrt")
def ccrt():
    return render_template("index.html", study="ccrt")


@app.get("/eggs")
def eggs():
    return render_template("index.html", study="eggs")


@app.get("/api/models")
def models():
    return jsonify(available_models())


@app.get("/api/dataset-format")
def dataset_format():
    return send_file(DATASET_FORMAT, mimetype="text/x-python")


@app.post("/api/detect")
def detect():
    image = request.files.get("image")
    if not image:
        return jsonify(error="Select an image first."), 400
    try:
        return jsonify(
            detect_wells(
                image.read(),
                request.form.get("method", "auto"),
                request.form.get("diameter"),
                request.form.get("region"),
                request.form.get("layout", "grid"),
                request.form.get("margin", 4),
            )
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/align")
def align():
    reference, current = request.files.get("reference"), request.files.get("image")
    if not reference or not current:
        return jsonify(error="Select reference and current images."), 400
    try:
        return jsonify(
            align_wells(
                reference.read(), current.read(), request.form.get("wells", "[]")
            )
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/analyze")
def predict():
    image = request.files.get("image")
    if not image:
        return jsonify(error="Select an image first."), 400
    try:
        return jsonify(
            analyze(
                request.form.get("model"),
                image.read(),
                request.form.get("wells", "[]"),
                request.form.get("sequence_session"),
                request.form.get("frame_index"),
            )
        )
    except (ValueError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 400


@app.errorhandler(413)
def too_large(error):
    return jsonify(error="Image exceeds the 32 MB upload limit."), 413
