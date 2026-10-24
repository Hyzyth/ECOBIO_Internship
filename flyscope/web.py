"""Local research UI; trained model adapters are optional."""

from flyscope.paths import FRONTEND, DATASET_FORMAT
import cv2

# Bound OpenCV workers; large host CPU counts can slow interactive Hough calls.
cv2.setNumThreads(2)
from flask import Flask, jsonify, render_template, request, send_file
from flyscope.imaging.detection import detect_wells
from flyscope.imaging.alignment import align_wells
from flyscope.models.registry import available_models, analyze
import base64
import json
import numpy as np

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


@app.get("/pose")
def pose_annotation():
    return render_template("pose.html")


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


@app.post("/api/input-preview")
def input_preview():
    from flyscope.imaging.geometry import parse_wells
    from flyscope.models.runtime import crop_rgb, pose_input
    from flyscope.imaging.foreground import MASKED, ISOLATED, isolate
    from flyscope.models.keras import crop_well, preprocess
    from flyscope.models.registry import ADAPTERS

    try:
        upload = request.files.get("image")
        if not upload:
            raise ValueError("Select an image first.")
        image = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unreadable image.")
        wells = parse_wells(request.form.get("wells", "[]"))
        if len(wells) != 1:
            raise ValueError("Select exactly one well.")
        mode = request.form.get("mode", "model")
        if mode not in ("model", "fly"):
            raise ValueError("Unsupported preview mode.")
        adapter = ADAPTERS.get(request.form.get("model"))
        meta = adapter.metadata if adapter else {}
        size = (
            int(meta.get("image_size", 128))
            if mode == "model"
            else int(request.form.get("size", 128))
        )
        if not 32 <= size <= 1024:
            raise ValueError("Preview size must be between 32 and 1024.")
        points = wells[0]["points"]
        preprocessing = ISOLATED if mode == "fly" else meta.get("preprocessing", MASKED)
        crop, mask = crop_well(image, points)
        status = "well_mask"
        if preprocessing.startswith("legacy-"):
            rgb = np.repeat((preprocess(crop, mask) * 255).astype(np.uint8), 3, axis=2)
            size, status = 200, "legacy_contrast_otsu"
        else:
            rgb, _ = crop_rgb(image, points, size, preprocessing)
            if preprocessing == ISOLATED:
                _, status = isolate(crop, mask)
            if mode == "model" and meta.get("architecture") == "sleap-rf":
                pose_meta = json.loads(
                    (adapter.path.parent / "pose/pose.json").read_text()
                )
                size = pose_meta["configuration"]["size"]
                rgb = cv2.resize(pose_input(rgb), (size, size))
                status += "_pose_centered"

        def png(rgb):
            return (
                "data:image/png;base64,"
                + base64.b64encode(
                    cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1]
                ).decode()
            )

        before, _ = crop_rgb(image, points, size)
        return jsonify(
            image=png(rgb),
            original=png(before),
            size=size,
            preprocessing=preprocessing,
            foreground_status=status,
        )
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400


@app.errorhandler(413)
def too_large(error):
    return jsonify(error="Image exceeds the 32 MB upload limit."), 413
