# FlyScope — model-independent experimental review

This local Flask app supports coma-recovery sequence review and manual egg-count annotations before trained models are integrated. The legacy CNN is no longer imported at startup. No trained classifier or egg counter ships with this repository; unclassified wells are **Unknown**, with no fabricated confidence.

## Run

Python 3.12 is supported. From `CCRT/logiciel-v2_2`:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open the local server on port 5000 in your browser. In the prepared cloud machine use `/workspace/ecobio-venv/bin/python app.py` from the same directory. This is a local development application, not a multi-user hosted service.

## Review workflow

1. Choose **Ordered sequence** or **Single image / Non-sequential**, then browse a folder or choose image files. PNG, JPEG, BMP and WebP are supported. If every filename has a `YYYY-MM-DD_HH-MM-SS` timestamp, timestamps determine chronology; otherwise natural filename sorting puts frame2 before frame10. Modification-time sorting is also available. Inspect the frame list and confirm the frame interval. Changing input or mode clears the review after confirmation.
2. Select a full-grid or individual layout. Individual layout proposes the full image as a single well. Suggest contour boundaries or draw arbitrary polygons; double-click to finish and Escape to cancel. Well IDs are editable and unique. Record unknown, empty, single, multiple or obscured occupancy explicitly.
3. Select and drag wells, or drag vertices with **Edit boundary vertices**. Edit the current frame for camera movement, or edit the shared boundary across frames. **Align current frame to first** uses feature matching to propose translation/rotation/scale corrections with stable identities. Weak matches and out-of-image results are rejected. Review and confirm proposals; alignment is not biological tracking and cannot establish occupancy.
4. Select a well to see masked T−1 / T / T+1 crops. Navigate with buttons, arrow keys, frame slider or jump input; adjust playback FPS separately from the acquisition interval. Red means coma, green awake, gray unknown. Use current-frame manual state, egg-count and notes annotations; annotations do not overwrite model predictions.
5. Export JSON to preserve the project, frame/well results, normalized boundaries, per-frame overrides, model/version/confidence, manual corrections, history and summaries. CSV contains one row per frame/well, with structured geometry/provenance fields encoded as JSON. To resume, select the same image files in the same order, then restore JSON. Images are identified by path, size and modification time, not embedded or hashed. Export before leaving: browser memory is not permanent storage.

Summaries use effective states (including manual overrides). Unknown frames interrupt coma→awake transitions. Durations are classified-frame counts multiplied by the user-confirmed uniform frame interval; they are sampling estimates, not exact recovery times. Single-image mode has no durations or transitions. Model confidence averages exclude missing scores and do not invent confidence for manual labels.

## Integrating future models

Register an adapter in `backend/services/model_registry.py:ADAPTERS` with a stable ID. It exposes:

- `metadata`: `name`, `version`, `description`, and `task` (e.g. `coma` or `eggs`). Use an immutable version or artifact hash for reproducibility.
- `predict(image_bytes, wells)`: one record per well, with `well_id`, `prediction` (`coma`, `awake`, `unknown`), `confidence` (0–1 or `None`) and optional nonnegative integer `egg_count`. An egg-only adapter can return `unknown` for coma state.

The UI discovers adapters via `/api/models`. `/api/analyze` sends one image and that frame's normalized polygon geometry at a time. An adapter must decode, polygon-mask/crop, resize and normalize according to its trained model, and load artifacts lazily. Keep model weights and preprocessing versions explicit; do not infer a task from a filename. The current contract is per-frame; a future temporal model needs an explicit sequence-aware adapter rather than silently using adjacent images. Completed frames survive cancellation/errors; manual annotations are retained independently. Geometry edits invalidate affected model predictions and preserve their previous values in project history.

Contour detection is a configurable starting heuristic, validated on synthetic rectangular and elliptical wells. It has not been calibrated against actual experimental images. Partial occlusion, lighting changes, highly overlapping wells and multiple flies require human review. Camera alignment uses distributed ORB matches and robust partial affine estimation; perspective changes or weak image texture may require manual boundaries. IDs remain stable across frame-specific geometry edits; re-detecting replaces wells only after explicit confirmation.

Images remain in the browser except when sent to local detection/alignment/model endpoints. These endpoints do not save image uploads. Each request is limited to 32 MB. In single-image mode, choosing multiple files loads only the first ordered file with a notice.

## Validation

```sh
python -m unittest discover -s tests -v
node --test tests/review-core.test.js
```

Browser smoke test (with the Flask server running, Node.js, Playwright installed, and Chromium available):

```sh
npm install --prefix /tmp/flyscope-tests --cache /tmp/flyscope-npm playwright
NODE_PATH=/tmp/flyscope-tests/node_modules PYTHON=.venv/bin/python \
  CHROMIUM_PATH=/usr/bin/chromium node tests/browser-smoke.cjs
```

Set `APP_URL` to test another local port. The browser test creates temporary synthetic images and checks ordering, detection, annotations, export/restore, geometry editing, playback, single-image mode and mobile layout. The adapter test uses a test-only stub to verify the contract; it is not a trained-model validation. Legacy scripts and CNN services remain on disk for reference and are not used by the new UI.
