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
2. Select a full-grid or individual layout. Individual layout automatically detects one dominant rim on image loading. It uses separate close-crop rules, with no requirement for repeated neighbours. If detection fails, draw a boundary, use contour mode for a non-circular well, or explicitly select **Use full image as one well**. Use automatic repeated-ring detection for circular grids, or select non-circular contour mode. Enter an approximate diameter in original-image pixels if automatic size estimation is wrong. Alternatively, draw arbitrary polygons; double-click to finish and Escape to cancel. Well IDs are editable and unique. The selector and **Remove selected well** button directly above the image remove any proposal and its annotations across all frames. You can also select the **Remove well** canvas tool, or press Delete/Backspace outside a text input. Other identities are not renumbered. Record unknown, empty, single, multiple or obscured occupancy explicitly.
3. Select and drag wells, or drag vertices with **Edit boundary vertices**. Edit the current frame for camera movement, or edit the shared boundary across frames. **Align current frame to first** or **Align entire sequence to first** uses feature matching to propose camera-motion corrections with stable identities. Similarity, affine and perspective transforms are compared; a more complex transform is selected only for a measurable gain. Every frame is aligned directly to the reference, avoiding accumulated drift. Inspect frames and use **Confirm this frame’s boundaries** to review all visible wells together. Failures are reported per frame, preserve existing geometry and mark it for review. Convex boundaries crossing an image edge are clipped and marked partially visible; entirely out-of-view or ambiguous clipped concave boundaries cause explicit review failures. Weak matches are rejected. Review and confirm proposals; alignment is not biological tracking and cannot establish occupancy.
4. Select a well to see masked T−1 / T / T+1 crops. Navigate with buttons, arrow keys, frame slider or jump input; adjust playback FPS separately from the acquisition interval. Red means coma, green awake, gray unknown. Use current-frame manual state, egg-count and notes annotations; annotations do not overwrite model predictions.
5. Export JSON to preserve the project, frame/well results, normalized boundaries, per-frame overrides, model/version/confidence, manual corrections, history and summaries. CSV contains one row per frame/well, with structured geometry/provenance fields encoded as JSON. To resume, select the same image files in the same order, then restore JSON. Images are identified by path, size and modification time, not embedded or hashed. Export before leaving: browser memory is not permanent storage.

Summaries use effective states (including manual overrides). Unknown frames interrupt coma→awake transitions. Durations are classified-frame counts multiplied by the user-confirmed uniform frame interval; they are sampling estimates, not exact recovery times. Single-image mode has no durations or transitions. Model confidence averages exclude missing scores and do not invent confidence for manual labels.

## Integrating future models

Register an adapter in `backend/services/model_registry.py:ADAPTERS` with a stable ID. It exposes:

- `metadata`: `name`, `version`, `description`, and `task` (e.g. `coma` or `eggs`). Use an immutable version or artifact hash for reproducibility.
- `predict(image_bytes, wells)`: one record per well, with `well_id`, `prediction` (`coma`, `awake`, `unknown`), `confidence` (0–1 or `None`) and optional nonnegative integer `egg_count`. An egg-only adapter can return `unknown` for coma state.

The UI discovers adapters via `/api/models`. `/api/analyze` sends one image and that frame's normalized polygon geometry at a time. An adapter must decode, polygon-mask/crop, resize and normalize according to its trained model, and load artifacts lazily. Keep model weights and preprocessing versions explicit; do not infer a task from a filename. The current contract is per-frame; a future temporal model needs an explicit sequence-aware adapter rather than silently using adjacent images. Completed frames survive cancellation/errors; manual annotations are retained independently. Geometry edits invalidate affected model predictions and preserve their previous values in project history.

Circular-grid detection combines gradient-direction-consistent Hough ring candidates with dominant-size and direction-specific repeated-spacing filters to suppress mounting fixtures. It does not assume a fixed grid count or invent missing wells. Automatic size estimation uses enclosed contour candidates; a diameter override is available when rims are weak or incomplete. Non-circular contour mode preserves simple contours including concave shapes; retraced contours are repaired with explicitly labelled convex hulls. Use the **Set detection region** canvas tool to draw a rectangle around the experimental area when hardware is geometrically indistinguishable from wells. The region is stored in JSON projects, and full boundary proposals outside it are excluded. Detected polygons cannot self-intersect, and drawn/edited/imported polygons with crossings, retraced edges or zero area are rejected. Detection has been validated on synthetic rectangular/elliptical wells and a staggered 72-well scene with broken rims, fly-like objects, lighting/noise variation and three concentric-ring fixtures. It has also been checked against the supplied five 4056×3040 sequence frames: 72 complete wells and six clipped bottom wells were proposed in every frame, with no proposals in the five manually identified fixture regions. Additional partial-ring candidates are considered only at image edges, then checked against visible rim support, dominant size and repeated neighbouring directions. Partial polygons are clipped to the image rectangle, labelled in the detailed view and exported as `boundary_partial`; unseen parts cannot be verified. Automatic single-well rim detection was run on all 14 provided coma/awake crops, with one rim proposal in every crop and visual review of the overlays; these labels are not used as predictions or to train a model. These samples cover one support, not every experimental setup. Partial occlusion, lighting changes, highly overlapping wells and multiple flies require human review. Camera alignment uses distributed ORB matches and robust similarity/affine/projective estimation; weak texture, implausible transforms, entirely out-of-view boundaries or ambiguous clipped concave polygons cause explicit review failures. Clipped reference shapes remain partial after alignment because missing rim geometry cannot be recovered from that template. IDs remain stable across frame-specific geometry edits; re-detecting replaces wells only after explicit confirmation.

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

## Checking private sample images

Original research images are kept outside the repository. Run the optional sample validator against an extracted folder containing `sequence`, `coma` and `awake`:

```sh
python tools/validate_samples.py --samples /path/Extract --output /tmp/flyscope-report
```

A `--manifest` JSON may map filenames to manually reviewed `complete_well_count`, `partial_well_count` and `fixture_exclusion_zones` (`center` and `radius` in normalized coordinates). Counts are validation expectations, never application assumptions. Outputs include boundary overlays and per-frame alignment metrics. Agreement with independent detections is measured at a 1200-pixel working width and is not ground-truth biological accuracy.

With the server running, test the full uploaded five-frame browser workflow:

```sh
NODE_PATH=/tmp/flyscope-tests/node_modules SAMPLE_DIR=/path/Extract \
  CHROMIUM_PATH=/usr/bin/chromium node tests/sample-browser.cjs
```

This checks detection-region selection, full-sequence alignment, stable identities, frame review, export/restore and automatic rim detection plus manual annotations on every supplied coma/awake crop. It expects the supplied extract, while the unit tests cover rotated/perspective transforms, non-circular and concave shapes, unequal directional spacing, missing wells and resolution changes.
