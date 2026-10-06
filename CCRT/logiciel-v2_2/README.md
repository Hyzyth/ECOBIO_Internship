# FlyScope — model-independent experimental review

This local Flask app has separate **CCRT** (`/ccrt`) and **Egg counting** (`/eggs`) pages. It supports manual dataset preparation and experimental review before trained models are integrated. The legacy CNN is no longer imported at startup. No trained classifier or egg counter ships with this repository; unclassified wells are **Unknown**, with no fabricated confidence.

## Run

Python 3.12 is supported. From `CCRT/logiciel-v2_2`:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open the local server on port 5000 in your browser. In the prepared cloud machine use `/workspace/ecobio-venv/bin/python app.py` from the same directory. This is a local development application, not a multi-user hosted service.

## Review workflow

1. Choose the study using the header navigation. On CCRT, choose **Ordered sequence** or **Single image / Non-sequential**, then browse a folder or choose image files. PNG, JPEG, BMP and WebP are supported. If every filename has a `YYYY-MM-DD_HH-MM-SS` timestamp, timestamps determine chronology; otherwise natural filename sorting puts frame2 before frame10. Modification-time sorting is also available. Inspect the frame list and confirm the frame interval. Changing input or mode clears the review after confirmation. Egg counting accepts a single image or an independent image collection: counts and regions belong only to that image, with no temporal assumptions. Single-image mode hides ordering, frame interval, playback, temporal previews and alignment.
2. Select a full-grid or individual layout. Individual layout automatically detects one dominant rim on image loading. It uses separate close-crop rules, with no requirement for repeated neighbours. If detection fails, draw a boundary, use contour mode for a non-circular well, or explicitly select **Use full image as one well**. Use automatic repeated-ring detection for circular grids, or select non-circular contour mode. Enter an approximate diameter in original-image pixels if automatic size estimation is wrong. Alternatively, draw arbitrary polygons; double-click to finish and Escape to cancel. Well IDs are editable and unique. The selector and **Remove selected well** button directly above the image remove any proposal and its annotations across all frames. You can also select the **Remove well** canvas tool, or press Delete/Backspace outside a text input. Other identities are not renumbered. Record unknown, empty, single, multiple or obscured occupancy explicitly.
3. Select and drag wells, or drag vertices with **Edit boundary vertices**. Edit the current frame for camera movement, or edit the shared boundary across frames. **Align current frame to first** or **Align entire sequence to first** uses feature matching to propose camera-motion corrections with stable identities. Similarity, affine and perspective transforms are compared; a more complex transform is selected only for a measurable gain. Every frame is aligned directly to the reference, avoiding accumulated drift. Manual boundaries are shared as a starting template. When positions remain stable, **Confirm boundary** defaults to approving that well across the whole sequence; **Approve stable boundaries for entire sequence** approves all wells in one operation. The approval scope can be changed to **Current frame only** for local inspection. Pending failed alignments are excluded from bulk approval; later geometry edits flag the affected boundaries for review again. The selected-well progress indicator, **Next unreviewed boundary** and **Next unlabeled frame** buttons help complete the whole sequence. Use **Confirm this frame’s boundaries** when only a particular frame needs review. Failures are reported per frame, preserve existing geometry and mark it for review. Convex boundaries crossing an image edge are clipped and marked partially visible; entirely out-of-view or ambiguous clipped concave boundaries cause explicit review failures. Weak matches are rejected. Review and confirm proposals; alignment is not biological tracking and cannot establish occupancy.
4. Select a well to see masked T−1 / T / T+1 crops. Navigate with buttons, arrow keys, frame slider or jump input; adjust playback FPS separately from the acquisition interval. Red/C means coma, green/A awake, gray/? unknown, cyan/E empty and purple/! multiple individuals. Use the quick **Coma**, **Awake** and **Uncertain** buttons or current-frame state and notes fields. **Coma through this frame** labels frames 1…T inclusive; **Awake from this frame onward** labels T…last inclusive. Custom ranges are also inclusive. Existing manual labels require confirmation before replacement, and the last quick/range operation can be undone without restoring old model predictions or erasing newer edits. Ambiguous transition frames can remain uncertain. Egg count controls appear only on the egg page.
5. Choose **Report ZIP** for readable results, summaries and numbered location maps; **readable results CSV** omits polygon arrays. Choose **Training dataset ZIP** for original images, masked crops, manual labels and frame-specific boundaries. **Project backup JSON** preserves normalized boundaries, per-frame overrides, model/version/confidence, manual corrections, history and summaries. To resume, select the same image files in the same order, then restore JSON. Images are identified by path, size and modification time, not embedded or hashed. Export before leaving: browser memory is not permanent storage.

Summaries use effective states (including manual overrides). Unknown frames interrupt coma→awake transitions. Human durations use sample-and-hold over the first-to-last observed image: the state at each interval start determines that interval, with no invented duration after the last image. CCRT is the first awake observation immediately following coma, relative to cold-exposure end; the frame interval limits precision. Frame 1 defaults to time zero. An optional delay adds the elapsed seconds from cold-exposure end to frame 1. Unknown transition gaps, later coma labels, already-awake wells, missing recovery and unverified timing yield explicit statuses instead of exact CCRT. Single occupancy must be confirmed before CCRT measurements are emitted. The legacy project-backup `summary` keeps classified-frame counts for compatibility; use human reports or machine `recovery` metrics for observed durations. Single-image mode has no durations or transitions. Model confidence averages exclude missing scores and do not invent confidence for manual labels.

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
node --test tests/*.test.js
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


## Reports and training datasets

**Human report ZIP** contains `report.html` to open directly, compact `results.csv` (one row per CCRT well), two-column `summary.csv`, `report.json` and an editable `project.json` backup. CCRT rows show occupancy, observed coma/awake/uncertain duration, sampled CCRT and status. Empty wells, multiple individuals and unverified occupancy cannot produce CCRT measurements. Use **Confirm unchecked wells contain one individual** after checking them to avoid approving each well separately. Well boundaries stay out of the human result table; `well-locations.csv` and numbered maps identify the wells on the reference frame. Egg reports contain independent image counts; custom-area maps are optional.

**Machine analysis ZIP** retains frame-level labels, normalized boundaries, stable IDs, prediction confidence/model version, separate manual corrections, recovery bounds, timing metadata and history in `machine.json`, `frames.csv`, `recovery.csv` and `project.json`. Those extras let scripts reconstruct timelines, associate labels with image pixels and audit reproducibility; they are unnecessary columns in a human summary. Machine exports contain no image pixels; use the training export when images are needed. All ZIP export types restore full editable projects through **Import report**.

The egg page defaults to **Whole image (no layout assumed)**, hides well geometry/occupancy tools, and accepts independent images. **Optional custom areas** exposes region editing if future data need it. Training annotations explicitly identify `annotation_unit: image|region` and `annotation_type: count`; whole-image counts are provisional labels, not a fixed well layout or an egg localization dataset. Adapt the annotation format/model adapter when the real experiment is known.

**Training dataset ZIP** includes:

- Original annotated frames in `images/`, with original filenames, dimensions and SHA-256 hashes in `annotations.json`.
- Per-frame/per-well crops at original pixel resolution in `crops/`. PNG alpha masks the area outside the manually defined or reviewed boundary. No UI screenshots or overlay labels enter the training images.
- `annotations.json` and `annotations.csv`, containing manual targets, stable well IDs, frame numbers/timestamps, experiment group, normalized/pixel boundaries, pixel crop boxes, label-range provenance, review status and partial visibility.
- A complete `project.json` backup, a framework-neutral `dataset.py` loader, and instructions.

Only manual annotations become training targets. Unannotated samples are omitted; model predictions are never promoted to labels. Explicit uncertain states, partial rims, unreviewed boundaries and CCRT empty/multiple/obscured/unconfirmed occupancy regions are retained but flagged `eligible_for_training: false`. Review boundaries individually or approve stable positions for the whole sequence before using its labels. The default loader selects eligible samples only, maps coma to 0 and awake to 1, and uses integer egg counts (including zero). Unknown is uncertainty, never an additional coma/awake training class. Occupancy defaults to unknown; only explicitly confirmed `single` occupancy is eligible for CCRT training. Exclusion reasons are recorded, and previously saved labels are preserved.

Unzip the dataset, install Pillow, then run `python dataset.py /path/to/dataset` or import `samples(root)` into a training pipeline. Each sample yields an RGB PIL image, target and annotation metadata. `include_review=True` allows reviewed exceptions to be inspected, while uncertain CCRT labels still have no binary target. Splits are not generated automatically: keep all related images/experiments in the same split using the editable **Experiment / split group** field. Randomly splitting neighbouring frames or wells from the same run risks leakage.

Exports are built locally in the browser and are cancellable. ZIP files use standard uncompressed entries to retain original images; split very large collections before reaching the 4 GB / 65,535 entry ZIP limits. No training images are uploaded or saved by the export process.

With the Flask server, Playwright and Chromium available, validate both pages and actual exported archives:

```sh
NODE_PATH=/tmp/flyscope-tests/node_modules PYTHON=.venv/bin/python \
  CHROMIUM_PATH=/usr/bin/chromium node tests/studies-browser.cjs
```

This exercises manual polygon drawing/editing across a sequence, range endpoints, uncertainty/undo, image-specific egg regions, contextual controls and project restoration. Python opens the downloaded ZIPs, verifies hashes and crop/mask pixels, and runs the bundled loader for both tasks.

## Continue an existing review

Select the original images in their saved order, then use **Import report / training ZIP or JSON**. New report ZIPs and training ZIPs include `project.json`, so importing restores well geometry, per-frame overrides, labels, occupancy, model provenance and history. Existing standalone project JSON files remain supported by either import control.

Older readable reports (`report.json`, directly or inside a stored/deflated ZIP) omitted polygon boundaries and full history. Detect/draw the matching wells first, keeping their saved IDs, then import the report to recover the frame labels or egg counts onto the current geometry. Image order and well locations are checked; a mismatch stops the import without changing annotations. Missing historical geometry cannot be reconstructed from a numbered screenshot. There is no need to manually re-enter the saved labels. Imports read only the selected JSON metadata from archives, validate its CRC, and support metadata up to 32 MB.

For CCRT, **Empty** occupancy applies to the entire well: its outline turns cyan with a dotted pattern and an E marker, the state selector displays **Empty — no state applicable**, and summaries/exports report empty frames rather than coma/awake/uncertain. Previous state annotations remain stored and become visible again if occupancy changes back. Empty wells remain excluded from normal coma/awake training samples.
