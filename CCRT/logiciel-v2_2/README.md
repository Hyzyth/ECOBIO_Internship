# FlyScope — model-independent experimental review

This local Flask app has separate **CCRT** (`/ccrt`) and **Egg counting** (`/eggs`) pages. It supports manual dataset preparation, reproducible model comparisons and experimental review. The two original Keras coma/awake models are selectable on the CCRT page and load lazily. An egg counter is not supplied; egg annotations remain independent-image counts. Manual mode stays available. The ResNet-18, SLEAP/random-forest and DeepEthogram training/inference suite is implemented; see **[WORKFLOW.md](WORKFLOW.md)** for the structured setup-to-evaluation guide, using one Python 3.12 environment and one requirements file.

## Run

Use Python 3.12 (tested on 3.12.12) from `CCRT/logiciel-v2_2`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_runtime.py --prepare-models
python app.py
```

The same environment runs every model and trainer. The Linux Torch wheel includes CUDA 12.8 and Blackwell support; CPU execution is also available. See [WORKFLOW.md](WORKFLOW.md) for OS libraries, the RTX PRO 3000 GPU check, exports, training and evaluation. TensorFlow and separate suite/pose interpreter paths are no longer required. Keep Keras's backend unset or `torch`.

Open the local server on port 5000 in your browser. In the prepared cloud machine use `/workspace/ecobio-unified-venv/bin/python`. The tested platform is Linux x86_64. No physical NVIDIA GPU is attached to the cloud validation machine.

## Review workflow

The sidebar guides you through **Images → Wells → Review → Export**. Load images before choosing well boundaries; Review and Export unlock once wells exist. Continue/Back buttons and the step navigation let you revisit settings without clearing geometry, predictions or annotations. Replacing images or changing analysis mode asks before clearing an existing review. Import a saved review from **Images → Resume a saved review** after selecting the original images; restoration opens Review with its saved labels.

Only the current step's controls are shown. Sequence timing appears after loading more than one frame; single-image inputs have no timing, playback or alignment controls. Configure nondefault file ordering under **Loading options** before loading. Automatic-detection adjustments, camera alignment, model predictions, detailed annotations and custom label ranges are expandable options. Quick Coma/Awake/Uncertain controls remain directly available during review. Boundary editing and deletion happen in Wells; clicking or dragging during Review selects wells without moving their boundaries.

Egg counting uses **Images → Count → Export** by default, with no well layout or model selector. Choosing **Custom areas** adds an Areas step. Export presents the human report and editable backup first, with summaries alongside; dataset/comparison settings and machine formats are expandable. Partial annotation remains supported, with no requirement to label the entire grid before exporting.

1. Choose the study using the header navigation. On CCRT, choose **Ordered sequence** or **Single image / Non-sequential**, then browse a folder or choose image files. PNG, JPEG, BMP and WebP are supported. If every filename has a `YYYY-MM-DD_HH-MM-SS` timestamp, timestamps determine chronology; otherwise natural filename sorting puts frame2 before frame10. Modification-time sorting is also available. Inspect the frame list and confirm the frame interval. Changing input or mode clears the review after confirmation. Egg counting accepts a single image or an independent image collection: counts and regions belong only to that image, with no temporal assumptions. Single-image mode hides ordering, frame interval, playback, temporal previews and alignment.
2. Select a full-grid or individual layout. Individual layout permits at most one well. Choose **Suggest well boundaries** to fit one dominant rim after selecting this layout; subsequent image loads also use the retained individual layout. Replacing a drawn/full-image boundary preserves that well identity and manual labels in the selected edit scope. Switching a multi-well review to individual layout requires removing extra wells first. Automatic single-well contours must cover at least 12% of the crop and span at least 45% of its shorter dimension; near-duplicate vertices closer than 1.5% of the boundary size are removed without a large area change, and angles below 25 degrees are rejected. These quality checks apply to automatic individual proposals; manually drawn simple concave shapes remain supported. It uses separate close-crop rules, with no requirement for repeated neighbours. If detection fails, draw a boundary, use contour mode for a non-circular well, or explicitly select **Use full image as one well**. Use automatic repeated-ring detection for circular grids, or select non-circular contour mode. Enter an approximate diameter in original-image pixels if automatic size estimation is wrong. Alternatively, draw arbitrary polygons; double-click to finish and Escape to cancel. Well IDs are editable and unique. The selector and **Remove selected well** button directly above the image remove any proposal and its annotations across all frames. You can also select the **Remove well** canvas tool, or press Delete/Backspace outside a text input. Other identities are not renumbered. In Review, record unknown, empty, single, multiple or obscured occupancy explicitly.
3. Select and drag wells, or drag vertices with **Edit boundary vertices**. Edit the current frame for camera movement, or edit the shared boundary across frames. **Align current frame to analysis start** or **Align analysis window to start frame** uses feature matching to propose camera-motion corrections with stable identities. Similarity, affine and perspective transforms are compared; a more complex transform is selected only for a measurable gain. Every frame is aligned directly to the reference, avoiding accumulated drift. Manual boundaries are shared as a starting template. When positions remain stable, **Confirm boundary** defaults to approving that well across the whole sequence; **Approve stable boundaries for entire sequence** approves all wells in one operation. The approval scope can be changed to **Current frame only** for local inspection. Pending failed alignments are excluded from bulk approval; later geometry edits flag the affected boundaries for review again. The selected-well progress indicator, **Next unreviewed boundary** and **Next unlabeled frame** buttons help complete the whole sequence. Use **Confirm this frame’s boundaries** when only a particular frame needs review. Failures are reported per frame, preserve existing geometry and mark it for review. Convex boundaries crossing an image edge are clipped and marked partially visible; entirely out-of-view or ambiguous clipped concave boundaries cause explicit review failures. Weak matches are rejected. Review and confirm proposals; alignment is not biological tracking and cannot establish occupancy.
4. Select a well to see masked T−1 / T / T+1 crops. Navigate with buttons, arrow keys, frame slider or jump input; adjust playback FPS separately from the acquisition interval. Red/C means coma, green/A awake, gray/? unknown, cyan/E empty and purple/! multiple individuals and amber/U low-confidence model output. Use the quick **Coma**, **Awake** and **Uncertain** buttons, or open **Notes and detailed annotation** for current-frame state and notes fields. **Coma through this frame** labels the analysis start…T inclusive; **Awake from this frame onward** labels T…last inclusive. Custom ranges are also inclusive. Existing manual labels require confirmation before replacement, and the last quick/range operation can be undone without restoring old model predictions or erasing newer edits. Ambiguous transition frames can remain uncertain. Egg count controls appear only on the egg page.
5. Choose **Report ZIP** for readable results, summaries and numbered location maps; **readable results CSV** omits polygon arrays. Choose **Training dataset ZIP** for original images, manual labels and shared boundaries with frame-specific corrections. **Project backup JSON** preserves normalized boundaries, per-frame overrides, model/version/confidence, manual corrections, history. To resume, select the same image files in the same order, then restore JSON. Images are identified by path, size and modification time, not embedded or hashed. Export before leaving: browser memory is not permanent storage.

Summaries use effective states, including manual overrides. Select the first analysis frame to exclude raw setup images without resetting original frame numbers or time offsets. Already-trimmed input clears the assumed timing reference; enter a known delay if available. Human reports prioritize the first Awake frame, filename and frame ID; CCRT seconds require an observed Coma→Awake transition and confirmed timing. Unknown gaps, later coma labels, already-awake wells, missing recovery and unverified timing yield explicit statuses. Single occupancy must be confirmed before CCRT measurements are emitted. Machine exports retain observed-window durations and timing bounds, with no invented duration after the final observation. Single-image mode has no temporal metrics. See [WORKFLOW.md](WORKFLOW.md) for confidence/streak decisions, species metadata and dataset timing semantics.

## Integrating future models

Register an adapter in `backend/services/model_registry.py:ADAPTERS` with a stable ID. It exposes:

- `metadata`: `name`, `version`, `description`, and `task` (e.g. `coma` or `eggs`). Use an immutable version or artifact hash for reproducibility.
- `predict(image_bytes, wells)`: one record per well, with `well_id`, `prediction` (`coma`, `awake`, `unknown`), `confidence` (0–1 or `None`) and optional nonnegative integer `egg_count`. An egg-only adapter can return `unknown` for coma state.

The UI discovers adapters via `/api/models`. `/api/analyze` sends one image and that frame's normalized polygon geometry at a time. An adapter must decode, polygon-mask/crop, resize and normalize according to its trained model, and load artifacts lazily. Keep model weights and preprocessing versions explicit; do not infer a task from a filename. Suite adapters additionally implement `predict_sequence(image_bytes, wells, session, frame)`. DeepEthogram requires an explicit ordered session and frame index; the browser sends analysis frames sequentially with a fresh session for each run. Completed frames survive cancellation/errors; manual annotations are retained independently. Geometry edits invalidate affected model predictions and preserve their previous values in project history.

Circular-grid detection refines candidate circles against the observed outer rim using robust elliptical fits and a configurable margin (default 4%), capped by neighbour spacing. Existing boundaries can be resized across the sequence while preserving IDs, manual labels and frame-specific geometry. Review resized boundaries again. Circular-grid detection combines gradient-direction-consistent Hough ring candidates with dominant-size and direction-specific repeated-spacing filters to suppress mounting fixtures. It does not assume a fixed grid count or invent missing wells. Automatic size estimation uses enclosed contour candidates; a diameter override is available when rims are weak or incomplete. Non-circular contour mode preserves simple contours including concave shapes; retraced contours are repaired with explicitly labelled convex hulls. Use the **Set detection region** canvas tool to draw a rectangle around the experimental area when hardware is geometrically indistinguishable from wells. The region is stored in JSON projects, and full boundary proposals outside it are excluded. Detected polygons cannot self-intersect, and drawn/edited/imported polygons with crossings, retraced edges or zero area are rejected. Detection has been validated on synthetic rectangular/elliptical wells and a staggered 72-well scene with broken rims, fly-like objects, lighting/noise variation and three concentric-ring fixtures. It has also been checked against the supplied five 4056×3040 sequence frames: 72 complete wells and six clipped bottom wells were proposed in every frame, with no proposals in the five manually identified fixture regions. Additional weak interior and partial edge ring candidates are checked against visible rim support, dominant size and repeated neighbouring directions. Partial polygons are clipped to the image rectangle, labelled in the detailed view and exported as `boundary_partial`; unseen parts cannot be verified. Automatic single-well rim detection was run on all 14 provided coma/awake crops, with one rim proposal in every crop and visual review of the overlays; these labels are not used as predictions or to train a model. These samples cover one support, not every experimental setup. Partial occlusion, lighting changes, highly overlapping wells and multiple flies require human review. Camera alignment uses distributed ORB matches and robust similarity/affine/projective estimation; weak texture, implausible transforms, entirely out-of-view boundaries or ambiguous clipped concave polygons cause explicit review failures. Clipped reference shapes remain partial after alignment because missing rim geometry cannot be recovered from that template. IDs remain stable across frame-specific geometry edits; re-detecting replaces wells only after explicit confirmation.

Images remain in the browser except when sent to local detection/alignment/model endpoints. These endpoints do not save image uploads. Each request is limited to 32 MB. In single-image mode, choosing multiple files loads only the first ordered file with a notice.

## Validation

```sh
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
node --test tests/*.test.js
```

Browser smoke test (with the Flask server running, Node.js, Playwright installed, and Chromium available):

```sh
npm install --prefix /tmp/flyscope-tests --cache /tmp/flyscope-npm playwright
NODE_PATH=/tmp/flyscope-tests/node_modules PYTHON=.venv/bin/python \
  CHROMIUM_PATH=/usr/bin/chromium node tests/workflow-browser.cjs
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

**Human report ZIP** contains `report.html` to open directly, compact `results.csv` (one row per CCRT well), two-column `summary.csv`, `report.json` and an editable `project.json` backup. CCRT rows show occupancy, the first Awake frame/image/ID, optional CCRT seconds and status. Observed durations remain in machine exports. Empty wells, multiple individuals and unverified occupancy cannot produce CCRT measurements. Use **Confirm unchecked wells contain one individual** after checking them to avoid approving each well separately. Well boundaries stay out of the human result table; `well-locations.csv` and numbered maps identify the wells on the reference frame. Egg reports contain independent image counts; custom-area maps are optional.

**Machine analysis ZIP** retains frame-level labels, normalized boundaries, stable IDs, prediction confidence/model version, separate manual corrections, recovery bounds, timing metadata and history in `machine.json`, `frames.csv`, `recovery.csv` and `project.json`. Those extras let scripts reconstruct timelines, associate labels with image pixels and audit reproducibility; they are unnecessary columns in a human summary. Machine exports contain no image pixels; use the training export when images are needed. Human, machine and training ZIPs restore editable projects through **Import report**. Evaluation-only test ZIPs deliberately omit project backups.

The egg page defaults to **Whole image (no layout assumed)**, hides well geometry/occupancy tools, and accepts independent images. **Optional custom areas** exposes region editing if future data need it. Training annotations explicitly identify `annotation_unit: image|region` and `annotation_type: count`; whole-image counts are provisional labels, not a fixed well layout or an egg localization dataset. Adapt the annotation format/model adapter when the real experiment is known.

**Training datasets** can be saved directly to a new folder in supported Chromium browsers, or downloaded as ZIPs. Large ZIP exports are split into ordinary parts around 256 MiB each; download every linked part and extract them into the same empty folder. Completion messages and download links remain visible. No folder is created by a ZIP download. See [WORKFLOW.md](WORKFLOW.md#large-experiments-and-exported-files) for the commands and complete procedure.

- `images/`: one original per analysis frame, including unlabeled temporal context. No UI overlays or duplicate per-well crops.
- `annotations.json` (schema 3): manual targets, stable well IDs, original image dimensions/identifiers/hashes, review/partial flags, shared normalized boundaries and frame overrides. Target rows reference wells and frames; the model loaders resolve their pixels and geometry.
- `comparison-plan.json`: immutable shared well/annotation selections, seeds and a hash binding the plan to this dataset.
- `project.json`: compact editable backup preserving annotations, geometry, prediction provenance and history. Derived frame results are omitted. Old report/project imports remain supported.
- `dataset.py` and `dataset_format.py`: standalone Pillow loader producing masked RGB crops lazily. All model trainers also derive crops from originals, and continue to accept old exports with pre-generated crops.
- `dataset-complete.json`: written last. Missing completion markers on new exports cause the trainers to reject incomplete datasets.

Only manual targets supervise training; unlabeled wells/frames never become negatives. Explicit Empty supplies occupancy labels. Coma/Awake training requires confirmed single occupancy and reviewed complete boundaries. Uncertain states, partial/unreviewed rims and multiple/obscured/unconfirmed occupancy targets retain exclusion flags. Test exports use the same compact image/annotation format, omitting project history, training plans and loaders. Keep test experiments separate from training.

Exports stay in the browser or your chosen folder. Folder writes are sequential; no ZIP size limit applies. ZIP parts have bounded size and stream their checksums. Original images are not uploaded during export. Cancellation or failure preserves the app annotations; incomplete folder exports should be discarded and re-exported. One unusually large metadata file may require folder saving. To resume editing a large export, import its `project.json` after selecting the original images.

The sequence-summary timeline spans the entire analysis window using proportional state runs, with no minimum width per frame; hover a segment for its frame range/state. Empty and unsure remain distinct colors. This also avoids thousands of per-frame DOM elements for a simple coma-to-awake transition.

With the Flask server, Playwright and Chromium available, validate both pages and actual exported archives:

```sh
NODE_PATH=/tmp/flyscope-tests/node_modules PYTHON=.venv/bin/python \
  CHROMIUM_PATH=/usr/bin/chromium node tests/studies-browser.cjs
```

This exercises manual polygon drawing/editing across a sequence, range endpoints, uncertainty/undo, image-specific egg regions, contextual controls and project restoration. Python opens the downloaded ZIPs, verifies hashes and crop/mask pixels, and runs the bundled loader for both tasks.

## Continue an existing review

Select the original images in their saved order, then use **Import report / training ZIP or JSON**. New report ZIPs and training ZIPs include `project.json`, so importing restores well geometry, per-frame overrides, labels, occupancy, model provenance and history. Existing standalone project JSON files remain supported by either import control.

Older readable reports (`report.json`, directly or inside a stored/deflated ZIP) omitted polygon boundaries and full history. Detect/draw the matching wells first, keeping their saved IDs, then import the report to recover the frame labels or egg counts onto the current geometry. Image order and well locations are checked; a mismatch stops the import without changing annotations. Missing historical geometry cannot be reconstructed from a numbered screenshot. There is no need to manually re-enter the saved labels. Imports read only the selected JSON metadata from archives, validate its CRC, and support metadata up to 32 MB.

For CCRT, **Empty** occupancy applies to the entire well: its outline turns cyan with a dotted pattern and an E marker, the state selector displays **Empty — no state applicable**, and summaries/exports report empty frames rather than coma/awake/uncertain. Previous state annotations remain stored and become visible again if occupancy changes back. Empty wells remain excluded from the original binary coma/awake head, but explicit manual Empty occupancy is exported as supervised occupancy data for every applicable frame, subject to boundary review/visibility.


## Original models and reproducible comparisons

`requirements.txt` includes Keras with its Torch backend for both original CNNs and fine-tuning. Loading is lazy. Choose Original CNN 1 or 2 in CCRT to analyze images/sequences; model probabilities/version and manual labels remain separate. Full commands and migration notes are in [WORKFLOW.md](WORKFLOW.md).

Both supplied artifacts accept 200×200 grayscale images and output **coma, awake** in the order documented by the original `backend/services/cnn.py`. They have no trained Empty class. Shared inference/training preprocessing reproduces contrast alpha=1.5/beta=120, BGR-to-gray, Otsu threshold, 200×200 resize and normalization. Polygon-exterior pixels are replaced with the interior background before thresholding, then white, to exclude other wells; this mask adaptation is explicitly versioned. The legacy networks' original training-set provenance is not supplied.

Default Empty handling is a separate **uncalibrated foreground screen**, based on relative dark/chromatic components inside an eroded well mask; ambiguous evidence and apparently empty clipped wells stay Unknown, with no invented Empty confidence. Review it on each experimental setup. When a selected training subset includes both Empty and occupied examples, fine-tuning fits a separate logistic occupancy head on exactly those annotations. A single-class subset retains an existing learned occupancy head, or the explicit foreground screen when starting from original weights. Occupancy is not written back as manual ground truth. Model versions fingerprint both CNN weights and occupancy/preprocessing behavior.

Annotate only the wells/frames you need. Unannotated wells are omitted from targets, never treated as Empty or coma. Review boundaries and confirm single/empty occupancy for the annotated wells; unknown, uncertain, partial, multiple and unreviewed data stay flagged. Training manifests record detected/annotated/eligible coverage, so "all annotated wells" does not falsely imply a fully annotated grid.

Before exporting a training ZIP, set the shared seed and optionally the poor/rich-pose **Well IDs**. No special annotation notes are required. `comparison-plan.json` saves actual well UIDs, annotation IDs and seeds for:

- All eligible annotated wells.
- 1 run of 25 random individuals, 2 of 10, 4 of 5 and 5 of 2. Empty wells are retained in the full selection and excluded from random individual budgets.
- One run for each manually chosen poor/rich-pose well.

Each selection includes all eligible labels of its selected wells. Oversized budgets are explicitly skipped. Repetitions are independently seeded; subsets may overlap or repeat when the pool is small, and their exact IDs are saved. Every model reads the **same plan**; the plan is cryptographically bound to its exported manifest. Every run starts from its parent model, rather than continuing the previous run. Runs with no coma/awake labels are explicitly skipped because the supplied binary CNN cannot update from Empty alone. Missing classes in a small subset are not filled using other wells or test images.

For independent evaluation grids, check **Export as test set**. The ZIP contains original images and annotations/geometry, without duplicate crops, training selections, loader or project/history. Unannotated wells remain available for prediction but do not enter metrics. Use a different experiment/split-group ID for independent test grids; the CLI rejects overlap by split group or image SHA-256. Test grids never fit the occupancy head, update CNN weights, perform early stopping or choose subsets. Save a project/report separately if you need to continue editing a test-grid review.

Unzip each dataset and run from `CCRT/logiciel-v2_2`:

```sh
# Evaluate original weights on one or several held-out grids.
.venv/bin/python tools/model_experiments.py evaluate \
  --test /data/test-grid-1 /data/test-grid-2 --output /data/baseline-results

# Same saved plan for both supplied architectures; fine-tune each selected run.
.venv/bin/python tools/model_experiments.py compare \
  --train /data/train-grid --test /data/test-grid-1 /data/test-grid-2 \
  --models legacy-model1 legacy-model2 --epochs 5 --batch-size 4 \
  --output /data/comparison-results

# Start small; the full plan intentionally launches many training runs.
.venv/bin/python tools/model_experiments.py compare \
  --train /data/train-grid --test /data/test-grid-1 \
  --runs all-annotated-wells poor-poses rich-poses --epochs 1 \
  --output /data/comparison-smoke
```

`--mode scratch` uses the same architectures and selections with fresh initialization; default `finetune` copies parent weights. Outputs include the unchanged plan, per-grid confusion matrices/class precision/recall/F1, raw predictions with scored/unscored flags, `summary.csv`, trained `model.keras` files and sidecars documenting seed, exact samples, parent version, preprocessing and hyperparameters. No train/validation/test split is secretly regenerated per model; test grids are external. Related frames should stay together, and experimental independence must still be established by the researcher.

To select trained artifacts in the app, set `FLYSCOPE_MODELS_DIR` to the comparison output directory before restarting Flask. Sidecars are discovered recursively and checked against the model hash/preprocessing contract. Keep datasets, generated weights and research results outside Git. This legacy trainer implements the two original Keras architectures. The new three-family suite uses `tools/model_suite.py` and `FLYSCOPE_SUITE_MODELS_DIR`; see [WORKFLOW.md](WORKFLOW.md).

With the server running, `tests/experiments-browser.cjs` checks actual partial-grid training/test ZIPs, fixed selections, model predictions, supervised Empty exports and the single-well limit. `tests/test_experiment_data.py` checks held-out group/hash separation, immutable sample membership and scoring only annotated wells. The supplied crop labels and synthetic workflow checks do not establish out-of-sample biological accuracy.
