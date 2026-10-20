# Using the app

Run `python app.py` from the repository root after following [WORKFLOW.md](WORKFLOW.md).
The header switches between **CCRT** and **Egg counting**. Images remain selected
in the browser; detection and prediction send the current frame to the local server.

## CCRT annotation

1. **Images:** choose an ordered sequence or independent-image mode, select images
   or a folder, and inspect filename/timestamp ordering. Sequence-only timing and
   playback options stay hidden for single images.
2. Set the acquisition interval. For raw input choose the first frame to analyze,
   excluding setup images. For trimmed input supply a known delay if available;
   an original frame number alone does not establish elapsed time.
3. **Wells:** choose a grid or single-well layout. Suggest boundaries, remove false
   proposals, resize/move them or draw polygons. Single-well layout permits one
   identity. Automatic proposals are suggestions, not an occupancy estimate.
4. Approve stable boundaries across the sequence. Use alignment if the camera
   shifts; inspect/correct its proposals. Remove a well with the button above the
   image, the removal tool, or Delete/Backspace outside a text input. IDs stay stable.
5. **Review:** check Empty/Single/Multiple/Obscured occupancy. Empty is cyan with an
   E marker; multiple individuals invalidate CCRT measurements. Select a well and
   inspect T−1/T/T+1. Use Coma through this frame, Awake from this frame onward, or
   explicit Uncertain labels. Ranges and undo help annotate long sequences.
6. **Export:** save a project backup or readable report before closing the page.
   Training can use only a few annotated wells; unannotated wells are never negative
   examples. Optional grid/per-well species fields support later mixed grids.

Coma is red, Awake bright green and uncertainty yellow. Playback has previous/next,
play/pause, speed, a frame slider and direct jumping. The summary compresses long
state timelines instead of drawing one unreadable marker per frame.

## Optional predictions

Choose a trained model under Review → Optional model predictions. New checkpoints
supply validation-selected confidence/recovery settings automatically. If no reliable
settings could be validated, automatic Awake confirmation stays Unsure. The UI
explains that status. Disable saved settings to access advanced manual overrides.
Existing model files keep their historical settings until retrained or overridden.

For sequences, observed Coma is required before irreversible Awake confirmation by
default. The onset and later confirmation frame are both retained. Offline inference
can fill uncertain gaps between matching confident states; the selected-well details
identify the bounding frames. Coma-to-Awake gaps remain unresolved. Raw probabilities,
inferred states and manual corrections are stored separately. Your explicit manual
uncertainty is preserved. Single-image predictions do not require a temporal baseline.

Original CNNs remain available, with no learned Empty class in their supplied weights.
New ResNet, pose/RF and temporal checkpoints are discovered after setting
`FLYSCOPE_SUITE_MODELS_DIR` and restarting Flask. Training and diagnostic commands
are documented only in the workflow to avoid conflicting command copies.

## Save, resume and export

| Export | Purpose |
| --- | --- |
| Project JSON | Editable geometry, labels, raw predictions, corrections, policy and history |
| Readable report ZIP | Compact per-well results/summary, HTML report, numbered locations and project backup |
| Machine analysis ZIP | Frame records, boundaries, probabilities, model/policy metadata and corrections; no source pixels |
| Training dataset | Original analysis-window images, eligible manual labels, shared geometry, hashes and comparison plan |
| Test dataset checkbox | Independent evaluation images/labels; omits training selections and heavy editable history |

Readable CCRT reports prioritize occupancy, first Awake frame/image/ID and status.
CCRT seconds require a confirmed origin, single occupancy and an observed recovery.
Already-Awake, unknown gaps, contradictory labels and missing events remain explicit.
Polygon arrays stay in machine exports; location maps identify wells for readers.

To resume, select the same original images in saved order, then import a project or
report under Images. Import checks identities and geometry before replacing labels.
A test dataset is not an editable backup: save a project separately.

For thousands of frames, use **Save dataset to folder** in Chrome/Chromium or Edge.
The app creates a new subfolder and writes a completion marker last. ZIP exports are
split into ordinary parts around 256 MiB; download every part and extract all parts
into the same empty directory. A ZIP download does not create a folder automatically.
No per-well PNG copies are generated: trainers crop source images lazily.

## Egg counting

Use independent images, without temporal controls. Whole-image counts are the
default; optional custom counting areas belong only to their image. Enter zero
explicitly for no eggs. Counts and exports are separate from CCRT. No egg model or
assumed well layout is supplied until the experimental data format is known.
