# Maintaining FlyScope

Start with [WORKFLOW.md](WORKFLOW.md) for commands and [TRAINING_REVIEW.md](TRAINING_REVIEW.md)
for the first experiment's evidence. Run commands from the repository root. One
root `requirements.txt` and Python environment run the app and every model.

## Layout and responsibilities

The active source is the `flyscope` package. Executable files in `tools/` are
small launchers or operational checks; training implementation is not duplicated
there. The old version-numbered app directory has been retired.

| Package/file | Responsibility |
| --- | --- |
| `app.py`, `flyscope/web.py` | Local server launcher and HTTP routes |
| `flyscope/paths.py` | Repository assets resolved independently of shell CWD |
| `flyscope/imaging/` | Polygon validation/parsing, well detection and camera alignment; no model registration dependency |
| `flyscope/data/format.py` | Standalone old/new export hydration, also served to exported loaders |
| `flyscope/data/datasets.py` | Manifest/image verification, manual targets, original comparison plans and independent-test checks |
| `flyscope/models/runtime.py` | Architectures, preprocessing, portable forest inference and immutable model versions |
| `flyscope/models/keras.py` | Two original CNN baselines and their occupancy handling |
| `flyscope/models/registry.py`, `adapter.py` | Discovery and lazy workers for trained models |
| `flyscope/models/upstream.py` | Verified upstream DeepEthogram architecture loader |
| `flyscope/models/recovery.py` | Raw-to-decision recovery rules and bounded gap inference |
| `flyscope/training/comparison.py` | Suite CLI and orchestration |
| `flyscope/training/data.py`, `plans.py` | Causal per-individual inputs and immutable matched budgets |
| `flyscope/training/validation.py` | Group isolation, weighting and best-checkpoint selection |
| `flyscope/training/trainers.py` | Real ResNet, pose/RF and flow/two-stream/TGM fitting with shared objectives |
| `flyscope/training/policy_selection.py` | Recovery search on reserved validation predictions only |
| `flyscope/training/support.py` | Progress/CSV, reusable pixel cache and bounded prefetch |
| `flyscope/training/pose.py`, `pose_quality.py` | Native SLP import, anatomical training, grouped holdout, PCK and overlays |
| `flyscope/training/legacy.py` | Fine-tuning original Keras baselines |
| `flyscope/evaluation.py` | Frozen checkpoints evaluated on independent test grids |
| `flyscope/reporting/` | Metrics, training plots, compact audit and anatomical diagnostics |
| `frontend/static/review-core.js` | Pure annotation/timing/geometry/recovery semantics |
| `frontend/static/review.js`, `templates/` | Staged workflow, controls, import/export and model selection |
| `frontend/static/dataset-plans.js`, `training-export.js` | Manual-only budgets and bounded-memory exports |
| `models/legacy/` | Supplied weights retained byte for byte |
| `archive/original-comparisons/` | Historical sample retained for reference, never loaded by the app |

Unused p5/Socket.IO/HTMX/Axios assets, obsolete socket/routes/image helpers and a
generated PyInstaller cross-reference were removed. Prior source remains in Git
history. Only browser assets used by the active app remain. Geometry operations
no longer initialize model registration, and dataset inspection does not import
training frameworks. RF checkpoints copy only pose weights/metadata, not repeated
pose logs, plots or diagnostic galleries.

`tools/model_suite.py`, `model_experiments.py`, `sleap_pose.py`, `pose_diagnostics.py`,
`audit_comparison.py` and `plot_training.py` retain their command names. Root
`app.py` also remains the launcher. Python code imports package implementations;
do not add implementation helpers back under `tools/`.

## Scientific boundaries

- Targets are eligible manual annotations only. Missing truth is neither Empty
  nor Coma. Model outputs and inferred gaps are not training labels.
- Reserve validation wells before fitting selections. Keep all eligible frames
  and causal clips with their individual; never split neighbouring frames at random.
  Whole-grid grouping is available with multiple training experiments.
- Validation selects checkpoints and recovery settings. Test datasets reach the
  evaluator, never the fitting or policy-selection API.
- Pixel caches fit no normalization or labels. Context uses current/past frames
  of the same well, with no future/setup images.
- Versions bind weight artifacts, preprocessing, selections and recovery policy.
  Preserve older checkpoint semantics and version changed inference behavior.
- Report class supports, raw accuracy, decision coverage, censoring and recovery
  denominators. An abstention is not an invented state or a successful recovery.
- Timing and physical experiment identity matter. A renamed grid is not a new
  experiment. Frame IDs remain usable when elapsed time is unknown.

CCRT behavior labels and anatomical keypoints are separate inputs. The egg page
has an independent-image annotation/export skeleton; CCRT commands do not train
an egg model.

## Artifacts and debugging

`annotations.json` and `images/` are source exports. `comparison-plan.json` records
exported budgets; a new `validation-plan.json` reserves a common cohort. Do not
edit existing manifests/plans. Existing exports and trained state dictionaries
remain readable after the source reorganization.

Comparisons produce `progress.jsonl`, `training.csv`, `loss.png`, model folders and
evaluations. Each checkpoint contains `model.json`, weights and
`recovery-policy-search.json`. `validation.selected_epochs` identifies restored
checkpoints. Missing classes and failed automatic selection are explicit. There
is no optimizer/partial-epoch resume; use a new output directory.

Pose training produces `pose.pt`, `pose.json` and `pose-validation/`. Inspect PCK,
per-node metrics and overlays. CCRT previews have no anatomical accuracy without
keypoint ground truth. A parent that saw proposed validation frames cannot create
an independent pose validation checkpoint. Use the compact audit command to share
results without gigabytes of repeated predictions.

## Checks

The workflow lists all development commands. `tests/test_validation_protocol.py`
covers isolation, weights, checkpoint restoration, policy search and pose quality.
Node tests cover annotation semantics, exports and recovery parity. The integration
check actually fits/reloads nine checkpoints, including every temporal stage,
independent validation and cached/uncached equivalence. CI additionally trains and
fine-tunes native SLEAP weights and exercises app imports, large exports and model
inference. These establish software contracts; accuracy claims need independent
experimental grids and a fixed final-test protocol.
