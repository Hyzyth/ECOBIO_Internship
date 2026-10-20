# Maintaining FlyScope

Start with [WORKFLOW.md](WORKFLOW.md) for setup and commands, and
[TRAINING_REVIEW.md](TRAINING_REVIEW.md) for the first experiment's evidence and
new validation protocol. One `requirements.txt` and Python environment run the
app, all model families, anatomical training and diagnostics. Keep generated
research data, weights, caches and results outside Git.

## Where work belongs

| Area | Files | Responsibility |
| --- | --- | --- |
| Browser review | `frontend/templates/index.html`, `frontend/static/review.js` | Staged workflow, controls, import/export and model selection |
| Annotation semantics | `frontend/static/review-core.js` | Effective/manual states, timing, geometry and recovery decisions |
| Shared exported selections | `frontend/static/dataset-plans.js` | Deterministic manual-only whole-well budgets |
| Dataset contract | `backend/services/dataset_format.py`, `tools/model_experiments.py` | Read old/new exports, verify hashes and prevent train/test overlap |
| Suite entry point | `tools/model_suite.py` | CLI, data/context access, saved plans, orchestration and evaluation |
| Validation protocol | `tools/suite_validation.py` | Group isolation, validation fingerprints, weights and best-checkpoint selection |
| Model fitting | `tools/suite_training.py` | Real fitting of appearance, pose/RF and flow/two-stream/TGM stages |
| Automatic recovery settings | `tools/recovery_selection.py` | Search reserved validation predictions; never receives test data |
| Runtime decisions | `backend/services/recovery_policy.py`, browser core | Apply saved settings; retain raw outputs separately |
| Architectures/inference | `backend/services/suite_models.py`, `upstream_architectures.py` | Preprocessing, model definitions, features, batching and immutable artifact versions |
| App model registration | `backend/services/suite_adapter.py`, `model_registry.py`, `tools/suite_worker.py` | Discover checkpoints and lazy inference workers |
| Training I/O | `tools/training_support.py`, `plot_training.py` | Progress/CSV, plots, pixel cache and prefetch |
| Anatomical training | `tools/sleap_pose.py`, `pose_quality.py` | Native SLP import, grouped holdout, balanced heatmaps, PCK and overlays |
| Anatomical inspection | `tools/pose_diagnostics.py` | Source-label metrics or qualitative CCRT overlays |
| Reports | `backend/services/suite_metrics.py`, `tools/audit_comparison.py`, `audit_plots.py` | Raw/decision metrics, censoring, compact result audit |

`model_experiments.py` also preserves the older Keras CNN baseline tools. CCRT
manual labels and anatomical labels are different inputs. Egg counting has its
own independent-image annotation/export skeleton; these CCRT commands do not
train an egg model.

## Scientific boundaries to preserve

- Targets are eligible manual annotations only. Missing truth is neither Empty
  nor Coma. Never use model outputs as ground truth implicitly.
- Validation wells are reserved before fitting selections. Training and validation
  retain all eligible frames of their respective wells; no random frame leakage.
  Whole-grid grouping is available when there are multiple training experiments.
- Validation selects checkpoints and recovery settings. Test datasets are accepted
  by orchestration for scoring, but never by the fitting/selection APIs.
- Cached pixels contain no fitted normalization or labels. Causal temporal context
  uses current/past frames of the same well, not future frames or setup images.
- Model metadata binds weights, artifacts, preprocessing, selection and policy.
  Preserve old checkpoint semantics; change the version when behavior changes.
- Report raw outputs, decision coverage, class support, censoring and recovery
  denominators. An abstention is not an invented state or successful recovery.
- Acquisition timing and physical experiment identity matter. Renaming the same
  grid is not a new independent experiment. Frame IDs remain usable when timing
  is unknown.

## Artifacts and debugging

`annotations.json` plus `images/` is the source export. `comparison-plan.json`
records exported budgets; a new `validation-plan.json` adds the common reserved
cohort and smaller study. Do not edit an existing manifest or plan in place.

Each comparison produces live `progress.jsonl`, `training.csv`, `loss.png`, model
folders and evaluation reports. Each model has `model.json`, weight artifacts and
`recovery-policy-search.json`. `validation.selected_epochs` identifies restored
checkpoints. Missing class support and failed automatic selection are explicit.
No optimizer/partial-epoch resume is implemented: use a new output directory.

Pose training writes `pose.pt`, `pose.json` and `pose-validation/`. Inspect PCK and
overlays, not just heatmap loss. A CCRT preview can show transfer failures but has
no anatomical accuracy without keypoint ground truth. An older parent that saw
proposed validation frames cannot create an independent validation checkpoint.

Use `audit_comparison.py` to share a small report instead of gigabytes of repeated
JSON predictions. It accepts older completed results and never alters them.

## Checks when changing code

Run the development checks in the workflow. `tests/test_validation_protocol.py`
covers isolation, weighting, best-checkpoint restoration, policy selection,
abstention, pose split/metrics and transforms. Node tests cover export budgets,
annotation semantics and browser recovery parity. `validate_model_suite.py`
actually fits and reloads nine checkpoints on synthetic data, including every
DeepEthogram stage, independent validation, and cached/uncached equivalence.

CI additionally trains/fine-tunes a native SLEAP checkpoint and exercises browser
annotation, imports, large exports and registered-weight inference. These are
software contracts, not biological accuracy claims. A new accuracy claim requires
representative independent experimental grids and an unchanged final-test protocol.
