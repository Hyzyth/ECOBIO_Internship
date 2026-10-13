# CCRT models, datasets and comparisons

Run commands from `CCRT/logiciel-v2_2`. Research images, exported datasets and trained weights belong outside Git. The Flask app prepares labels and runs registered weights; training and batch comparison use the commands below. Egg counting remains a separate, independent-image annotation skeleton: its input format and trained counter have not yet been specified.

## Prepare the experiment in the app

Load an ordered sequence. For raw input, select **First frame to analyze** after inspecting setup images. Frame numbers and IDs remain those of the loaded sequence: choosing frame 101 does not reset the timer to zero. For pre-filtered input, select **Already trimmed / pre-filtered**. The app then clears the assumed time reference and original frame ordinal. Enter an original ordinal and the delay from cold-exposure end to the first loaded image only when known; an ordinal alone does not establish elapsed time.

The human report prioritizes the first Awake frame, image filename and frame ID. CCRT seconds appear only with a known time reference and an observed transition following Coma. Already-awake, missing, uncertain and contradictory transitions have explicit statuses. Machine reports retain timing bounds and durations within the observed window. Frame interval is the acquisition interval; playback speed is independent.

Define wells at the first analysis frame, review the geometry and approve stable boundaries across the analysis window. Alignment uses that frame as its reference. Setup images stay in the editable project but are excluded from labeling prompts, inference, temporal context and dataset exports. Use **Coma through this frame** and **Awake from this frame onward** for labels; leave genuinely ambiguous transition frames Uncertain. Manual labels and model decisions remain separate. Mark Empty, Single, Multiple or Obscured occupancy explicitly. Each eligible occupied well represents one individual. Multiple/obscured/unknown occupancy, partial boundaries and unreviewed boundaries are excluded from training targets.

Optional grid species metadata can be overridden per well for mixed-species grids. Species is exported and used for stratified metrics; it is not an input feature or inferred from the public pose dataset.

## Export and reuse partial labels

Choose **Training dataset ZIP** and extract it, for example to `/data/train-grid`. It includes original images for the entire analysis window, geometry, manual annotations, hashes, masked crops and an editable project. Unlabeled frames are retained as temporal context, never assigned a negative state. Only eligible manual targets enter supervised losses. App ZIP crops are for inspection and external consumers; all three trainers derive their own versioned crops from the same original images and geometry.

The saved default plan uses whole individuals:

| Selection | Repetitions |
| --- | ---: |
| All eligible annotated wells, including Empty | 1 |
| 25 randomly selected single-individual wells | 2 |
| 10 randomly selected single-individual wells | 5 |
| 5 randomly selected single-individual wells | 10 |
| 2 randomly selected single-individual wells | 20 |
| One poor-pose individual, selected by Well ID | 1 |
| One rich-pose individual, selected by Well ID | 1 |

That is 40 runs per architecture when at least 25 eligible occupied wells and both selected IDs are available. Enter the poor/rich IDs in comparison options before exporting; no additional pose-diversity label is needed. Those two runs are omitted if the IDs are not supplied. Oversized random budgets are recorded as skipped. Empty wells never enter random individual budgets. Repeated random subsets can overlap. Every selection contains all eligible annotations for its selected wells; every architecture uses exactly those saved IDs and seeds. No architecture silently resamples data or borrows labels to fill a missing class.

For independently collected evaluation grids, check **Export as test set**. This omits duplicate crops, training plans, loaders and project history, retaining the original analysis-window images, geometry and annotated targets. Save an editable project separately. Fully and partially annotated test grids are supported: all visible wells receive predictions, but only eligible annotated frames are scored. Missing ground truth is never evidence of Empty or of no recovery. Give each independent grid a distinct experiment ID. The runner rejects train/test experiment-ID and image-content overlap, and duplicate training grids.

## Install the three Python 3.10.7 runtimes

The base app and original Keras weights use `.venv`. The new suite uses `.venv-suite`; native SLEAP uses `.venv-sleap`. SLEAP 1.4.1 is pinned because newer SLEAP requires Python 3.11. Its TensorFlow 2.9 dependencies conflict with the app's current TensorFlow; isolate them rather than changing the deployment interpreter.

```sh
python3.10 -c 'import sys; assert sys.version_info[:3] == (3,10,7), sys.version'
python3.10 -m venv .venv-suite
.venv-suite/bin/python -m pip install pip==24.0
.venv-suite/bin/python -m pip install -r requirements-suite.txt
.venv-suite/bin/python -m pip check

python3.10 -m venv .venv-sleap
.venv-sleap/bin/python -m pip install pip==24.0
.venv-sleap/bin/python -m pip install -r requirements-sleap.txt
.venv-sleap/bin/python -m pip check
export FLYSCOPE_SLEAP_PYTHON="$PWD/.venv-sleap/bin/python"
```

Pip 24.0 is required for DeepEthogram's legacy Lightning dependency metadata. TLS and downloaded-artifact verification remain enabled. The suite requirements above install CPU Torch. `requirements-suite-cuda.txt` supplies a CUDA 12.1 alternative in a separate environment; GPU execution has not been hardware-tested. CPU training/inference were tested on Linux x86_64, exact Python 3.10.7. Windows runtime has not been validated.

The prepared cloud environment uses `/workspace/ecobio-venv-3107`, `/workspace/ecobio-suite-venv` and `/workspace/ecobio-sleap-venv` instead of the local names above.

For cloud training, set `FLYSCOPE_SLEAP_PYTHON=/workspace/ecobio-sleap-venv/bin/python` and `TORCH_HOME=/workspace/ecobio-data/torch-cache` to reuse the verified ImageNet download in writable storage. Run the suite commands with `/workspace/ecobio-suite-venv/bin/python` and native pose commands with `/workspace/ecobio-sleap-venv/bin/python`.

## Train the anatomical pose estimator

The creator's [SLEAP datasets](https://github.com/talmolab/sleap-datasets) repository provides the [Drosophila courtship release](https://github.com/talmolab/sleap-datasets/releases/download/dm-courtship-v1/drosophila-melanogaster-courtship.zip). Download/extract it outside Git; keep `courtship_labels.slp` and its `20190128_113421.mp4` together. The native SLP contains 100 labeled frames with two annotated individuals per frame. The importer resolves videos next to the label file and supports Windows paths when running on Linux.

```sh
QT_QPA_PLATFORM=offscreen .venv-sleap/bin/python tools/sleap_pose.py train \
  --labels /data/drosophila-melanogaster-courtship/courtship_labels.slp \
  --output /data/pose-base --epochs 25 --batch-size 8
```

This trains the actual SLEAP UNet backbone with anatomical confidence-map targets; hidden/missing landmarks do not contribute loss. It saves `pose.json` and `pose.weights.h5`, recording source SLP/video hashes, skeleton, selected instances, seed, parameters and loss. For fine-tuning on a compatible native SLP with overhead anatomical annotations, add `--parent /data/pose-base` and use a new output directory. Parent hash and skeleton must match. Coma/Awake state labels alone cannot train anatomical keypoints.

The public courtship dataset supplies anatomical pretraining, not Coma/Awake truth or proof of performance on overhead wells. Model B keeps the same external pose estimator for every CCRT subset, then trains its random forests from that subset only. Thus the CCRT sample budgets remain comparable. To study pose fine-tuning budgets too, supply actual anatomical annotations for those individuals and maintain independent pose train/test separation.

## Train the same selections across all three approaches

Implemented components:

| ID | Implementation | Initialization and targets |
| --- | --- | --- |
| `resnet18` | Torchvision ResNet-18, three output logits | ImageNet backbone by default; Coma/Awake/Empty labels |
| `sleap-rf` | Native SLEAP UNet keypoints plus scikit-learn random forests | Fixed externally trained pose backbone; state forest uses anatomical distances/visibility; separate foreground-feature occupancy forest uses selected Empty/occupied labels |
| `deepethogram` | Official DeepEthogram TinyMotionNet, hidden two-stream ResNet-18 features, TGM sequence classifier | Flow reconstruction on selected individuals' clips; ImageNet appearance/flow backbones; state and sequence losses on the same selected manual targets |

DeepEthogram uses a custom CPU-capable three-stage training runner around its actual upstream modules. It does not invoke the upstream GUI/Lightning trainer or claim Kinetics-pretrained behavior weights. Temporal inputs are causal: only the current frame and previous analysis frames are used. At the window start, the first analysis frame is repeated. Missing interior context images/geometry raise an error. DeepEthogram requires a sequence for both training and evaluation.

```sh
# Uses the app's saved comparison-plan.json automatically for one training grid.
.venv-suite/bin/python tools/model_suite.py compare \
  --train /data/train-grid --test /data/test-grid-1 /data/test-grid-2 \
  --pose-model /data/pose-base --output /data/ccrt-comparison \
  --epochs 5 --flow-epochs 3 --sequence-epochs 5 --device cpu
```

Use `--runs all-annotated-wells poor-poses rich-poses` to start with three selections instead of 40. `--architectures resnet18 sleap-rf deepethogram` is the default. `--initialization scratch` supports a controlled fresh-weight comparison; otherwise Torchvision verifies and loads ImageNet weights. Each run starts independently from its common base, not the preceding run's CCRT-trained weights. Species, labels and manual occupancy are never passed to held-out inference as answers. Output directories must be new, preventing accidental overwrites.

For several training grids, create one shared plan first:

```sh
.venv-suite/bin/python tools/model_suite.py plan \
  --train /data/train-grid-1 /data/train-grid-2 --seed 2026 \
  --poor-well 'MANIFEST_HASH_PREFIX::WELL_UID' \
  --rich-well 'OTHER_HASH_PREFIX::WELL_UID' --output /data/shared-plan.json
.venv-suite/bin/python tools/model_suite.py compare \
  --train /data/train-grid-1 /data/train-grid-2 \
  --plan /data/shared-plan.json --test /data/test-grid \
  --pose-model /data/pose-base --output /data/multiple-grid-comparison
```

Qualified IDs are recorded in the plan; a plain Well ID is accepted only when unambiguous. The manifest hashes bind plans to exact exports. The first plan command can be run without the poor/rich options to discover IDs, then rerun with those selections.

## Recovery decisions and evaluation

Raw three-class probabilities and confidence are retained. Low-confidence output becomes **Unsure** (amber/U); Unsure is an abstention, not a fourth training class. Defaults require confidence ≥0.8 and three consecutive high-confidence Awake frames. Once confirmed, the decision stays Awake to sequence end, while every contradictory raw prediction is retained. Retrospective reporting assigns recovery to the first frame of that sustained run and records the later confirmation frame separately. Carried-forward decisions have no invented confidence. Manual corrections always take priority.

`--false-awake-cost 2` weights false-Awake mistakes during supervised training; it is not an inferred belly-up orientation penalty. Pose visibility and anatomy are learned from keypoint data; the code does not guess whether folded legs or upside-down posture mean coma. Probability scores are not assumed calibrated. Choose `--confidence-threshold` and `--awake-frames` using separate training/validation data **before** final held-out evaluation. The runner never tunes on its test grids.

```sh
# Evaluate every saved checkpoint on one or multiple never-seen grids.
.venv-suite/bin/python tools/model_suite.py evaluate \
  --models /data/ccrt-comparison --test /data/new-test-1 /data/new-test-2 \
  --output /data/final-evaluation --device cpu
```

Open the generated `comparison.html` for a sortable comparison. `summary.csv`/`summary.json` identify architecture, selection, artifact version and test grid, with accuracy, F1, coverage, false-Awake counts and recovery error. Per-grid JSON retains class confusion matrices, precision/recall/F1/support, species strata, original probabilities, Brier score, calibration bins/ECE and class ROC-AUC/AP where defined. It scores both raw output and the recovery policy. Recovery errors use annotated observed transitions, excluding censored or uncertain ground truth; an incomplete negative timeline cannot count as false recovery. Metrics include frame IDs and optional CCRT seconds. Independent-image evaluation has no recovery metrics.

## Select trained weights in the app

```sh
export FLYSCOPE_SUITE_PYTHON="$PWD/.venv-suite/bin/python"
export FLYSCOPE_SLEAP_PYTHON="$PWD/.venv-sleap/bin/python"
export FLYSCOPE_SUITE_MODELS_DIR=/data/ccrt-comparison
.venv/bin/python app.py
```

After restart, model selection lists each trained architecture/selection with its immutable version. No untrained model pretends to predict. Existing Keras models remain available; `FLYSCOPE_MODELS_DIR` continues to discover legacy fine-tuned Keras weights. New suite workers are isolated from Flask and load weights lazily. Torch checkpoints use `weights_only=True`; forest checkpoints are portable JSON, avoiding pickle. Hashes and interpretation metadata are checked when weights load. Only select artifacts produced by this workflow. The app preserves manual annotations when rerunning weights; geometry edits invalidate affected predictions.

## Reproduce the software checks

```sh
.venv-suite/bin/python -m unittest tests.suite_contracts -v
QT_QPA_PLATFORM=offscreen .venv-sleap/bin/python tools/sleap_pose.py train \
  --labels /data/drosophila-melanogaster-courtship/courtship_labels.slp \
  --output /data/pose-smoke --epochs 1 --max-examples 8 \
  --size 64 --filters 8 --batch-size 2
FLYSCOPE_SLEAP_PYTHON="$PWD/.venv-sleap/bin/python" \
  .venv-suite/bin/python tools/validate_model_suite.py \
  --pose-model /data/pose-smoke --output /data/suite-smoke
```

The last command actually trains and reloads nine checkpoints: three architectures × full/poor/rich selections, then scores a separate partially annotated synthetic grid. All three DeepEthogram stages execute. It verifies software contracts, not biological accuracy. Real CCRT training still requires your app-annotated experimental grids, and the pose estimator may need overhead keypoint annotations for good transfer.
