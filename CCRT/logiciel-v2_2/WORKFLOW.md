# FlyScope: setup, annotation, training and evaluation

All commands below run from `CCRT/logiciel-v2_2` in **one Python 3.12 environment**, installed from **one `requirements.txt`**. The app, original CNNs, ResNet-18, SLEAP-NN/random forests and temporal model share it.

## Contents

1. [Install once](#1-install-once)
2. [Check the Blackwell GPU](#2-check-the-blackwell-gpu)
3. [Annotate images](#3-annotate-images)
4. [Choose and extract exports](#4-choose-and-extract-exports)
5. [Train the anatomical pose estimator](#5-train-the-anatomical-pose-estimator)
6. [Create the shared training plan](#6-create-the-shared-training-plan)
7. [Train the three approaches](#7-train-the-three-approaches)
8. [Evaluate all weights](#8-evaluate-all-weights)
9. [Load trained weights in the app](#9-load-trained-weights-in-the-app)
10. [Original CNN baselines](#10-original-cnn-baselines)
11. [Migration and troubleshooting](#11-migration-and-troubleshooting)
12. [Development checks](#12-development-checks)

## 1. Install once

The supported GPU setup is **Linux x86_64**, including suitable WSL2 installations, with an NVIDIA driver supporting your RTX PRO Blackwell card. Python is pinned to **3.12.12** in `.python-version`. Create a fresh environment rather than reusing a Python 3.10 one. Windows native GPU installation has not been validated.

On Debian/Ubuntu, install OpenCV's native libraries:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

Then, using an installed Python 3.12:

```bash
cd ECOBIO_Internship/CCRT/logiciel-v2_2
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python tools/check_runtime.py --prepare-models
```

If using `uv`, replace environment creation with `uv python install 3.12.12` and `uv venv --python 3.12.12 --seed .venv`, then activate it and run the same install/check commands. Future terminals only need `source .venv/bin/activate`.

The default Linux PyPI Torch 2.8 wheel includes **CUDA 12.8 and native `sm_120` Blackwell kernels**. It runs on CPU too. No full CUDA toolkit installation or Torch compilation is needed; the GPU's NVIDIA driver is needed. The initial download is several GB. An exclusively CPU workstation can optionally install CPU wheels with `python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`, using the same file.

SLEAP uses its current PyTorch backend (`sleap-nn`) and native-label reader (`sleap-io`). The original Keras CNNs use Keras's Torch backend. TensorFlow and separate interpreter variables are unnecessary. Keep `KERAS_BACKEND` unset or set it to `torch`.

DeepEthogram's full package pins incompatible old Torch/Lightning versions and rejects SLEAP's Hydra dependency. Model C loads the author's TinyMotionNet and TGM modules from a **pinned, SHA-256-verified author wheel**, cached outside the repository. Torchvision supplies the ResNet-18 streams, preserving the previous feature/fusion layout. The [upstream research license](https://github.com/jbohnslav/deepethogram/blob/master/license.txt) applies; third-party source is not redistributed here. The old GUI/trainer is not installed. `--prepare-models` performs the architecture download before training; ImageNet weights download at first ImageNet initialization. Retain these caches for offline work. `FLYSCOPE_CACHE_DIR` and `TORCH_HOME` can relocate them to writable storage.

## 2. Check the Blackwell GPU

Your **RTX PRO 3000 Blackwell with approximately 12 GB VRAM** is a target for this setup. Before a long experiment:

```bash
nvidia-smi
python tools/check_runtime.py --require-cuda --prepare-models
```

The check reports GPU name, compute capability, compiled architectures, VRAM, Torch's CUDA runtime and Python version. It executes convolution, optical-flow grid sampling and pose-network **forward/backward operations on the GPU**. It fails if CUDA is unavailable rather than silently using CPU.

Use a current NVIDIA driver supporting your exact card. CUDA 12.8's Linux toolkit driver baseline is 570.26; a specific RTX PRO model can need a newer driver. The CUDA version displayed by `nvidia-smi` is the driver's supported version, not the runtime installed in Python. A newer driver reporting CUDA 13 can run this CUDA 12.8 stack.

For 12 GB, start with **batch size 4, image size 128, temporal window 5**, running training jobs sequentially. If memory runs out, reduce batch size to 2 or 1 and use a new output directory. The random forest trains on CPU; pose training/inference and the neural classifiers can use CUDA. Playback FPS and acquisition interval are independent of batch size.

The cloud validation machine has no NVIDIA GPU. Training/inference was tested on CPU using the CUDA-capable wheel, and the Blackwell kernels are present; physical GPU execution must be checked on your workstation. Some CUDA flow-backward kernels are nondeterministic. Saved data selections and seeds are identical across models; bit-for-bit GPU training reproducibility is not promised.

## 3. Annotate images

```bash
python app.py
```

Use the local server on port 5000 in your browser. CCRT and Egg counting have separate pages.

### CCRT

1. **Images:** select Ordered sequence, browse the folder, inspect chronology and set the acquisition interval.
2. For raw input, select **First frame to analyze** after setup. Earlier images are excluded from annotation prompts, inference and training exports; original frame indices and timer offsets remain intact.
3. For pre-filtered input, select **Already trimmed / pre-filtered**. The assumed time reference is cleared. If known, enter the delay from cold-exposure end to the first loaded image, confirm the reference, and optionally supply its original frame ordinal. An ordinal alone does not establish elapsed time.
4. Optionally set grid species; per-well overrides support future mixed-species grids.
5. **Wells:** suggest/draw boundaries, remove incorrect proposals and correct geometry. Approve stable boundaries across the analysis window. Alignment uses the first analysis frame as reference. Single-well mode permits one identity.
6. **Review:** confirm Empty/Single/Multiple/Obscured occupancy. Inspect T−1/T/T+1 and label **Coma through this frame** / **Awake from this frame onward**. Leave ambiguous frames Uncertain. Only reviewed, sufficiently visible human targets with empty/single occupancy are eligible for training. Unannotated wells are never negative examples.
7. **Export:** give the physical grid a unique experiment ID and save an editable backup before closing the page.

Human reports prioritize the first Awake frame, filename and frame ID. Exact CCRT seconds require a known origin and an observed Coma→Awake transition. Already-awake, missing, uncertain and contradictory transitions have explicit statuses. Machine exports retain timing bounds and observed-window durations.

For independent CCRT images, select Single image / Non-sequential. ResNet and pose/RF support them; the temporal model requires an ordered multi-frame sequence.

### Egg counting

Select one image or independent images on the Egg counting page. Whole-image counts are the default; custom areas are optional. Counts/areas belong to that image alone. This remains a skeleton pending the actual egg data format and counter; CCRT training commands do not train an egg model.

## 4. Choose and extract exports

| Export | Use and contents |
| --- | --- |
| Project backup JSON | Resume editable geometry, labels, predictions and history; select the same original images before importing |
| Human report ZIP | Read/share compact HTML/CSV, occupancy, recovery frame IDs, optional CCRT, maps and project backup |
| Machine analysis ZIP | Frame-level states/geometry, raw probabilities, model versions, corrections and timing; no source pixels |
| Training dataset ZIP | Original analysis-window images, human targets, geometry, hashes, inspection crops and comparison plan |
| Dataset ZIP with **Export as test set** checked | Held-out images/targets/geometry; omits duplicate crops, plans and editable history |

Save one train ZIP and one or more **independent physical grids** as test ZIPs. Partial annotation is valid for either role. All analysis-window images remain available for temporal context; only eligible manual annotations enter losses. A test ZIP is not an editable backup.

Set a data directory outside Git once per terminal, then extract exports:

```bash
export DATA_DIR=/absolute/path/to/ecobio-data
mkdir -p "$DATA_DIR/train-grid" "$DATA_DIR/test-grid-1"
unzip /path/to/train.zip -d "$DATA_DIR/train-grid"
unzip /path/to/test.zip -d "$DATA_DIR/test-grid-1"
```

Each extracted directory must directly contain `annotations.json` and `images/`. Do not edit an export after creating its plan: manifest/image hashes bind the exact data. Output directories must be new; training refuses to overwrite existing results.

## 5. Train the anatomical pose estimator

This stage is required for **Model B**. State labels cannot supervise anatomical keypoints. The creator's [SLEAP datasets](https://github.com/talmolab/sleap-datasets) include a native courtship example with 100 labeled frames and two individuals per frame:

```bash
mkdir -p "$DATA_DIR/public-pose"
curl --fail --location --output "$DATA_DIR/public-pose/courtship.zip" \
  https://github.com/talmolab/sleap-datasets/releases/download/dm-courtship-v1/drosophila-melanogaster-courtship.zip
unzip -n "$DATA_DIR/public-pose/courtship.zip" -d "$DATA_DIR/public-pose"
python tools/sleap_pose.py train \
  --labels "$DATA_DIR/public-pose/drosophila-melanogaster-courtship/courtship_labels.slp" \
  --output "$DATA_DIR/pose-base" --epochs 25 --batch-size 4 --device cuda
```

Keep the SLP and companion video together. Moved and Windows-style video paths are resolved. Training uses the actual SLEAP-NN UNet and anatomical confidence maps; missing/hidden keypoints do not contribute loss. Outputs are **`pose.json` and `pose.pt`**, recording skeleton, source/weight hashes, seed and loss.

The public courtship data is anatomical pretraining, not CCRT truth or proof of overhead-well transfer. Fine-tune on suitable overhead anatomical labels with:

```bash
python tools/sleap_pose.py train --labels /path/to/overhead-labels.slp \
  --parent "$DATA_DIR/pose-base" --output "$DATA_DIR/pose-overhead" \
  --epochs 25 --batch-size 4 --device cuda
```

Parent hash and skeleton must match. Use the same pose parent across CCRT selections. Model B trains state and occupancy forests separately per selected CCRT subset; the external pose parent stays fixed, preserving the CCRT annotation budgets.

## 6. Create the shared training plan

Set the seed and poor/rich-pose **Well IDs** in the app's training-export comparison options. `comparison-plan.json` saves actual well/annotation IDs and seeds. No special pose-diversity notes are needed.

| Selection | Runs |
| --- | ---: |
| All eligible annotated wells, including Empty | 1 |
| 25 random single individuals | 2 |
| 10 random single individuals | 5 |
| 5 random single individuals | 10 |
| 2 random single individuals | 20 |
| One selected poor-pose individual | 1 |
| One selected rich-pose individual | 1 |

That is **40 runs per architecture** when enough eligible individuals and both IDs exist. Oversized budgets are explicitly skipped; poor/rich runs are omitted if unspecified. Empty wells never enter random individual budgets. Repeated subsets can overlap; all eligible labels of a selected well are retained. Every architecture uses the **same saved selections and seeds**.

If needed, generate a plan explicitly (also supports several `--train` directories):

```bash
python tools/model_suite.py plan --train "$DATA_DIR/train-grid" --seed 2026 \
  --poor-well W12 --rich-well W45 --output "$DATA_DIR/shared-plan.json"
```

Replace those IDs with your choices. If names are ambiguous across grids, use `manifest-hash-prefix::well-uid` from the plan. Run once without poor/rich options to inspect qualified IDs, then regenerate with the desired choices.

## 7. Train the three approaches

Approaches are `resnet18` (appearance), `sleap-rf` (anatomy plus separate occupancy forest), and `deepethogram` (optical flow, two-stream visual features and TGM sequence classification).

Start with three selections before launching the full 120-run suite:

```bash
python tools/model_suite.py compare \
  --train "$DATA_DIR/train-grid" --plan "$DATA_DIR/shared-plan.json" \
  --test "$DATA_DIR/test-grid-1" --pose-model "$DATA_DIR/pose-base" \
  --runs all-annotated-wells poor-poses rich-poses \
  --epochs 5 --flow-epochs 3 --sequence-epochs 5 \
  --batch-size 4 --image-size 128 --window 5 \
  --confidence-threshold 0.8 --awake-frames 3 --false-awake-cost 2 \
  --device cuda --output "$DATA_DIR/comparison-pilot"
```

Omit `--plan` to use a single grid's exported plan automatically. Remove `--runs` and choose a new output directory for all available selections. Use `--architectures resnet18 sleap-rf` for independent images, or choose any one approach. Model B is the only approach needing `--pose-model`.

Defaults fine-tune ImageNet backbones; `--initialization scratch` starts fresh weights. Each run restarts independently from the common base, never the preceding subset's CCRT weights. The temporal model uses current/past analysis frames from selected individuals; no setup/future frames or unselected wells enter targets. Missing interior context/geometry is an explicit error.

`--device cuda` requires the GPU; `cpu` is the tested fallback and `auto` chooses CUDA when available. Low confidence becomes **Unsure**, an abstention rather than a fourth training class. Defaults require three consecutive high-confidence Awake frames; once confirmed, the decision stays Awake. Recovery is retrospectively assigned to the first frame of that run, with the confirmation frame recorded separately. Raw contradictions and manual corrections remain intact.

The false-Awake cost increases Coma/Empty class weights relative to Awake; there is no guessed belly-up orientation rule. Scores are not assumed calibrated. Select thresholds, costs, streak lengths and other hyperparameters on separate training/validation data **before** final held-out tests. Test grids never fit weights, choose subsets or tune thresholds. CCRT annotation splits are shared across approaches; their external ImageNet/pose pretraining differs and is documented.

## 8. Evaluate all weights

Export independent test grids with **Export as test set** checked and extract them. Then:

```bash
python tools/model_suite.py evaluate --models "$DATA_DIR/comparison-pilot" \
  --test "$DATA_DIR/test-grid-1" "$DATA_DIR/test-grid-2" \
  --device cuda --output "$DATA_DIR/final-evaluation"
```

Only list existing test directories. Experiment IDs/image hashes prevent train/test overlap. Renaming the same physical grid does not make it independent. Fully/partly annotated test grids are accepted: all available wells receive predictions; only eligible human targets are scored. Missing truth is never a negative recovery.

Open `comparison.html` for the sortable comparison. `summary.csv`/`summary.json` identify architecture, selection, immutable version and test grid. Per-grid JSON contains raw/decision confusion matrices, class precision/recall/F1/support, accuracy, coverage/accuracy when decided, false-Awake counts, species strata, calibration/Brier/ECE, defined ROC-AUC/AP and recovery frame errors/censoring. CCRT seconds require known timing. Uncertain/censored truth does not become an exact event error. Results remain separate per held-out grid so grid-to-grid variation stays visible.

## 9. Load trained weights in the app

Stop Flask, then restart in the **same activated environment**:

```bash
export FLYSCOPE_SUITE_MODELS_DIR="$DATA_DIR/comparison-pilot"
python app.py
```

Model selection discovers trained architecture/selection checkpoints recursively. App inference defaults to CPU to conserve GPU memory; optionally set `FLYSCOPE_INFERENCE_DEVICE=cuda` or `auto` before starting. No suite/pose interpreter paths are required. Lazy inference worker processes use the current interpreter.

Select a checkpoint and analyze a review. Predictions, probabilities, versions, recovery decisions and manual corrections remain separate. Geometry edits invalidate dependent temporal predictions; changing the analysis start clears temporal predictions based on the old context. Export the project to retain results.

## 10. Original CNN baselines

Original weights remain selectable. They have Coma/Awake outputs with a separate occupancy screen/head. Their 200×200 grayscale contrast/Otsu preprocessing and versioned polygon masking remain unchanged.

```bash
python tools/model_experiments.py evaluate \
  --test "$DATA_DIR/test-grid-1" --output "$DATA_DIR/original-baseline"
python tools/model_experiments.py compare \
  --train "$DATA_DIR/train-grid" --test "$DATA_DIR/test-grid-1" \
  --models legacy-model1 legacy-model2 --epochs 5 --batch-size 4 \
  --runs all-annotated-wells poor-poses rich-poses \
  --output "$DATA_DIR/original-finetuning"
```

These tools use the same environment, with the older baseline report format rather than the complete new-suite recovery/calibration report. Set `FLYSCOPE_MODELS_DIR` to their output directory before restarting Flask to select the fine-tuned legacy weights.

## 11. Migration and troubleshooting

- **Resume annotation:** select the same original images in saved order, then Images → Resume a saved review → Import report/project. Existing project and dataset exports remain compatible; this environment change does not require reannotation.
- **Old pose weights:** prior TensorFlow `pose.weights.h5` checkpoints do not become Torch weights. Retrain from the same native SLP labels, then rerun SLEAP/RF selections with the same CCRT plan. The loader gives an explicit migration message. Original CNNs and previous ResNet/temporal parameter layouts remain supported; temporal predictions were compared between implementations.
- **CUDA unavailable/unsupported:** activate the correct environment, inspect `nvidia-smi`, update the model-supported driver and rerun the GPU check. The previous Torch 2.4/CUDA 12.4 setup is unsuitable for Blackwell.
- **Out of VRAM:** stop other training jobs and reduce batch size using a new output directory. Record changed hyperparameters; keep saved data selections identical across models.
- **Missing pose video:** keep the SLP beside its companion video. An app state-label ZIP is not an anatomical-keypoint SLP.
- **Cache hash mismatch:** remove only the named DeepEthogram cache directory and rerun `python tools/check_runtime.py --prepare-models`. Verification stays enabled.
- **Unknown timing offset:** leave the origin unconfirmed and report frame IDs; filenames do not establish cold-exposure-end time.
- **Egg model pending:** continue independent-image annotations; its eventual training contract is separate from CCRT.

Keep data, backups, caches, generated weights and results outside Git. Export before closing the browser. Software checks do not establish biological accuracy; that needs your annotated experimental grids.

## 12. Development checks

```bash
python -m pip check
python -m unittest discover -s tests -v
python -m unittest tests.suite_contracts -v
node --test tests/*.test.js
```

An actual small training/reload check, using new output directories:

```bash
python tools/sleap_pose.py train \
  --labels "$DATA_DIR/public-pose/drosophila-melanogaster-courtship/courtship_labels.slp" \
  --output "$DATA_DIR/pose-smoke" --epochs 1 --max-examples 8 \
  --size 64 --filters 8 --batch-size 2 --device cpu
python tools/validate_model_suite.py \
  --pose-model "$DATA_DIR/pose-smoke" --output "$DATA_DIR/suite-smoke"
```

This trains/reloads nine checkpoints (three approaches × full/poor/rich), executes all temporal stages and scores an independent partly annotated synthetic grid. CI uses the same requirements with CPU wheels for its GPU-less runner. Optional browser development tools are described in [README.md](README.md#validation).
