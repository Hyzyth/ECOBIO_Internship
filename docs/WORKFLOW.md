# FlyScope: setup, annotation, training and evaluation

All commands below run from `.` in **one Python 3.12 environment**, installed from **one `requirements.txt`**. The app, original CNNs, ResNet-18, SLEAP-NN/random forests and temporal model share it.

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
cd ECOBIO_Internship
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

For 12 GB, **image size 128 and temporal window 5** are conservative starting points. The default batch size 4 prioritizes fitting every architecture. Try `--resnet-batch-size 32 --pose-batch-size 16 --temporal-batch-size 8` for more work per GPU call, then measure throughput and memory on your machine. These are starting values, not GPU-validated limits. If memory runs out, reduce the relevant override and use a new output directory. Run training jobs sequentially. The random forest trains on CPU; pose inference and the neural classifiers can use CUDA. Playback FPS and acquisition interval are independent of batch size.

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

### Updating automatic well detection

Stop the running Flask server after updating the code, restart it with `python app.py`, and reload the browser page. A page served by an older running detector now asks for a restart before accepting proposals. The new fitter searches the original-resolution outer lip rather than relying on the smaller Hough ring. Nonzero margin includes at least three source pixels to protect edge details at small resolutions. Nearby wells limit added margin without shrinking the fitted boundary; weak rims are recovered only with neighbour and image evidence. Check flies near the edges before approving boundaries. Use the existing resize/manual tools to retain annotations; replacing all detection proposals asks for confirmation and clears that review, so save a project backup first.

### Large experiments and exported files

For thousands of frames, use **Save dataset to folder (large experiments)** in Chrome/Chromium or Edge. Choose a parent folder; the app creates a new `flyscope-ccrt-…` (or `flyscope-eggs-…`) subfolder and writes originals one at a time. Wait for **Dataset saved in …**. `dataset-complete.json` is written last; a cancelled/failed export folder must be exported again. Your app annotations are retained. Keep the source images until you have verified the saved dataset.

**Export training/test dataset ZIP** works in other browsers too. Large datasets are automatically divided into independent ZIP files around 256 MiB each, avoiding the old 4 GB / 65,535-file limit. Download links remain on the page until the next dataset export or navigation/reload. Download **every part**, then extract all parts into the **same empty directory**. These are ordinary ZIPs, not a multipart `.z01` archive. ZIP downloading does not create a folder automatically; check the browser's Downloads list if it starts automatically. Use the visible link if it does not. Very large individual metadata files need direct folder saving.

There are no pre-generated per-well PNG crops. `images/` contains one original for every analysis frame, including unlabeled context required by temporal models. Schema 3 `annotations.json` stores manual targets plus shared normalized boundaries and sparse frame overrides. `flyscope/data/format.py` resolves these references for all three architectures and the original CNN trainers; old dataset exports still work. The included `dataset.py` and `dataset_format.py` provide lazy masked RGB crops for other trainers. You can check a training export before starting:

```bash
python "$DATA_DIR/train-grid/dataset.py" "$DATA_DIR/train-grid"
```

`project.json` remains an editable backup with geometry, model outputs, manual corrections and history; derived results are omitted to avoid repeating boundaries for every frame. Select the original images in the app and import this JSON to continue editing. Human reports also include this compact backup. Test exports omit the backup and training loaders: save a project separately if needed. Models and plans still use the original image hashes, manual-only targets, exclusion flags and stable Well IDs.

### Egg counting

Select one image or independent images on the Egg counting page. Whole-image counts are the default; custom areas are optional. Counts/areas belong to that image alone. This remains a skeleton pending the actual egg data format and counter; CCRT training commands do not train an egg model.

## 4. Choose and extract exports

| Export | Use and contents |
| --- | --- |
| Project backup JSON | Resume editable geometry, labels, predictions and history; select the same original images before importing |
| Human report ZIP | Read/share compact HTML/CSV, occupancy, recovery frame IDs, optional CCRT, maps and project backup |
| Machine analysis ZIP | Frame-level states/geometry, raw probabilities, model versions, corrections and timing; no source pixels |
| Training dataset ZIP | Original analysis-window images, manual targets, shared boundaries, hashes and comparison plan |
| Dataset ZIP with **Export as test set** checked | Held-out images/targets/geometry; omits plans and editable history |

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

This stage is required for **Model B**. State labels cannot supervise anatomical keypoints. The creator's [SLEAP datasets](https://github.com/talmolab/sleap-datasets) include a native courtship example with 100 labeled frames with multiple individuals:

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

Native label-record indices distinguish repeated annotations of the same video/frame/instance. The public courtship data is anatomical pretraining, not CCRT truth or proof of overhead-well transfer. Fine-tune on suitable overhead anatomical labels with:

```bash
python tools/sleap_pose.py train --labels /path/to/overhead-labels.slp \
  --parent "$DATA_DIR/pose-base" --output "$DATA_DIR/pose-overhead" \
  --epochs 25 --batch-size 4 --device cuda
```

Parent hash and skeleton must match. If you trained `pose-overhead`, use that directory instead of `pose-base` in the comparison commands below. Use the same pose parent across CCRT selections. Model B trains state and occupancy forests separately per selected CCRT subset; the external pose parent stays fixed, preserving the CCRT annotation budgets.

### Inspect pose quality before training all forests

New pose training reserves whole videos when available. With the public single-video example, it holds out a contiguous labeled time block with a raw-frame gap; the same animals may recur, so this does not demonstrate cross-animal generalization. Equal foreground/background heatmap loss prevents mostly blank pixels from dominating. Seeded rotations move images and keypoints together without guessing left/right swaps. Best checkpoints are selected using held-out PCK@0.05 (pixel error breaks ties), with patience 5.

`pose-validation/metrics.json` reports PCK, localization error, peak coverage and per-node results. `pose-validation/overlays.jpg` shows predicted peaks and human keypoints. A low heatmap loss alone is not evidence of accurate anatomy.

```bash
python tools/pose_diagnostics.py --pose-model "$DATA_DIR/pose-base" \
  --labels "$DATA_DIR/public-pose/drosophila-melanogaster-courtship/courtship_labels.slp" \
  --output "$DATA_DIR/pose-source-check"
python tools/pose_diagnostics.py --pose-model "$DATA_DIR/pose-base" \
  --dataset "$DATA_DIR/train-grid" --output "$DATA_DIR/pose-ccrt-check"
```

Open each `report.html`. The first command uses stored held-out source identities when available; `--all-labels` explicitly includes potentially seen source data or a separately supplied anatomical-label dataset. The CCRT preview samples Coma/Awake/Empty separately and has **no anatomical accuracy score**: state labels are not keypoints. Inspect folded/hidden legs and belly-up flies. Add representative CCRT anatomical labels in SLEAP if transfer is poor, fine-tune, then repeat the checks before launching all forests. New behavior models abstain on absent/collapsed predicted poses; this screen cannot prove anatomical correctness.

An old pose checkpoint trained on every source example cannot provide an unseen validation subset of that same source. For a new validated baseline retrain from scratch, or supply genuinely untouched anatomical labels. Fine-tuning rejects a proposed holdout already seen by its parent. Existing pose weights remain loadable for inference and diagnostics.

## 6. Create the shared training and validation plan

Keep the app’s exported plan as a provenance record. Create a **new plan** that reserves whole validation individuals before drawing training selections. Existing annotations need no changes. The command inherits your seed and poor/rich Well IDs from the exported plan:

```bash
python tools/model_suite.py plan \
  --train "$DATA_DIR/train-grid" \
  --source-plan "$DATA_DIR/train-grid/comparison-plan.json" \
  --validation-fraction 0.2 --output "$DATA_DIR/validation-plan.json"
```

Alternatively omit `--source-plan` and supply `--seed 2026 --poor-well W12 --rich-well W45` with your chosen IDs. With several grids, ambiguous IDs require `manifest-hash-prefix::well-uid`, available in a saved plan.

For roughly 60 occupied wells, reserve about 12 individuals for validation. The poor/rich training individuals are protected from this holdout. No frame of a validation individual enters a training loss, including temporal context for a training well. All eligible annotations of each selected training well stay together.

| Selection | Runs per architecture | Purpose |
| --- | ---: | --- |
| All eligible **training-pool** wells, including Empty | 1 | Largest available training budget |
| 25 random single individuals | 1 | About half of the remaining pool |
| 10 random single individuals | 2 | Medium budget and selection variability |
| 5 random single individuals | 2 | Small budget and selection variability |
| 2 random single individuals | 2 | Extreme scarcity and selection variability |
| Selected poor-pose individual | 1 | Manually chosen low diversity |
| Selected rich-pose individual | 1 | Manually chosen high diversity |

That is **at most 10 selections per architecture / 30 training runs** for three approaches. Oversized budgets are skipped; unspecified poor/rich selections are omitted. Random selections contain occupied single-individual wells; Empty wells remain in the full training pool. Repetitions may overlap. Every architecture uses exactly the same saved selections and validation pool. Existing 15/40-selection plans remain readable and unchanged; regenerate a plan to use the smaller protocol.

“Full” means all remaining training individuals after validation is reserved, not training on validation animals. The validation annotation budget is **additional to every training budget**, including the one-well comparisons; report it. This answers how much data the model needs to fit, with a common development budget. One-well training does not provide its own independent validation animals.

Do not randomly divide frames: neighbouring frames of the same fly would leak its appearance and temporal context. The default holds out whole individuals within the training grid; shared lighting/camera background means it is less demanding than a new experiment. With enough training grids use `--validation-unit grid`, or explicitly choose validation wells with `--validation-wells W3 W8`. Leave the independent test grid outside `--train`.

For a later robustness study, regenerate a plan for each `--validation-folds 5 --validation-fold 0` through `4`, then compare each plan into a new directory. Folds hold out whole individuals, never random frames; poor/rich individuals stay in training. This multiplies compute by five and is optional after a fixed-holdout pilot. Test grids never choose the best fold. Prefer additional independent experiments over treating thousands of correlated frames as thousands of independent examples.

## 7. Train the three approaches

Approaches are `resnet18` (appearance), `sleap-rf` (anatomy plus separate occupancy forest), and `deepethogram` (optical flow, two-stream visual features and TGM sequence classification).

Start with three selections: **9 training runs** across the three approaches. The full new protocol has **30 runs** (10 selections × 3 approaches). Inspect the saved plan and printed `suite_start.training_runs` before a long job.

```bash
python tools/model_suite.py compare \
  --train "$DATA_DIR/train-grid" --plan "$DATA_DIR/validation-plan.json" \
  --test "$DATA_DIR/test-grid-1" --pose-model "$DATA_DIR/pose-base" \
  --runs all-annotated-wells poor-poses rich-poses \
  --epochs 5 --flow-epochs 3 --sequence-epochs 5 \
  --batch-size 4 --image-size 128 --window 5 \
  --policy auto --false-awake-cost 2 --patience 3 \
  --device cuda --output "$DATA_DIR/comparison-pilot"
```

For the **full training suite**, run this command without a `--runs` filter, using the same saved plan and a fresh output directory:

```bash
python tools/model_suite.py compare \
  --train "$DATA_DIR/train-grid" --plan "$DATA_DIR/validation-plan.json" \
  --test "$DATA_DIR/test-grid-1" --pose-model "$DATA_DIR/pose-base" \
  --epochs 5 --flow-epochs 3 --sequence-epochs 5 \
  --batch-size 4 --resnet-batch-size 32 --pose-batch-size 16 --temporal-batch-size 8 \
  --image-size 128 --window 5 \
  --policy auto --false-awake-cost 2 --patience 3 \
  --device cuda --output "$DATA_DIR/comparison-full"
```

This expands the selections, not the training duration per selection. The epoch values are examples, not established convergence settings. Choose epochs and batch sizes on training/validation data, then keep them fixed across the matched runs. Changing batch size changes optimization, so use the same chosen overrides for both pilot and full comparisons when comparing their scores. Use the validation plan from step 6. Old plans without validation are rejected by default. `--allow-no-validation` is an explicit legacy/software-smoke opt-out, with no independent checkpoint selection; it is unsuitable for a reported training comparison. Use `--architectures resnet18 sleap-rf` for independent images, or choose any one approach. Model B is the only approach needing `--pose-model`.

### Live progress, loss logs and GPU throughput

Progress starts with dataset loading and image-hash verification, then shows cache preparation, architecture/selection, phase, epoch, batch counts, loss, samples/second and an epoch ETA. Messages print about every **5 seconds** (`--log-every 2` makes them more frequent), at phase/epoch boundaries, and after the first batch. A single slow operation can exceed that interval; it is not a background heartbeat. Random-forest fitting uses CPU and reports its start/completion rather than neural-network epochs.

After dataset/plan validation, the output directory contains:

| File | Contents |
| --- | --- |
| `progress.jsonl` | Incrementally flushed progress events, selected settings, device/GPU, artifact paths and failure/interruption reasons. |
| `training.csv` | Every neural batch loss plus epoch means, architecture/selection/phase, counts, throughput, input wait and allocated/reserved GPU MiB. Buffered rows flush at progress messages and epoch completion. |
| `loss.png` | Training and validation curves, refreshed after each completed epoch, with separate panels for appearance, optical flow, two-stream and TGM losses. |

`elapsed_s` and `input_wait_s` in training rows describe the current epoch; ETA estimates its remaining time. Input wait measures time waiting for prepared CPU batches. CUDA work is asynchronous, so these are throughput diagnostics, not a GPU kernel profiler. Phase losses have different meanings/scales. There is no SLEAP/RF gradient-loss curve: its forests are fitted on fixed pose features. These logs **do not use held-out test data as validation loss**.

In a second terminal, inspect progress or regenerate an exportable plot:

```bash
tail -f "$DATA_DIR/comparison-full/progress.jsonl"
python tools/plot_training.py \
  --input "$DATA_DIR/comparison-full/training.csv" \
  --output "$DATA_DIR/comparison-full/training-loss.svg"
```

The plot command needs at least one completed epoch; batch losses are readable in the CSV immediately. To also retain messages from initial verification and full Python tracebacks on Linux, use `set -o pipefail` and append `2>&1 | tee "$DATA_DIR/comparison-full-console.log"` to the compare command. The Python progress records themselves flush without requiring `python -u`.

The old trainer repeatedly decoded full-grid images for randomly ordered single-well crops and only printed at epoch completion. A long silence after the ImageNet download did not establish that it was stuck. The new default prepares a **reusable, memory-mapped uint8 crop cache**, decoding each required source frame once, then reading the same masked crops across epochs and selections. Selected fitting and validation wells and their causal context enter this label-free cache. Validation pixels never enter gradient updates; tests never fit it. Hash verification still runs on every command. Preparation is CPU/disk work, so low GPU usage during that phase is expected.

The cache defaults to `$DATA_DIR/.flyscope-training-cache` when output is directly inside `$DATA_DIR`. Put `--cache-dir /path/on/local-ssd/flyscope-crops` on **both** commands to relocate/reuse it. At 128×128 RGB, 5,400 frames × 61 wells need about **15.1 GiB** plus small occupancy features/headers. Preparation prints its actual size estimate and checks free disk space. Temporary DeepEthogram feature files need another ~1.3 GiB for that many frame/well pairs and are removed after each temporal run. Completed caches can be deleted between commands; original exports and annotations are not changed. `--no-cache` disables disk caching if storage is constrained, at the cost of slower epochs. A cancelled cache build is rebuilt; completed model checkpoints and logs remain in their output directories. There is no optimizer/partial-epoch resume: rerun with a new output directory, or evaluate completed `model.json` checkpoints separately.

Bounded CPU prefetch overlaps input preparation with training (`--prefetch-batches 2`, `0` to disable), with pinned uint8 CUDA transfers for ResNet/DeepEthogram training. `--cache-workers 2` controls initial frame preparation, not model training. Keep the cache on a local SSD when possible. Temporal feature extraction is batched and fixed features remain on disk rather than accumulating in GPU memory. GPU memory usage is not a utilization target: a small ResNet with batch size 4 can legitimately occupy much less than 12 GB. Increase batch size gradually while comparing samples/second; SLEAP/RF fitting, image verification and cache preparation will still have low GPU utilization. Restart with the updated code to obtain these features; an already running process cannot acquire new logging or caching.

Defaults fine-tune ImageNet backbones; `--initialization scratch` starts fresh weights. Each run restarts independently from the common base, never the preceding subset's CCRT weights. The temporal model uses current/past analysis frames from selected individuals; no setup/future frames or unselected wells enter targets. Missing interior context/geometry is an explicit error.

`--device cuda` requires the GPU; `cpu` is the tested fallback and `auto` chooses CUDA when available.

### Validation, balancing and automatic recovery settings

Epoch counts are **maximums**. Every supervised neural stage evaluates reserved validation individuals after each epoch, monitors balanced negative log likelihood, stops after `--patience 3` epochs without improvement and restores its best checkpoint. Optical flow selects its best held-out photometric loss, which is not a behavior accuracy score. Forests use bounded depth/leaf size and report held-out class metrics. `training.csv` and `loss.png` show both training and validation. `model.json` records supports, split fingerprint and selected epochs.

Default `--balance well-class` gives equal mass to represented classes and, within each class, to contributing individuals. `--false-awake-cost 2` adds an explicit Coma/Empty cost. All selected targets are still visited; weights do not invent missing classes. `--balance class` and `none` are available for controlled ablations. AdamW weight decay and regularized forests reduce memorization risk; they do not establish generalization. No belly-up appearance penalty is guessed from state labels.

**Leave `--policy auto` enabled.** After restoring the best checkpoint, training searches confidence thresholds and sustained-Awake durations using the same reserved validation pool, with no test predictions. It requires observed Coma before irreversible Awake confirmation. Candidates must have at most 5% false-early events among informative validation wells and detect at least 80% of observed recoveries. Among feasible candidates it prioritizes false-early events, recovery coverage, onset delay and then confirmation latency. The full 50-candidate audit is saved as `recovery-policy-search.json` in each checkpoint directory.

These are empirical development criteria, not statistical guarantees, especially with a dozen animals. With fewer than two Coma-containing validation individuals, no observed recovery, inconsistent recovery labels, or no passing candidate, **automatic recovery confirmation is disabled**. Raw scores remain available; confident Coma/Empty can still be shown. Awake decisions remain Unsure for manual review rather than inventing a Coma label. Scores are not presumed calibrated: high confidence on wrong examples cannot always be fixed by threshold selection.

The app uses the checkpoint’s saved settings by default and explains its validation status. **Advanced: override recovery settings** is available after unchecking saved settings. The CLI equivalent is `--policy manual --confidence-threshold 0.8 --awake-frames 3`; manual overrides are recorded and do not constitute validated settings. `--allow-initial-awake` permits left-censored automatic confirmations for explicitly chosen protocols. By default, initially Awake predictions without an observed Coma baseline stay Unsure. Independent-image classification does not require a temporal baseline.

When acquisition intervals are known, saved confirmation duration is converted to a suitable observation count at inference. Three observations at one-second spacing span two seconds. Model C’s temporal window still represents a specific sampling interval: resample deliberately or retrain when that changes. Once confirmed, sequence decisions stay Awake; retrospective onset and later confirmation frames are both saved, together with unchanged raw probabilities. Manual corrections remain separate.

Offline sequence review also infers an uncertain gap bounded by matching confident Coma/Coma (or Awake/Awake) decisions, using the single irreversible-recovery assumption. It does not bridge a Coma/Awake transition, missing frames, changed analysis sessions or unusable geometry/poses. Inferred states have no invented confidence: `temporal_inference` records their bounding frames, and raw scores/manual labels remain unchanged. New models save `infer_matching_gaps: true`; advanced manual settings can disable it. Existing model policies retain their historical default.

The shared validation cohort is reused for checkpoint and policy selection: it is **development data**, not an unbiased performance estimate. Freeze the protocol before a fresh final test grid. A grid whose errors have already guided changes should be reported as a development benchmark, even if exported with the test checkbox. The supplied first run had no Empty examples in training or testing; annotate Empty wells before claiming Empty detection quality.

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

This trains/reloads nine checkpoints (three approaches × full/poor/rich), executes all temporal stages and scores an independent partly annotated synthetic grid. CI uses the same requirements with CPU wheels for its GPU-less runner. Optional Linux browser checks use Node and Playwright as development tools only; the application and model commands do not require them. After the smoke command above, start the app in a second terminal with `FLYSCOPE_SUITE_MODELS_DIR="$DATA_DIR/suite-smoke/results" python app.py`, then run:

```bash
npm install --prefix /tmp/flyscope-browser playwright@1.63.0
node /tmp/flyscope-browser/node_modules/playwright/cli.js install --with-deps chromium
export NODE_PATH=/tmp/flyscope-browser/node_modules
export PYTHON="$(python -c 'import sys; print(sys.executable)')"
export CHROMIUM_PATH="$(node -p 'require("playwright").chromium.executablePath()')"
export SUITE_SMOKE_DIR="$DATA_DIR/suite-smoke"
export POSE_SMOKE_MODEL="$DATA_DIR/pose-smoke"
export RUN_MODEL_TRAINING=1
for check in workflow detection long-sequence studies experiments model-suite; do
  node "tests/$check-browser.cjs"
done
```

For the audit of the first 45-checkpoint experiment, see [TRAINING_REVIEW.md](TRAINING_REVIEW.md). For the source-code and artifact map, see [MAINTAINING.md](MAINTAINING.md).

Repository cleanup: the working directory is now the repository root, not the former versioned app folder. The `tools/...` command names and exported dataset/model formats remain supported. Restart Flask after updating. Existing environments or user datasets left in the old local folder are not migrated by Git; retain your data and activate/recreate your environment deliberately. See [MAINTAINING.md](MAINTAINING.md) for the actual package layout.
