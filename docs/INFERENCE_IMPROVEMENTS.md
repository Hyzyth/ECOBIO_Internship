# Resumed efficiency and model inspection work

The initial audit found a clean `work` branch at `7cd9b9c`. Existing work already
provided polygon masking, reusable crop caches, bounded prefetch, native SLEAP
batches, whole-individual validation, immutable checkpoints, recovery confirmation,
an anatomical annotation page and corrected CCRT reports. Those implementations
and the original weights were retained. `CHAT5.txt` confirmed that its proposed
efficiency changes and DeepEthogram comparisons had not been implemented.

## Changes available now

- The app stops sending a well for inference after **confirmed** Awake, using
  the selected confidence, consecutive-frame and Coma-baseline settings. Onset
  and confirmation remain separate. Later records say `inference_skipped`, have
  no new probabilities, and preserve manual corrections. A new session analyzes
  again from the analysis start. Disable the checkbox to collect every raw output.
- Suite inference chooses CUDA when available, with CPU fallback. Set
  `FLYSCOPE_INFERENCE_DEVICE=cpu` or `cuda` to force a device. Portable random
  forests traverse batches together and compile their arrays once per worker.
- App project backups include `inference_profile`: browser round trip, adapter
  and worker time, image read/decode, crop preparation and model execution.
  First requests include worker/model startup. Offline evaluation records phase
  totals in `inference`. Compare warm requests separately from cold startup.
- Optional contrast isolation uses the same versioned preprocessing in training,
  cached crops and inference. The app can preview the selected model's actual
  input or the proposed isolation. Original images and legacy CNN preprocessing
  remain intact. Occupancy features use the original well. Failed/ambiguous
  segmentation retains the well; it never invents an Empty label.
- Optional thinning removes repetitive **fitting targets**, measuring local pixel
  changes against the last retained crop, including legs without body translation.
  It compensates uniform brightness drift, retains spaced static anchors, both
  sides of manual transitions and a dense recovery window. It does not delete
  images, change the saved comparison plan, thin validation/test truth, or join
  nonadjacent frames into a temporal clip. `training-sampling.json` records every
  selection and removal; checkpoints bind this sampling and their preprocessing.
- Pose dots have a small adjustable radius in screen pixels. Visible points can
  be connected and numbered independently. A new 19-node project adds one
  femur–tibia joint per leg. Parent transfer copies existing confidence maps by
  node name and initializes added maps. New joints require human labels; hidden
  anatomy remains occluded. Retrain the pose model and behavior forests together.
- `inspect_model.py` writes a native-resolution report, exact inputs, ResNet
  class Grad-CAM and four strong channels from each residual layer. DeepEthogram
  inspections show causal context and optical flow with a magnitude legend.
- DeepEthogram has explicit `full`, `two-stream` and `resnet-temporal` variants.
  The last freezes a matched fitted ResNet's image encoder, uses no motion, and
  fits only its TGM. Parent pixels, fitting IDs, datasets and validation wells
  must match. `component_validation` and `temporal_validation_regressed` expose
  stage degradation. `auto` fits both two-stream/TGM stages, then chooses the
  lower validation balanced NLL (ties prefer two-stream) before selecting the
  recovery policy. Existing checkpoints retain their full-pipeline semantics.

## Commands for the next real experiment

Use the existing immutable validation plan and **new output directories**. Apply
one change at a time on training/validation before combining them. The paths below
are placeholders for the user's exports; they are not guessed dataset choices.

```bash
# New preprocessing and/or target sampling; omit either flag for its ablation.
python tools/model_suite.py compare \
  --train TRAIN --test TEST --plan VALIDATION_PLAN \
  --pose-model POSE_PARENT --output NEW_RUN \
  --foreground fly --thin-static \
  --motion-delta 8 --motion-pixels 3 \
  --static-keep-every 30 --transition-radius 5

# Complete raw-frame scoring remains the CLI default. This measures saved work:
python tools/model_suite.py evaluate --models NEW_RUN --test TEST \
  --output NEW_STOPPED_EVALUATION --stop-after-awake

# Keep preparation resolution/preprocessing matched to behavior training.
python tools/pose_labels.py --dataset TRAIN --output NEW_POSE_LABELS \
  --skeleton POSE_PARENT/pose.json --articulated-legs --size 128
# Annotate visible points on /pose, save/export, then:
python tools/sleap_pose.py train --labels LABELED/pose-labels.json \
  --parent POSE_PARENT --extend-skeleton --output NEW_POSE_PARENT

# Inspect training/validation examples. No weights are fitted by this command.
python tools/inspect_model.py --dataset TRAIN \
  --model NEW_RUN/resnet18/all-annotated-wells/model.json \
  --samples 12 --output NEW_INSPECTION

# Two-stream without TGM, on the same selection/validation cohort:
python tools/model_suite.py compare --train TRAIN --test TEST \
  --plan VALIDATION_PLAN --architectures deepethogram \
  --runs all-annotated-wells --deepethogram-variant two-stream --output NEW_TWO_STREAM

# Same fitted ResNet encoder, with temporal learning isolated from motion:
python tools/model_suite.py compare --train TRAIN --test TEST \
  --plan VALIDATION_PLAN --architectures deepethogram \
  --runs all-annotated-wells --deepethogram-variant resnet-temporal \
  --appearance-model BASELINE/resnet18/all-annotated-wells/model.json \
  --image-size 128 --output NEW_RESNET_TEMPORAL
```

Use `--foreground fly` when preparing pose labels **only if** the behavior model
will use that preprocessing. Parent transfer keeps the parent's neural input
resolution. A higher-resolution pose parent must be trained at that resolution.

## Resolution and interpretation

| Family | Input contract | Validation comparison from CHAT5 |
| --- | --- | --- |
| Original CNNs | Fixed 200×200, grayscale contrast/Otsu | Retain supplied input contract |
| ResNet-18 | RGB, checkpoint's saved size; default 128 | 128 versus 224 |
| SLEAP/RF | Well crop → centered input → `pose.json` neural size | 128 versus 256, with visible limb labels |
| DeepEthogram | Causal RGB clips, saved size/window; size ≥32 and divisible by 8 | 128 versus 192; also compare time spans |

A window of five images spans four sampling intervals. Original frame IDs,
analysis start, camera-corrected geometry and the same individual's full causal
context remain binding. Upsampling cannot recover a missing leg. The inspection
report measures source crop dimensions and resize ratios rather than assuming
that increasing network size improves accuracy. Grad-CAM is a diagnostic, not a
biological explanation; independently scaled feature maps are not probabilities.
Contrast cannot guarantee removal of dust touching the fly or preservation of
every faint appendage. Review rim/leg cases before adopting that preprocessing.

## Verification and remaining experimental work

Software checks exercise original behavior and new options, actual training and
reloads, browser request stopping, pose annotation/export/resume, isolated-cache
equivalence, manual transition retention, causal training/inference parity,
component variants and visualization outputs. All experiment fixtures are
synthetic or existing software smoke models, not evidence of biological accuracy.

Verified on 2026-10-23 in the published Python 3.12.12 environment:

| Check | Result |
| --- | --- |
| Full Python discovery and separate suite contracts | 83 + 7 passed; subsequent focused checks also passed after final changes |
| JavaScript unit checks | 42 passed |
| Existing real training/reload workflow | All nine checkpoints; cached/uncached weights agree |
| New preprocessing/thinning options | All three families trained, evaluated and reloaded |
| Temporal comparisons | Two-stream, frozen matched ResNet/TGM and automatic stage selection trained/evaluated/reloaded |
| Articulated pose | 19 confidence maps trained and reloaded on a software fixture |
| Browser checks | Registered inference for all families; request stopping; pose save/export/resume and spatial export passed |
| Inspection outputs | 21 ResNet input/CAM/layer images and 10 causal-context/flow images generated and inspected |
| Final source check | `git diff --check` clean |

The CPU forest microbenchmark (128 samples, 200 trees, 91 features, five repeats)
measured median 1.632 s before versus 0.123 s after, with identical probabilities.
This is about 13.3× for the forest calculation alone, not the pose pipeline or
whole application. No CUDA device was available in this session.

The user clarified the target on 2026-10-24: at least **95% of individuals with an
exact annotated recovery must have estimated onset within ±5 original frames,
and every remaining recovery within ±8 frames (±7 preferred)**. This is a bound
on individual errors, not an MAE threshold. Missing detections fail. Reports now
include `individual_timing_target` with counts and a pass/fail result; no exact
human events means the target cannot be assessed. Never-recovered and
interval-censored individuals remain separately reported.

The two uploaded historical runs precede these software changes. Their result
archives do not contain complete training/test image exports or the neural
weights, including `pose.pt`. Those inputs and fresh anatomical joint annotations
are still needed for a new training/accuracy experiment. Existing reports count
missed recoveries as failures and report coverage, early events and conditional
MAE. No new 95% claim is made. Previously reviewed Ts1 must not become a new
untouched test.

Reports now also include ±7/±8-frame tolerance results, the maximum paired
absolute error and whether all exact annotated events were detected. This keeps
an average error separate from a bound on every individual.
