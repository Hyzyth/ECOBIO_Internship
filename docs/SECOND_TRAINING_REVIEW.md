# Second experiment: diagnosis and changes

This review uses the supplied 30 evaluated checkpoints (10 selections for each of three approaches), their `model.json` files, ten hash-matched forest payloads, saved per-frame predictions, pose diagnostics and training logs. It re-scores **existing predictions**: it does not claim new biological performance from the code changes. The uploaded neural/pose metadata do not include the corresponding `model.pt`/`pose.pt` files, so inference with those exact weights was not rerun.

## What the results establish

The full fitting selection contains 49 individuals and 53,802 targets: 20,676 Coma and 33,126 Awake (61.6% Awake). The shared validation cohort has 12 different individuals and 13,176 targets. The independent grid Ts1 has 61 individuals, 80,093 scored frames and 60 observed recoveries; one individual never recovers in the observed window. Neither fitting nor testing includes Empty targets. An Always Awake classifier would reach 57.4% on this test, without measuring recovery correctly.

| Full fitting selection | Raw frame accuracy | Raw Coma recall | Recoveries detected / human events | Within ±5 frames / all human events | CCRT MAE on detected events |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet-18 | 87.6% | 75.8% | 58/60 | 29/60 (48.3%) | 85.7 frames |
| DeepEthogram pipeline | 77.9% | 55.8% | 0/60 | 0/60 | Undefined |
| Pose + random forest | 58.9% | 31.4% | 0/60 | 0/60 | Undefined |

The temporal and pose full-grid models failed their **validation** policy search (`no_safe_candidate`). Automatic confirmation was therefore disabled. This is abstention requiring review, not evidence that all individuals stayed comatose. Some smaller selections did pass validation, but generalize poorly to Ts1. Do not select a new threshold from Ts1: its failures have now informed development, so a fresh grid is needed for an unbiased final evaluation.

### A reporting bias was hiding bad CCRT estimates

The old scorer required an immediately preceding *model* Coma decision to include a predicted event in MAE. An Unsure region before Awake excluded the event, even when human onset was exact. For full ResNet this gave 32.94 frames on only 18 pairs. With identical predictions, all 58 available onset estimates give **85.69 frames**; two missed events are counted as failures in tolerance success. There are 18 early estimates, 29/60 within ±5 frames, and 31/60 within ±10. This change is a reporting correction, not a degradation from retraining.

Scoring v2 uses human truth to establish the measurable population. It reports detection coverage, missed/early/false recoveries, conditional MAE and median error, and success within ±1/5/10/30/60 frames. Missing predictions count against tolerance success. A human Coma→unlabeled→Awake gap gives an interval, reported separately from exact timing. Incompletely annotated non-recovering wells are not negative controls. Frame indices in machine files are zero-based; app frame numbers are one-based. Onset and later confirmation are distinct.

Browser exports used `frame_interval_seconds`, while one training path expected `interval`. The loader now normalizes the name without changing the original manifest fingerprint. New saved evaluations retain timing metadata for reproducible rescoring. Old evaluations lacking that metadata are scored in frames; seconds are not invented.

## Why the simpler model can win

Thousands of adjacent images of one fly are not thousands of independent individuals. Small selections memorize appearance, lighting or pose. Additional pose/temporal machinery adds parameters and depends on its own input quality; it does not guarantee improvement. The full ResNet selected epoch 3 on validation (loss 0.1204), not the final epoch (0.7563). Poor-pose ResNet reaches training loss 2.8e-7 while validation stays around 1–3; rich-pose validation grows from 0.838 to 8.526. More epochs alone are not the remedy.

The existing balancing, shared whole-individual validation, best-checkpoint restoration, early stopping and false-Awake cost are retained. Whole-well folds are available; random frame folds would leak adjacent temporal context. For tiny budgets, reserve the same external validation animals for every architecture rather than carving one training animal into near-duplicate training/validation frames. Whole-grid holdouts are preferable when more grids become available.

A new explicit `--backbone frozen` ablation keeps pretrained feature CNNs fixed while training classification heads (and motion/temporal modules for Model C). It requires ImageNet initialization. The default remains `finetune`; this is an option to compare on **validation**, not an asserted fix already proven on Ts1. It reduces trainable feature capacity and compute. Choose the protocol before using a new final test.

## Pose transfer and the wall hypothesis

The public source diagnostic is not perfect overall: PCK@5% is **52.2% on 738 visible targets from 58 held-out examples**. Head/thorax are 100%, but leg tips range from 12.1% to 40.7%. A good-looking body overlay does not establish accurate legs. The CCRT preview contains 12 Coma and 12 Awake examples, without anatomical ground truth: it shows a clear transfer problem but cannot yield a numerical CCRT PCK.

Models A/C see the masked well image, including the rim **when the approved boundary includes it**, resized to the configured image size. They can learn two-dimensional context but do not measure actual contact or depth. The current pose/RF behavior features are normalized inter-keypoint distances, visibility and peak scores. They do **not** encode distance to the wall. Rim appearance can also influence the pose foreground-centering heuristic. A sideways comatose fly with extended legs is consequently a plausible confound, but the supplied results do not prove that this is the causal error. No arbitrary belly-up penalty has been added.

The practical fix is to label anatomy on actual overhead CCRT crops: Coma/Awake, center/edge, folded/extended legs, different orientations and lighting. The new Pose annotation page marks visible points and explicitly occluded nodes, saves/resumes JSON and exports labeled pixels. Preparation uses the exact current inference preprocessing; existing weights keep their historical preprocessing. Only reviewed points train. Validation holds out complete individuals, and parent-checkpoint provenance prevents reusing previously fitted individuals as independent pose validation. See the workflow commands.

Do not guess hidden anatomy. Include leg annotations where resolvable; use body landmarks alone when not. Fine-tune pose first, inspect held-out CCRT PCK and overlays, then retrain all RF selections from that common pose base with the same behavior plan. Manual Coma/Awake labels alone cannot train anatomical localization.

## Why three DeepEthogram plots and no SLEAP epoch curve?

Each Model C checkpoint has **three sequential stages**, not three independently scored variants: (1) optical-flow learning, (2) appearance + motion classification, (3) temporal TGM classification. The final test predictions use the complete pipeline. ResNet has one classification stage. Model B reuses one fixed pose neural network, then fits two random forests (occupancy and behavior); forest fitting has no neural training epochs. Pose training has its own `training.csv` and validation overlays.

Reports now use separate training and validation axes/files, with legends outside the data area. Newly fitted forests log training/validation negative log-likelihood, balanced accuracy and Coma recall. Existing checkpoints already contain validation forest metrics; plotting metadata displays those and leaves unrecorded training values absent. Missing data are not reconstructed as fake epoch curves.

## What a loss means

| Stage | Loss | Interpretation |
| --- | --- | --- |
| ResNet, two-stream, TGM | Weighted cross-entropy, mean `weight × -log(probability of human class)` | Penalizes confident wrong labels strongly. Not a percentage and not a CCRT error. |
| Optical flow | Robust RGB reconstruction (Charbonnier) + 0.01 × spatial flow smoothness | Whether adjacent images can be reconstructed using motion. A static background can give a low loss despite poor behavior classification. |
| Pose | Visible-node heatmap squared error, balancing foreground/background | Localization training signal; inspect keypoint errors/PCK and overlays as well. |
| Random forest | Weighted Gini splitting while fitting; reported balanced NLL after fitting | No differentiable epoch curve. NLL summarizes probability quality and can be large for overconfident mistakes. |

For **unweighted classification**, loss 0.03 corresponds to geometric mean true-class probability about 97.0%; 1.0 → 36.8%; 5.0 → 0.67%. It does not mean 97%, 37% or 0.67% accuracy. Uniform guesses give log(3)=1.10 for three represented states or log(2)=0.693 for two. Here fitting uses well/class weights and Coma/Empty cost 2; validation uses balanced weights with cost 1. Their scales are related but their ratio is not a pure overfitting statistic. Never compare optical-flow 0.01 to classification 0.01 as if they measured the same thing.

## Spatial exports and next experiment

CCRT dataset exports default to the union of all well boundaries and camera corrections plus 32 pixels of margin on the first exported image. One constant normalized region is used across frames. No individual-well image duplication is introduced. PNG preserves pixels; optional JPEG quality 98 reduces storage with explicit loss. Fewer pixels do not guarantee fewer bytes than an already compressed source JPEG. Exported hashes and original source hashes are retained so different crops of the same image cannot bypass train/test separation. Geometry is transformed into the exported image; project backups keep original coordinates for editable review. Egg exports remain independent and uncropped by this option.

Next priority: representative manual CCRT anatomical labels, validation on untouched individuals/grids, a small pretrained/frozen pilot, then matched full comparisons. Set an acceptable timing tolerance with the CCRT researcher before the next final grid. The software changes were checked with actual CPU training/reload and browser export round trips; they are not new biological validation.
