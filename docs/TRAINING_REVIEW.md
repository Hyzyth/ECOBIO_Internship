# First comparison: findings and corrective protocol

This audit covers the supplied 45 checkpoints: three architectures and fifteen
training selections, scored against 61 individuals on a different grid. It
examines saved results, not a new biological retraining experiment. The separate
`final_eval` archive records an interrupted evaluation after about eleven seconds;
the completed metrics below come from `comparison_full`.

## What the results show

The full training grid contains 24,957 Coma and 42,021 Awake targets (37.3% / 62.7%).
The test grid has 33,321 Coma and 46,772 Awake targets (41.6% / 58.4%). Neither has
Empty examples. Always predicting Awake scores **58.4% accuracy** on this test.

For the **full-training-grid selection**, rather than cherry-picking different
subsets for different architectures:

| Approach | Raw frame accuracy | Accuracy after recovery rule | Raw Coma recall | Initially decided Awake |
| --- | ---: | ---: | ---: | ---: |
| ResNet-18 | 81.6% | 76.6% | 58.5% | 15 / 61 wells |
| DeepEthogram | 71.7% | 65.5% | 33.5% | 27 / 61 wells |
| SLEAP + random forest | 56.0% | 58.4% | 5.1% | 53 / 61 wells |

The sequence rule worsened the two neural classifiers by carrying early mistakes
forward. The RF decision score is essentially the Always-Awake baseline, not
useful recovery measurement. One never-recovering individual received a false
recovery in each full-grid run. This is not a false recovery on an Empty well:
there were no Empty ground-truth examples.

Reported recovery MAE was about 127 / 303 / 484 frames, respectively, but only
**22 / 9 / 3 well pairs** qualified. Left-censored/uncertain cases are excluded;
these averages therefore cannot fairly rank models without their denominators.
New comparison tables show those counts, initial-Awake counts, class recalls and
baselines alongside accuracy. Raw probabilities and decisions remain separate.

### Imbalance contributes, but is not the whole explanation

Awake predominates, but the ratio is only about 1.68:1. The previous loss already
applied a Coma/Empty cost of 2. The median confidence of **wrong Awake predictions
on true Coma frames** was approximately 0.999999 for ResNet, 1.0 for DeepEthogram
and 0.97 for the forest. Raising the confidence threshold alone cannot repair
wrong, overconfident features. Calibration metrics also show substantial error.

The selected poor-pose individual has 451 Coma / 647 Awake targets; the rich-pose
individual has 201 Coma / 897 Awake targets. Both classes are present, so these
are not accidental single-state training runs. New weighting balances represented
classes and contributing individuals, with explicit false-Awake cost, without
replacing unannotated wells with negative labels.

### Very low training loss is not a validation result

There was no held-out checkpoint selection in the original comparison. Full-grid
ResNet loss fell from about 0.0177 to 0.000864; temporal TGM loss fell from 0.118
to 0.00207. Single-individual neural losses often collapsed to near zero by the
second epoch. This is consistent with memorization risk, but training loss alone
cannot prove overfitting. Thousands of consecutive frames of one animal do not
provide thousands of independent animals.

The corrected protocol reserves complete validation individuals **before** any
training budget is drawn. It shares this pool across all selections and models,
logs validation losses/class recall, stops early and restores the best checkpoint.
A contiguous temporal clip stays entirely with its well. Random frame splits
would leak appearance and temporal neighbours. Optional grouped folds and
whole-grid validation are documented in [WORKFLOW.md](WORKFLOW.md).

### Pose transfer must be checked, not assumed

The supplied pose metadata describes 298 instances from 100 labeled frames in
one public courtship video, with 13 anatomical nodes. Some source SLP labeled-frame
records repeat the same video/frame and local instance index, so these are not
298 independent animals or images. New provenance records distinguish them. Its final legacy full-image
heatmap MSE is about 0.001543. A continuous-Gaussian approximation gives an
all-zero-map MSE of about 0.001726 at 128 pixels with sigma 3. That proximity is
a warning that empty background pixels can dominate the loss. **It does not prove
collapse**: the actual pose weights and image/keypoint overlays were not supplied.

Public upright flies are not guaranteed to cover belly-up, folded-leg or occluded
CCRT poses. Behavior state labels cannot supply missing anatomical keypoints.
The updated pose trainer balances foreground/background heatmap loss, rotates
images and keypoints together, reserves videos or a gapped temporal block, selects
by held-out PCK and writes per-node metrics plus overlays. The standalone pose
diagnostics tool also previews Coma and Awake CCRT crops separately. Inspect these
before training all RFs; add target-domain anatomical labels when transfer is poor.
A low-confidence/collapsed-pose screen makes new forests abstain, but cannot
certify that a high-confidence pose is anatomically correct.

### Extra model complexity does not guarantee better performance

ResNet has strong generic visual pretraining. A forest can only use the anatomical
features it receives. A temporal network needs useful visual/motion features and
sufficient independent examples; a five-frame window at one frame/second is
short, while recovered flies may remain stationary. Low optical-flow photometric
loss measures reconstruction, not behavior recognition. Validate upstream pose
and visual features before increasing model complexity or training duration.

No belly-up penalty has been added: the images contain both legitimate recovered
belly-up poses and comatose poses, and no orientation annotations justify such a
rule. Penalizing one could encode the wrong biology.

## Automatic recovery settings and the smaller study

New comparisons select confidence and sustained-Awake evidence on validation
individuals, after restoring the checkpoint. The app loads those saved settings
by default. Automatic confirmation requires observed Coma first and candidates
meeting explicit false-early/coverage criteria. If validation cannot support a
reliable setting, Awake stays Unsure for review. This avoids turning an initially
Awake prediction into a falsely measured recovery. The full search and its status
are saved; manual override remains available. Offline review can infer uncertain
gaps bounded by matching confident states under the irreversible-recovery
assumption. This inference is explicitly marked, has no fabricated confidence,
and leaves transition gaps, raw scores and manual labels intact.

New plans contain at most **ten selections per architecture**: full training pool,
1 × 25, 2 × 10, 2 × 5, 2 × 2 individuals, and the poor/rich-pose individuals.
The common validation cohort is an additional annotation budget for every run.
Old saved plans/checkpoints are immutable and readable; generate a new plan to
use this protocol. Existing annotations need not be redone.

See the workflow for exact plan, training, diagnostics and evaluation commands.
Do the three-selection pilot first and inspect class recall, validation curves,
pose overlays, raw versus decided performance and decided coverage. Reducing
coverage is not an accuracy improvement unless the abstentions are reported.
The already-inspected test grid is now a development benchmark; freeze choices
before using a fresh grid for a final blind comparison. Empty quality cannot be
claimed until both fitting and independent evaluation include Empty examples.

## Reproduce a compact result audit

```bash
python tools/audit_comparison.py --results /path/to/comparison \
  --pose-json /path/to/pose-base/pose.json \
  --training-log /path/to/comparison/training.csv \
  --output /path/to/new-audit
```

The directory and ZIP contain compact per-model metrics, a comparison figure,
HTML report, epoch-only logs and pose-training metadata. Original pixels, weights
and repeated per-frame records are not copied. Omit optional inputs if absent.
The audit can inspect older completed comparison outputs; it does not tune models
or modify original results.
