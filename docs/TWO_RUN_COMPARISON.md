# Historical run comparison against the clarified recovery target

The 27 supplied files contain 75 frozen evaluations: 45 in the first run and 30
in the second. Every prediction JSON was SHA-256 matched against its supplied
archive member. The second batch also matches retained archives byte for byte.
Existing audits were reused; recovery events were rechecked using one scorer.
No predictions, weights, thresholds or recovery policies were changed. Neither
historical run tests the software changes implemented in this conversation.

## Acceptance criterion

At least 95% of exact manually annotated recovery onsets must be estimated
within ±5 original frames, and every remaining onset within ±8 frames (±7
preferred). For these 60 exact recoveries this means **at least 57/60 within ±5
and 60/60 within ±8**. Missing detections fail. MAE is a diagnostic average,
not this acceptance criterion. One never-recovered fly is reported separately;
false recovery of that fly is not hidden in the timing denominator.

**None of the 75 recorded evaluations meets the criterion.**

## Full fitting selections

| Run | Model | Raw frame accuracy | Within ±5 | Within ±8 | Missed | Early | MAE (detected) | Max error (detected) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first | ResNet-18 | 81.6% | 12/60 | 12/60 | 0 | 52 | 286.2 | 951 |
| first | DeepEthogram | 71.7% | 3/60 | 3/60 | 0 | 58 | 428.2 | 1024 |
| first | SLEAP/RF | 56.0% | 0/60 | 0/60 | 0 | 60 | 528.9 | 1036 |
| second | ResNet-18 | 87.6% | 29/60 | 31/60 | 2 | 18 | 85.7 | 643 |
| second | DeepEthogram | 77.9% | 0/60 | 0/60 | 60 | 0 | undefined | undefined |
| second | SLEAP/RF | 58.9% | 0/60 | 0/60 | 60 | 0 | undefined | undefined |

MAE and maximum error use available onset estimates; undefined is not zero.
The first-run MAEs above use all human-measurable predicted events, correcting
the older exclusion of initially Awake or Unsure-bracketed errors. They should
not be compared with the older, selectively reported MAEs of 127/303/484 frames.

The second-run full-data ResNet is more useful than its first-run counterpart
on the recorded evaluations, but remains far from the target: median error is
4.5 frames, maximum paired error is 643 frames, and two recoveries are missed.
Its raw frame accuracy of 87.6% is not 87.6% recovery timing success.

The MAE of **85.69 frames (approximately 86)** averages absolute errors over
58 detected recoveries. These same results contain 29 errors ≤5, another 2
errors between 5 and 8, 27 errors >8, and 2 missed recoveries. Large errors in
that tail raise the average despite nearly half the individuals being close.
A missed event has no finite onset error, so the two misses are excluded from
MAE and explicitly fail the 60-event tolerance counts. Thus 29/60 within ±5 and
an MAE of 86 are consistent statistics, not conflicting claims.

Full-data DeepEthogram and SLEAP/RF both failed their second-run validation
policy search (`no_safe_candidate`), disabling automatic Awake confirmation.
Their 0/60 timing result is abstention, not evidence that all flies remained
comatose or that raw neural classification became uniformly worse. The first
run falsely announced recovery for the never-recovered fly in all three full
models; the second-run full models did not.

## Comparability and annotation provenance

Both exports align on 80,093 well/frame/filename keys, with no
unmatched keys. Nevertheless **841 saved manual labels differ across
16 wells**: 29 coma → awake, 812 awake → coma.
The test IDs are W26 and Ts1, and their dataset fingerprints differ. Original
pixel exports are needed to verify physical identity; matching names alone
are not independent experiment evidence. The annotation-change log is retained
in `changed-manual-labels.json`. The user describes these as train/test labels;
that does not establish why individual Coma/Awake values changed. No intentional
correction is assumed, and both original versions are retained.

The full first run fitted 61 wells / 66,978 targets with no reserved validation
cohort. The second fitted 49 wells / 53,802 targets and reserved 12 other wells
/ 13,176 targets for validation. It also changed weighting, checkpoint and
recovery-policy selection. The timing comparison above uses each run's own
recorded truth and therefore does not isolate one causal improvement.

## What the training screenshot means

The saved metadata confirm a large fitting/validation separation in several
selections. Extra epochs are not the immediate remedy. For the full selections:

| Model | Stage | Selected epoch | Selected validation loss | Last validation loss |
| --- | --- | --- | --- | --- |
| ResNet-18 | appearance | 3 | 0.1204 | 0.7563 |
| DeepEthogram | optical_flow | 3 | 0.0105 | 0.0105 |
| DeepEthogram | two_stream | 3 | 0.6790 | 0.9844 |
| DeepEthogram | temporal_tgm | 1 | 0.9670 | 1.9517 |

Optical-flow reconstruction loss is a different objective from the three
classification losses. The two-stream and temporal curves are sequential stages
of the old complete DeepEthogram model, not independently evaluated classifiers.
The worse temporal loss motivates the explicit component comparisons now available
in the code; it does not yet establish which new variant will perform best.
Forest fitting has no neural epoch curve. Pose localization has its own
diagnostics; the supplied source-domain PCK is approximately 52.2%, with much
weaker leg-tip localization than body landmarks. CCRT pose previews lack manual
keypoints, so they cannot establish target-domain anatomical accuracy.

Readable training, validation and forest plots from the already verified audit
are provided in `assets/`, keeping the legends outside the data area.

## What these files enable next

The supplied result archives contain no neural `.pt` weights, including no
`pose.pt`, and no complete training/test image exports. The `pt1`–`pt3` zip
suffixes identify parts containing prediction JSON, not PyTorch weights.
They enable this historical analysis, but not fresh inference, true-model
Grad-CAM or retraining with those exact checkpoints.

Next use manually reviewed overhead CCRT anatomical labels and the same fixed
training/validation plan to compare: baseline inputs, optional isolation,
conservative target thinning, and image / image+motion / temporal variants.
Measure both recovery tails and missed events. Freeze choices on validation,
then use an unused experimental grid for the final target claim. Keep the
already reviewed test series as a development benchmark; do not choose another
threshold by optimizing these test predictions.

## Reproducibility and checks

`verified-prediction-sources.json` binds every result to its uploaded member.
`summary.csv` and `summary.json` hold all 75 evaluations; `recovery-wells.csv`
contains the per-individual event errors. The clarified pass/fail rule is now
included in the project's recovery evaluator and CSV reports. Thirty relevant
recovery, validation and model-contract tests passed, including a low-MAE case
that correctly fails because one individual's error exceeds eight frames.
