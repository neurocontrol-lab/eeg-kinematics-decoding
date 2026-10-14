# C1 condition audit and C2 P1 weight classification

This first Category 1 experiment follows the condition audit and classifies
object weight from P1 EEG at a fixed sandpaper surface. It does not train
stage-specific kinematics decoders or adapt a model to other participants.

## Declared protocol

- Verify named `AllLifts` columns, weight/surface labels, and trial-condition
  coverage. Audit P1–P12 current and previous conditions and series types.
- For P1, train on weight runs **1 and 4** (68 trials), validate on **7**
  (34 trials), and test on **8–9** (68 trials). Each selected run must contain
  165, 330 and 660 g trials at the same sandpaper surface.
- Use the full event-defined **LEDOn to hand-stop** interval. Verify the
  WS/AllLifts-to-HS clock conversion and raw sample matches for every selected
  trial. Fit EEG channel scalers on the training-trial intervals only.
- Keep 128-sample, 32-channel EEG windows with stride 64. Every input window
  stays within one trial and one run. No kinematics, event times, weight labels,
  run IDs or previous-condition metadata are classifier input features.
- Compare a multinomial logistic regression using 64 channel mean/standard-
  deviation features with the original CNN + ELU feature extractor and a
  three-class softmax head. Include a training-majority reference.
- Logistic C candidates are 0.01, 0.1, 1 and 10, selected by validation **trial**
  balanced accuracy. Neural seed 42, Adam, batch 64, fixed 15 final epochs;
  no test-based checkpoint selection. Require actual GPU neural computation.
- Weight training windows so each trial contributes equally within each class
  and the three classes have equal total training weight. Longer trials should
  not dominate the fitting objective simply by providing more windows.
- Primary prediction is one class per trial from the mean window probabilities.
  Report trial balanced accuracy, accuracy, macro F1, confusion matrices,
  per-class recalls and separate test-run results. Window scores are secondary
  and are not counted as independent test observations.
- For the permutation reference, hold predictions fixed and circularly shift
  whole-trial labels independently within test runs 8 and 9. Enumerate all
  **34 × 34 = 1,156** offsets, including the observed alignment. This preserves
  each run's class counts and repeated-weight schedule under circular wrapping.
  Report the exact upper-tail fraction and a Bonferroni adjustment across the
  two learned classifiers. This is a conditional label-association reference,
  not a training-label-permutation experiment or a guarantee of exchangeability.

## Interpretation limits

This is exploratory within-P1 discrimination on two held-out runs, not evidence
of a universal decoder. Correct classification can reflect condition-associated
EEG, movement, artifacts or experimental structure. Full-trial classification
does not establish that weight can be decoded before object contact. The
existing reduced EEG's upstream preprocessing remains incompletely documented.

Circular-shift inference assumes label/prediction alignment can be shifted
within a run under the reference null. Its preservation of label serial
structure does not establish that assumption, especially with time drift or
changing trial durations. Trial grouping prevents the inference from treating
overlapping windows as independent observations. If performance is weak, report
it without enlarging the search using test labels.

Condition and stage analyses are separate tasks. C3 pooled/fixed-condition
regression and C5 cross-stage decoders remain subsequent experiments.
