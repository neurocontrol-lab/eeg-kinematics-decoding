# EEG-to-Kinematics: project ideas and experiment backlog

Recorded 13 October 2026 from the second biweekly meeting transcript and the
earlier project discussion. This is an inventory, not authorization to execute
every experiment. Meeting timestamps locate the discussion; speakers are not
assigned reliably by the automatic transcript. Proposed implementation details
below are planning suggestions, not verbatim meeting decisions.

## Work already completed

| Work | Current outcome |
| --- | --- |
| Reproduce the original pipeline | Original CNN and CNN–BiLSTM runs retained as references, including CPU/GPU results. |
| Correct baseline evaluation in v2 | Whole-run splits; training-only scalers; independent validation; windows and next-sample targets stay within recording runs. |
| Identify the kinematic targets | Raw/reduced checks identify wrist X/Y/Z; the check was extended to P1–P12. Exact upstream processing and reduced physical units remain unresolved. |
| Establish six P1 references | Training mean, ridge, original CNN, CNN–BiLSTM, CNN + ELU, and EEGNet regression. ELU improves the original CNN in this single-seed experiment; CNN–BiLSTM leads within P1. |
| Test frozen P1 transfer | All six models and P1 scalers evaluated on P2–P12 without adaptation. Primary runs 8–9 and secondary all-run results show poor transfer. |
| Document results | Repository reports, numeric results, plots, and three Drive decks with linked indices. Deck condition labels now distinguish weight-series tests from trial surface codes. |

Links: [v2 methods](PIPELINE_V2_CORRECTED_BASELINE.md),
[model audit](MODEL_REFERENCE_AUDIT.md),
[six-model comparison](SIX_MODEL_COMPARISON.md),
[target provenance](KINEMATICS_PROVENANCE.md).

## 1. Dataset understanding and condition-aware experiments

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| C1 | Audit trial conditions and split coverage | Tabulate current/previous weight and surface, series type, and trial counts in train/validation/test. Establish which parameters change within a run and whether held-out conditions also occur in training. | 25:31–26:03; 28:08–30:02; 1:01:48–1:05:04. Completed initial P1–P12 audit: 3,528 trials; no held-out current weight/surface combinations absent from v2 training. Series-type composition still differs. |
| C2 | Classify experimental condition from EEG | Can EEG distinguish weight, surface, or joint weight/surface conditions above an appropriate chance reference? Begin with one controlled classification question. | 27:11–28:08; 36:03–36:20. First fixed-sandpaper P1 weight experiment completed: 68 test trials, logistic balanced accuracy 32.48%, CNN + ELU 27.73%, versus 33.33% chance. No reliable discrimination in this setup; surface/joint tasks and broader validation remain open. |
| C3 | Compare fixed-condition and pooled decoders | Compare training/testing within a defined condition with mixed-condition training/testing. Report condition-specific performance instead of relying only on an aggregate score. | 28:28–30:02; 33:26–34:16. Open. |
| C4 | Evaluate held-out experimental conditions | Separate generalization to new recordings of known conditions from generalization to conditions absent during training. | 1:03:45–1:05:04; earlier discussion of holdouts covering other friction conditions. Open. |
| C5 | Test whether decoding mappings differ between movement stages | Use verified event markers to plot a pooled decoder's error across stages within a held-out trial. Then compare stage-trained decoders within and across stages using a training-stage × evaluation-stage error matrix. Consistently high cross-stage error alongside low within-stage error would support stage-specific decoders, after controlling for condition, sample availability and stage difficulty. | 31:31–34:16; 43:03–43:21; 50:45–51:09. Clarified in follow-up discussion. Distinct neural control routines remain a hypothesis, not a conclusion from decoder errors alone. |
| C6 | Examine changed versus repeated conditions | Check whether trials following unexpected weight/surface changes have different EEG or prediction errors from repeated-condition trials. | 30:23–31:00. A discussion concern, not a finalized experiment. |

Working definitions: a run here is a dataset series containing multiple trials.
`BlockType` identifies the series type; `CurW` and `CurS` identify trial weight
and surface. In the supplied P1–P12 files, runs 7–9 are weight series with a
fixed surface. Local WS files label their surface as sandpaper. Runs 1–6 include
weight, friction and mixed series, with order depending on participant. Do not
infer trial condition from run number alone.

[C1 audit](../outputs/condition-audit-20261014T115618Z/RESULTS.md)
· [C2 classifier results](../outputs/weight-classification-p1-gpu-20261014T115817Z/RESULTS.md)
· [Declared methods and limits](CONDITION_CLASSIFICATION.md).
The classifier uses full LEDOn-to-hand-stop trials, not stage-specific cropping.
All 170 selected P1 trial clocks and every trial/window boundary were verified.
These findings do not establish whether another representation or task could
classify condition successfully. No C3–C6 decoder training occurred.

## 2. Additional results from the existing decoders

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| R1 | Reconstruct predictions in chronological order | Recover observed and estimated X/Y/Z in time order for intact held-out trials or runs, keeping the original target timestamps. Supply these aligned samples to R2. | 47:36–50:12; 1:28:23–1:29:12. Initial P1 runs 8–9 analysis completed for the six reference models without retraining. |
| R2 | Plot observed/predicted coordinates and error versus time | Overlay y(t) and ŷ(t) for each wrist axis, with an aligned error curve and verified trial-stage boundaries. Compare errors across stages within the same trial to identify consistently harder stages and inform C5. | 49:00–51:09. Initial P1 event-aligned plots completed. Movie/animation was the user's visualization idea and is not a required deliverable. |
| R3 | Analyze systematic errors | Determine whether errors are persistent offsets, amplitude mismatches, temporal lag, or vary with movement stage and experimental condition. | 50:45–51:09; 1:19:25–1:20:27; 1:28:45–1:29:12. Initial P1 bias, phase and descriptive lag diagnostics completed; broader condition-controlled analysis remains open. |
| R4 | Explain dense scatter-plot clusters | Trace points back to their times and trial events. Check whether clusters correspond to stationary/resting positions, and report the fraction of samples inside/outside them. | 1:06:18–1:09:06. Initial P1 density, phase and speed-composition summaries completed. Density alone does not establish rest. |
| R5 | Report movement and stationary performance separately | Assess whether aggregate scores mainly reflect stationary periods rather than reconstruction during movement. Event labels and velocity can support the distinction after validation. | 50:55–51:09; 1:07:13–1:08:23; earlier discussion of full-recording rest periods. Initial P1 velocity-threshold/smoothing sensitivity completed; low speed is an exploratory proxy, not validated physical rest. |

These are analyses of existing models. They do not require silently replacing
the original training protocol or treating interpolated predictions as newly
decoded samples.

[Initial P1 results and plots](../outputs/saved-decoder-analysis-p1-20261013T172654Z/RESULTS.md)
· [Methods and limits](SAVED_DECODER_ANALYSES.md).
The analysis verifies all 68 held-out trial clocks, reports both target-phase
and pure-window metrics, and retains trial-level phase summaries. It does not
train stage-specific decoders or cover P2–P12 phase analysis yet.

## 3. Timing, windows and trajectory prediction

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| T1 | Optimize the EEG-to-kinematics lag | Sweep a declared prediction horizon/alignment, selecting it on validation data. Compare with a literature-informed hypothesis about delay. | 07:42–10:33. Open. A physiological conduction delay is not automatically the optimal wrist-kinematics lag. |
| T2 | Vary input-window length | Test whether more or less EEG history changes accuracy and latency. | 13:27–13:40. Open. |
| T3 | Vary stride and output frequency | Compare the current 64-sample stride with denser sampling, potentially one sample. Distinguish more frequent predictions from a model that explicitly learns trajectory dynamics. | 13:27–13:40; 40:01–41:07; 1:29:14–1:29:49. Open. |
| T4 | Predict a sequence rather than one point | Use sequence outputs to estimate several future positions or a trajectory segment; consider whether state should persist across windows. | 10:43–12:03; 1:29:33–1:30:00. Open. |
| T5 | Combine EEG with past kinematics | Compare a kinematics-only autoregressive reference with a decoder that also receives EEG. Determine what extra information EEG contributes. | 1:30:00–1:32:03. Open. |
| T6 | Compare observed-feedback and recursive rollout | Predict one step using the latest observed state, or feed predictions back when future observations are unavailable. Measure error accumulation over longer horizons. | 1:33:03–1:34:42. Open. Declare what information is available at each prediction time. |

Future or contemporaneous target kinematics must not be supplied as past-state
inputs. Any comparison changing lag/window/stride should also account for which
target times remain eligible for evaluation.

## 4. Participant transfer and adaptation

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| P1 | Diagnose participant distribution differences | Examine EEG baseline levels, output offsets/scales, apparent lag and possible representation changes before attributing transfer failure to the learned mapping itself. | 45:48–47:36; 1:19:39–1:21:31. Open. Rotation/cap alignment were explanatory possibilities, not established causes. |
| P2 | Test calibration-based normalization | Keep model weights frozen but estimate normalization from a target participant's reserved training/calibration runs. Compare with the original frozen P1 scalers. | 1:20:27–1:21:31. Open. EEG-only normalization and target-coordinate calibration should be separate ablations. |
| P3 | Fine-tune only the output layer | Freeze the source feature extractor and train the final prediction layer using target calibration data, then test held-out target runs. | 1:24:35–1:25:29. Open. This is supervised adaptation, unlike the completed zero-adaptation experiment. |
| P4 | Repeat transfer for every source participant | Train on each participant in turn and evaluate the others. Build a source-by-target transfer matrix to check whether P1 is representative. | 1:23:08–1:24:35. Open. |
| P5 | Train a population decoder | Train across several participants and evaluate a completely held-out participant, using a participant-level split. | 1:25:39–1:27:03. Open. A leave-one-participant-out design was discussed. |
| P6 | Adapt a population decoder to an individual | Compare population-model zero-adaptation results with individual calibration/fine-tuning as a second training stage. | 1:27:06–1:27:25. Open. |

Neither failed transfer from P1 nor a successful condition classifier establishes
whether a universal neural encoding exists. The former tests a particular
decoder/representation; the latter may also reflect artifacts or experimental
structure. Keep calibration data separate from evaluation and state whether
target kinematic labels were used.

## 5. EMG and broader extensions

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| M1 | Use an EEG-to-EMG-to-kinematics cascade | Test the intermediate EMG representation in the project’s proposed second phase. Compare the cascade with direct EEG-to-kinematics decoding. | 1:32:08–1:32:48. Open. |
| M2 | Feed EEG and EMG jointly into the decoder | Compare EEG-only, EMG-only and combined inputs to determine the contribution of each modality. | 1:32:25–1:32:56. Joint input was discussed; the modality comparison is a suggested evaluation detail. |
| M3 | Test cross-dataset or cross-laboratory transfer | Extend beyond within-dataset participant transfer once preprocessing, target definitions and evaluation are stable. | Earlier project roadmap. Longer-term; not the immediate meeting priority. |

## 6. Pipeline, evaluation and deployment improvements

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| E1 | Measure inference latency for every architecture | Record time to produce an output with frozen weights, rather than only training duration. Compare CPU/GPU and single-window versus batch processing. | 16:24–17:34. Explicit requested result. Warm-up, synchronization and latency distributions are suggested measurement details. |
| E2 | Establish uncertainty and robustness | Repeat seeds and predeclared run holdouts, including other surface conditions. Assess whether model rankings and gains persist. | Earlier project discussion and documented baseline limitations. Open. Use trials/runs/participants appropriately rather than counting overlapping windows as independent repetitions. |
| E3 | Make raw preprocessing and target coordinates reproducible | Resolve upstream EEG cleaning, target filtering/scaling and units. Define consistent preprocessing and coordinates before an end-to-end deployment claim. | Earlier provenance and pipeline discussion; linked to normalization and calibration concerns in this meeting. Open. |
| E4 | Preserve controlled architecture comparisons | Retain the original CNN/ridge references and ELU-only ablation when refining EEGNet or other models. Do not attribute a multi-component model change solely to activation. | Earlier six-model discussion. Initial comparison complete; future refinements remain open. |
| E5 | Add condition/phase/trajectory reporting | Extend the current numeric reports and decks with condition coverage, chronological trajectories, stage-specific errors and inference timing. | Multiple meeting requests above. Open reporting features. |
| E6 | Propagate metadata-label corrections | Replace ambiguous “friction 3” descriptions with explicit series type and surface-code labels in repository documentation/metadata where still needed. Keep the term “run” when it already means a dataset series. | Follow-up chat clarification. Deck corrections complete; repository consistency needs checking. |

## Suggested order, without starting new experiments

1. C1 and the initial P1 R1/R4/R5 analyses are complete. Establish what
   the decoder is getting right and whether rest or experimental composition
   dominates aggregate scores.
2. The first C2 experiment is complete: weight classification at fixed sandpaper,
   using runs 1 and 4 for training, 7 for validation, and 8–9 for testing.
   Neither learned classifier reliably exceeds chance on held-out trials.
   Retain this result before proposing a separate condition or representation
   experiment; do not retune against these test labels.
3. Use the audit to choose C3/C5 or T1 next, changing one factor at a time.
4. Address P1/P2/P3 before interpreting the frozen transfer result as evidence
   about shared encoding. Then broaden to P4/P5/P6.
5. Develop T4/T5/T6 and M1/M2 after the current evaluation and target definitions
   are established. E1 can be measured in parallel with analyses of saved models.

The order is a recommendation, not a claim that the meeting fixed every step.
In particular, it discussed several possible branches rather than one final
architecture or one mandatory event interval.

## Clarifications to retain when planning

- V2 uses continuous recordings, including rest. Its boundaries are recording
  runs, not trial-event pairs. The meeting's mention of trial-contained windows
  does not describe the implemented v2 preparation.
- V2 has 50% overlap within a partition; whole-run separation prevents shared
  samples across its train/validation/test partitions.
- The current target is one sample after the last input sample, not the entire
  next window or a trajectory sequence.
- Window-to-point prediction is still regression. The relevant distinction is
  independent window predictions versus explicit trajectory/dynamical modeling.
- The three reduced targets are wrist X/Y/Z, not glove/finger outputs. Their
  upstream transformation and physical-coordinate interpretation remain open.
- Experimental weight/surface can vary between trials within one run. Distinguish
  current/previous trial parameters from `BlockType` and from participant identity.

## Sources

- Meeting transcript: `C:/Users/satya/Music/Meeting Notes/transcripts/nasuto_2nd_biweekly.mkv/transcript.txt`.
  Automatic transcription contains technical-term errors; timestamps above
  point to the discussion rather than certify every statement as a fact.
- Earlier conversation: baseline corrections, six-model references, frozen P1
  evaluation, metadata clarification, and the project’s longer-term roadmap.
- Existing repository reports and methods linked above. Historical result
  files remain records of their original experiments.
