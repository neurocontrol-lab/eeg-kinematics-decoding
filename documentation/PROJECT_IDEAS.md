# EEG-to-Kinematics: project ideas and experiment backlog

Started 13 October 2026 from the second biweekly meeting and earlier project
discussion; revised 28 October 2026 after reviewing the third biweekly transcript. This is an inventory, not authorization to execute
every experiment. Speakers are not assigned reliably by the automatic transcript. Proposed implementation details
below are planning suggestions, not verbatim meeting decisions. **M3** marks
the third meeting.
Only the shared registry maintains live experiment statuses and priorities.

## Experiment tracking

The public [experiment registry and roadmap](https://docs.google.com/spreadsheets/d/1Q2nJqr66ftuz7CP9dMiYRdE2RGwavmGbMGDkOS7yiWg/edit) is the single maintained record of defined experiments, statuses, priorities, runs and findings. Update the Sheet directly. This document retains the ideas and discussion context; dated outcomes below are historical notes, not a second live status tracker.

## Earlier project milestones (historical context)

| Work | Current outcome |
| --- | --- |
| Reproduce the original pipeline | Original CNN and CNN–BiLSTM runs retained as references, including CPU/GPU results. |
| Correct baseline evaluation in v2 | Whole-run splits; training-only scalers; independent validation; windows and next-sample targets stay within recording runs. |
| Identify the kinematic targets | Raw/reduced checks identify wrist X/Y/Z; the check was extended to P1–P12. Exact upstream processing and reduced physical units remain unresolved. |
| Establish six P1 references | Training mean, ridge, original CNN, CNN–BiLSTM, CNN + ELU, and EEGNet regression. ELU improves the original CNN in this single-seed experiment; CNN–BiLSTM leads within P1. |
| Test frozen P1 transfer | All six models and P1 scalers evaluated on P2–P12 without adaptation. Primary runs 8–9 and secondary all-run results show poor transfer. |
| Document results | Repository reports, numeric results, plots, and five Drive decks with linked indices. Deck condition labels now distinguish weight-series tests from trial surface codes. |

Links: [v2 methods](methods/PIPELINE_V2_CORRECTED_BASELINE.md),
[model audit](methods/MODEL_REFERENCE_AUDIT.md),
[six-model comparison](methods/SIX_MODEL_COMPARISON.md),
[frozen P1 transfer](methods/FROZEN_P1_TRANSFER.md),
[target provenance](dataset/KINEMATICS_PROVENANCE.md).

## 1. Dataset understanding and condition-aware experiments

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| C1 | Audit trial conditions and split coverage | Tabulate current/previous weight and surface, series type, and trial counts in train/validation/test. Establish which parameters change within a run and whether held-out conditions also occur in training. | Completed initial P1–P12 audit: 3,528 trials; no held-out current weight/surface combinations absent from v2 training. Series-type composition still differs. |
| C2 | Classify experimental condition from EEG | Compare full-trial and event-restricted weight classification at fixed surface. Establish what subjects knew about the upcoming weight; compare preparation, loading/lift and hold intervals without assuming preparation has no information. Test declared temporal and band-power representations with shrinkage LDA/logistic regression and CNN on matched trial sets; balance trial contributions and report per-stage class coverage. | C_2 remains the completed negative full-trial reference. Stage-focused follow-up is C_10; feature ablations and exact intervals require a new protocol. C2 classification is separate from C3 kinematics regression. |
| C3 | Compare fixed-condition and pooled kinematics decoders | Compare pooled and condition-specific regressors on identical held-out trials, with full-pooled and sample-matched pooled references. Report condition × stage/dynamics errors to separate training-size and movement-composition effects. Keep architecture, preprocessing and timing fixed. | Refines untried C_3. A condition classifier succeeding is not a prerequisite, and improvement from specialization is a hypothesis. |
| C4 | Evaluate held-out experimental conditions | Separate generalization to new recordings of known conditions from generalization to conditions absent during training. | Earlier discussion of holdouts covering other friction conditions. Open. |
| C5 | Test stage-specific and cross-stage kinematics mappings | Compare pooled and stage-trained regressors with a training-stage × evaluation-stage error matrix. Match sample budgets and examine position variance, speed/direction, condition and input-window phase purity. Use event labels first; consider dynamics-based partitions from R6/C8 as a separate comparison. | Refines untried C_5. Event stages need not equal dynamical regimes; higher cross-stage error alone does not establish distinct neural populations. C7 success is not required. |
| C6 | Examine changed/repeated conditions and prior expectations | First audit whether upcoming weight/surface was announced, visible or predictable from the protocol and previous trials. Then compare changed/repeated trials while controlling current condition, stage and serial dependence. Distinguish documented instruction, previous-condition proxies and measured expectation. | Refines untried C_6. Do not assume pre-contact EEG cannot contain condition information, or infer expectation from a change label alone. |
| C7 | Retrospective movement-stage classification | Classify full event-defined EEG intervals with channel × time-bin features, shrinkage LDA and CNN. Add elapsed-time/duration-only controls and declare coarse stages, binning, short-stage handling and trial/run separation. Event identity is the label, never an EEG-model input. | Agreed 27 October 2026; refined 28 October after M3. Retrospective description of event stages is distinct from discovering dynamical regimes (C8). Real-time detection is deferred; C7 success is not required for C5. |
| C8 | Discover data-driven movement partitions | Compare EEG-input similarity clusters, kinematic dynamical regimes and optionally training-error-guided partitions with event-defined stages. Fit scaling, clustering, error thresholds and partition choices using training/validation only; assess held-out stability and relation to decoder error. | New: C_8. Kinematic/error-based labels provide retrospective or oracle analyses; an EEG-only gate is a separate question. Never define regimes using held-out residuals and claim independent improvement on those same residuals. |
| C9 | Combine specialist decoders across a whole trial | If C5/C8 show a specialist benefit, compare one pooled model, an oracle selector using known stage/regime labels, and an EEG-based selector or blend of specialists. Measure boundary errors and any loss from routing mistakes. | New: the third meeting raised how to put stage decoders together. C_9. Gate/blending comparisons are proposed implementation choices, not meeting-agreed methods. Whole-stage retrospective C7 cannot be used as an online gate without redesign. |

Working definitions: a run here is a dataset series containing multiple trials.
`BlockType` identifies the series type; `CurW` and `CurS` identify trial weight
and surface. In the supplied P1–P12 files, runs 7–9 are weight series with a
fixed surface. Local WS files label their surface as sandpaper. Runs 1–6 include
weight, friction and mixed series, with order depending on participant. Do not
infer trial condition from run number alone.

[C1 audit](../outputs/condition-audit-20261014T115618Z/RESULTS.md)
· [C2 classifier results](../outputs/weight-classification-p1-gpu-20261014T115817Z/RESULTS.md)
· [Declared methods and limits](methods/CONDITION_CLASSIFICATION.md).
The completed classifier used full LEDOn-to-hand-stop trials, not stage-specific cropping.
Its logistic model used **64 features** (32 channel means plus 32 standard
deviations); the CNN used **128 × 32 samples**. Same windows did not mean equal
input dimensionality. Circular controls rotated whole-trial labels within each
run with predictions fixed; they did not cycle class identities or retrain
models on shuffled labels. The constant majority reference stays at 1/3 balanced
accuracy under these shifts. Clarify these points in future presentations; the
transcript description is not a replacement for the saved implementation.
All 170 selected P1 trial clocks and every trial/window boundary were verified.
These findings do not establish whether another representation or task could
classify condition successfully. No C3–C6 decoder training occurred.

## 2. Additional results from the existing decoders

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| R1 | Reconstruct predictions in chronological order | Recover observed and estimated X/Y/Z in time order for intact held-out trials or runs, keeping the original target timestamps. Supply these aligned samples to R2. | Initial P1 runs 8–9 analysis completed for the six reference models without retraining. |
| R2 | Plot observed/predicted coordinates and error versus time | Overlay y(t) and ŷ(t) for each wrist axis, with an aligned error curve and verified trial-stage boundaries. Compare errors across stages within the same trial to identify consistently harder stages and inform C5. | Initial P1 event-aligned plots completed. Movie/animation was the user's visualization idea and is not a required deliverable. |
| R3 | Separate systematic bias, amplitude and timing errors | Describe per-axis and within-stage offsets, amplitude mismatch and prediction-versus-observation delay. Distinguish diagnostic post-hoc curve alignment from deployable correction; any learned correction is fitted on training/validation only. | Initial P1 diagnostics exist. T1 changes the EEG history interval; it does not resolve all output-trajectory delays or justify test-fitted offsets. |
| R4 | Explain diagonal clusters and off-diagonal scatter streaks | Trace diagonal clusters and horizontal/vertical streaks back to trial time, event stage, observed position and movement dynamics. Report error and sample composition separately; distinguish dense observed-position bins from dense observed/predicted 2D cells. | Existing density/speed summaries do not directly explain streaks. R6 is the new follow-up; do not assume diagonal points are all slow or every off-diagonal prediction is random. |
| R5 | Report low-speed and moving performance separately | Report continuous error-versus-speed curves and sensitivity to smoothing/thresholds, alongside within-stage comparisons. Keep low speed distinct from relaxed rest or absence of motor activity; holding the object can require ongoing muscle activity. | Initial P1 sensitivity summaries exist. R6 tests whether dynamics explain more than a binary speed threshold. |
| R6 | Explain decoder error through movement dynamics | Reuse saved P1 predictions to compare error against observed position, speed, signed velocity/direction, acceleration and exploratory jerk. Contrast position-only, stage-only, dynamics-only and combined explanations within trials/stages, and inspect the scatter streaks on all axes. | New follow-up: R_6. Reuse R4/R5 evidence without marking the new question complete. Derivatives come from observed high-rate trajectories with smoothing/edge sensitivity, not interpolated decoded points. These are retrospective kinematic explanations, not EEG-only prospective error prediction. |

These are analyses of existing models. They do not require silently replacing
the original training protocol or treating interpolated predictions as newly
decoded samples.

[Initial P1 results and plots](../outputs/saved-decoder-analysis-p1-20261013T172654Z/RESULTS.md)
· [Methods and limits](methods/SAVED_DECODER_ANALYSES.md).
The analysis verifies all 68 held-out trial clocks, reports both target-phase
and pure-window metrics, and retains trial-level phase summaries. It does not
train stage-specific decoders or cover P2–P12 phase analysis yet.

## 3. Timing, windows and trajectory prediction

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| T1 | Optimize the EEG-to-kinematics lag | Sweep a declared prediction horizon/alignment, selecting it on validation data. Compare with a literature-informed hypothesis about delay. | Initial T_1 pilot verified 27 October 2026: fixed 256-ms windows and 2/50/100/200-ms gaps. Ridge selected 2 ms; CNN selected 100 ms but test MSE worsened. No adopted improvement; broader timing/robustness questions remain open. A predictive lag is not a physiological conduction delay. |
| T2 | Vary EEG history with fixed temporal bins | Compare 256/512/1,024-ms histories represented by 128 channel-wise mean bins, with matched target timestamps and a fixed 2-ms end gap. Repeat separate ridge and CNN models for P1–P12 and summarize paired participant effects. | T_2 implements the user-requested history-plus-averaging comparison. Equal input dimensions do not isolate context from smoothing or temporal resolution; full-resolution controls remain future work. [Declared protocol](methods/HISTORY_DURATION.md). Live progress is in the shared registry. |
| T3 | Vary stride and output frequency | Compare the current 64-sample stride with denser sampling, potentially one sample. Distinguish more frequent predictions from a model that explicitly learns trajectory dynamics. | Open. |
| T4 | Predict a sequence rather than one point | Use sequence outputs to estimate several future positions or a trajectory segment; consider whether state should persist across windows. | Open. |
| T5 | Combine EEG with past kinematics | Compare a kinematics-only autoregressive reference with a decoder that also receives EEG. Determine what extra information EEG contributes. | Open. |
| T6 | Compare observed-feedback and recursive rollout | Predict one step using the latest observed state, or feed predictions back when future observations are unavailable. Measure error accumulation over longer horizons. | Open. Declare what information is available at each prediction time. |
| T7 | Compare position and movement targets | With fixed EEG inputs/splits, compare decoding wrist position and velocity, optionally a joint target. Use target-appropriate baselines and metrics; evaluate reconstructed position and drift separately if integrating velocity. | New experiment formulation inspired by the open question in the third meeting: position, movement or both? T_7. Target comparison is our proposed operationalization, not an explicit instruction to implement a velocity decoder. Lower error on differently scaled targets does not prove what EEG encodes. |

Future or contemporaneous target kinematics must not be supplied as past-state
inputs. Any comparison changing lag/window/stride should also account for which
target times remain eligible for evaluation.

## 4. Participant transfer and adaptation

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| P1 | Diagnose participant distribution differences | Examine EEG baseline levels, output offsets/scales, apparent lag and possible representation changes before attributing transfer failure to the learned mapping itself. | Open. Rotation/cap alignment were explanatory possibilities, not established causes. |
| P2 | Test calibration-based normalization | Keep model weights frozen but estimate normalization from a target participant's reserved training/calibration runs. Compare with the original frozen P1 scalers. | Open. EEG-only normalization and target-coordinate calibration should be separate ablations. |
| P3 | Fine-tune only the output layer | Freeze the source feature extractor and train the final prediction layer using target calibration data, then test held-out target runs. | Open. This is supervised adaptation, unlike the completed zero-adaptation experiment. |
| P4 | Repeat transfer for every source participant | Train on each participant in turn and evaluate the others. Build a source-by-target transfer matrix to check whether P1 is representative. | Open. |
| P5 | Train a population decoder | Train across several participants and evaluate a completely held-out participant, using a participant-level split. | Open. A leave-one-participant-out design was discussed. |
| P6 | Adapt a population decoder to an individual | Compare population-model zero-adaptation results with individual calibration/fine-tuning as a second training stage. | Open. |

Neither failed transfer from P1 nor a successful condition classifier establishes
whether a universal neural encoding exists. The former tests a particular
decoder/representation; the latter may also reflect artifacts or experimental
structure. Keep calibration data separate from evaluation and state whether
target kinematic labels were used.

## 5. EMG and broader extensions

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| M1 | Use an EEG-to-EMG-to-kinematics cascade | Test the intermediate EMG representation in the project’s proposed second phase. Compare the cascade with direct EEG-to-kinematics decoding. | Open. |
| M2 | Feed EEG and EMG jointly into the decoder | Compare EEG-only, EMG-only and combined inputs to determine the contribution of each modality. | Joint input was discussed; the modality comparison is a suggested evaluation detail. |
| M3 | Test cross-dataset or cross-laboratory transfer | Extend beyond within-dataset participant transfer once preprocessing, target definitions and evaluation are stable. | Earlier project roadmap. Longer-term; not the immediate meeting priority. |

## 6. Pipeline, evaluation and deployment improvements

| ID | Idea | Question or deliverable | Discussion / status |
| --- | --- | --- | --- |
| E1 | Measure inference latency for every architecture | Record time to produce an output with frozen weights, rather than only training duration. Compare CPU/GPU and single-window versus batch processing. | Initial warmed P1 CPU/GPU benchmark completed for all six references, batch sizes 1/64 and 300 calls per configuration. Acquisition/preprocessing and cold-start latency remain outside its scope. |
| E2 | Establish uncertainty and robustness | Repeat seeds and predeclared run holdouts, including other surface conditions. Assess whether model rankings and gains persist. | Earlier project discussion and documented baseline limitations. Open. Use trials/runs/participants appropriately rather than counting overlapping windows as independent repetitions. |
| E3 | Make raw preprocessing and target coordinates reproducible | Resolve upstream EEG cleaning, target filtering/scaling and units. Define consistent preprocessing and coordinates before an end-to-end deployment claim. | Earlier provenance and pipeline discussion; linked to normalization and calibration concerns in this meeting. Open. |
| E4 | Preserve controlled architecture comparisons | Retain the original CNN/ridge references and ELU-only ablation when refining EEGNet or other models. Do not attribute a multi-component model change solely to activation. | Earlier six-model discussion. Initial comparison complete; future refinements remain open. |
| E5 | Add condition/phase/trajectory reporting | Extend the current numeric reports and decks with condition coverage, chronological trajectories, stage-specific errors and inference timing. | Multiple meeting requests above. Open reporting features. |
| E6 | Propagate metadata-label corrections | Replace ambiguous “friction 3” descriptions with explicit series type and surface-code labels in repository documentation/metadata where still needed. Keep the term “run” when it already means a dataset series. | Follow-up chat clarification. Deck corrections complete; repository consistency needs checking. |
| E7 | Simulate dilution by uninformative labelled intervals | Use controlled synthetic data with a known informative feature-label relation, then mix increasing fractions of uninformative or randomly labelled observations. Keep those two mechanisms distinct and compare simple classifiers/CNN with balanced classes, independent splits and repeated seeds. | New: the third meeting raised the corrupted-data thought experiment. E_7. This isolates a mechanism; it cannot establish the cause of the real C2 failure or reproduce the CNN result by assumption. Lower priority than analysing the real data. |

## Planning decisions and recommendations

The 27 October agreement was T1 timing → retrospective stage classification
(C7) → C5 stage-specific regression → C3 condition-specific regression. T1's
executed negative finding remains part of the evidence, rather than a reason to
retune on test data.

**Recommendation from the third-meeting review, 28 October 2026:** insert the
saved-prediction dynamics analysis (R_6) before that remaining sequence.
It directly addresses the meeting's closing request and can inform whether
event stages are the most useful partitions. Retain C7 → C5 → C3 after this
review; stage-focused C2 classification (C_10) is a subsequent branch.
Data-driven segmentation and specialist integration are contingent extensions,
not prerequisites for all existing experiments. Position/velocity target
comparisons and the dilution simulation remain lower-priority ideas.

The shared registry records the recommended ranks and next decisions. New ranks
are planning recommendations, not a claim that the meeting agreed a strict
order or authorization to execute every proposal. Review each result before
finalizing the next protocol. Historical P1 runs 8–9 have already informed
these hypotheses: new analyses of them are exploratory. Any claim of predictive
explanatory-model performance requires fitting and selecting it on separate
training/validation groups; any stronger generalization claim needs newly
reserved runs/participants.

## Clarifications to retain when planning

- Event stages describe experimental/task landmarks; they need not identify
  separate dynamical regimes or neural control routines. A stage/dynamics
  analysis based on observed kinematics is retrospective unless a prospective
  information set and evaluation are explicitly defined.
- Subtracting preparation error gives excess error relative to that baseline;
  it does not isolate error caused by movement. Preparation and holding are
  not evidence of relaxed muscles or zero neural information.
- V2 uses continuous recordings, including rest. Its boundaries are recording
  runs, not trial-event pairs. The meeting's mention of trial-contained windows
  does not describe the implemented v2 preparation.
- V2 has 50% overlap within a partition; whole-run separation prevents shared
  samples across its train/validation/test partitions.
- The current target is one sample after the last input sample, not the entire
  next window or a trajectory sequence.
- Window-to-point prediction is still regression. The relevant distinction is
  independent window predictions versus explicit trajectory/dynamical modeling.
- The three reduced targets are wrist X/Y/Z, not glove/finger outputs. Z is vertical (+ upward); X/Y anatomical orientation and the exact upstream transformation remain unresolved.
- Experimental weight/surface can vary between trials within one run. Distinguish
  current/previous trial parameters from `BlockType` and from participant identity.

## Sources

- Second biweekly meeting transcript (local meeting notes).
  Automatic transcription contains technical-term errors; consult the source
  when checking the original discussion.
- Third biweekly meeting transcript (local meeting notes).
  M3 references concern this source. Review date is 28 October 2026; the file
  modification date is not used to infer the meeting date. Methodological
  controls and suggested algorithms above are our planning interpretations.
- Earlier conversation: baseline corrections, six-model references, frozen P1
  evaluation, metadata clarification, and the project’s longer-term roadmap.
- Existing repository reports and methods linked above. Historical result
  files remain records of their original experiments.
