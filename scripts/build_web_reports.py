"""Generate the standalone experiment reports from saved aggregate results."""
import argparse
import base64
import csv
import html
import json
from math import isclose
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/"
CAT = [
    ("dataset-audits", "Dataset audits", "Trial conditions, split coverage and coordinate checks."),
    ("continuous-decoding", "Continuous kinematics decoding", "Original and corrected P1 baselines."),
    ("model-comparison", "Model comparison", "Six P1 regressors on the corrected protocol."),
    ("participant-transfer", "Cross-participant decoding", "Frozen P1 fits evaluated across P2–P12."),
    ("movement-stages", "Movement-stage errors", "Stage-resolved errors and retrospective dynamics."),
    ("condition-decoding", "Condition decoding", "Weight decoding across stages, repeats and switches."),
    ("inference-latency", "Inference benchmark", "Warmed CPU and GPU model-call latency."),
    ("timing-alignment", "Timing alignment", "Prediction leads and paired held-out results."),
    ("history-duration", "EEG history duration", "Matched 256, 512 and 1,024 ms EEG histories."),
    ("movement-targets", "Movement targets", "Wrist position compared with signed velocity."),
]
NAMES = {
    "training_mean": "Training mean", "training_majority": "Training majority", "cnn": "CNN",
    "cnn_elu": "CNN + ELU", "cnn_bilstm": "CNN–BiLSTM", "eegnet_regression": "EEGNet",
    "ridge": "Ridge", "shrinkage_lda": "Shrinkage LDA", "logistic": "Logistic regression",
    "timing_only": "Duration only", "logistic_plus_duration": "EEG + duration",
    "previous_weight": "Previous weight", "schedule": "Schedule lookup",
}
def label(code):
    return NAMES.get(code, code.replace("_", " ").capitalize())


def build_data(root):
    outputs = root / "outputs"
    def data(run, file="results.json"):
        return json.loads((outputs / run / file).read_text(encoding="utf-8-sig"))
    def rows(run, file):
        with (outputs / run / file).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    def study(title, summary, sources, views, figures=(), limits="", terms=()):
        return dict(title=title, summary=summary, sources=[(title, GITHUB + "outputs/" + path)
                    for title, path in sources], views=views,
                    figures=[(caption, "data:image/png;base64," + base64.b64encode((outputs / path).read_bytes()).decode("ascii"))
                             for caption, path in figures], limits=limits, terms=terms)
    def view(title, headers, records, filters=(), chart=None, reference=None, description="", order=None, initial=None, selectors=()):
        records = list(records)
        if not records or any(len(row) != len(headers) for row in records):
            raise ValueError(f"Missing data or mismatched columns: {title}")
        return dict(title=title, headers=headers, rows=records, filters=list(filters), chart=chart,
                    reference=reference, description=description, order=order, initial=initial or {}, selectors=list(selectors))
    def ci(values, mult=1):
        return tuple(float(x)*mult for x in values)
    def pct(value):
        return 100 * float(value)
    def source(run, filename="RESULTS.md", caption="Detailed report"):
        return caption, f"{run}/{filename}"

    audit = "condition-audit-20261014T115618Z"
    axes = "vertical-axis-audit-20261020"
    counts = {}
    for r in rows(audit, "trial_conditions.csv"):
        key = tuple(r[k] for k in ("participant","run","v2_split","series_type","weight_grams","surface_label"))
        counts[key] = counts.get(key, 0) + 1
    audit_records = [(p,int(s),sp,kind,int(w),surface,n) for (p,s,sp,kind,w,surface),n in counts.items()]
    assert sum(row[-1] for row in audit_records) == 3528
    axis_records = [(p,target.capitalize(),axis.upper(),r["hold_minus_prelift_cm_median"],
                     r["postreplace_minus_hold_cm_median"],r["dominant_lift_displacement_count"],val["trials"])
                    for p,val in data(axes)["participants"].items() for target in ("object","wrist")
                    for axis,r in val[target].items()]
    audits = [
        study("Trial-condition coverage", "Inspect the distribution of weight and surface across participants, series and the corrected decoding split.",
              [source(audit)], [view("Trial counts",["Participant","Series","Split","Series type","Weight (g)","Surface","Trials"],
              audit_records,[0,2,3,4,5],description="Counts are grouped by participant, series and current weight/surface. Series type comes from BlockType.")]),
        study("Vertical-axis audit", "Raw object and wrist trajectories rise most along Z during lift. Force sensors use a different axis convention.",
              [source(axes,"REPORT.md")], [view("Lift and lowering displacement",["Participant","Target","Axis","Median lift (cm)","Median lowering (cm)","Dominant lift trials","Trials"],axis_records,[0,1,2])],
              [("Example lift and lowering traces",f"{axes}/axis_check_example.png")]),
    ]

    def model_view(run):
        saved = data(run)["models"]
        rec = []
        for name, val in saved.items():
            score = val.get("test",val)
            for i, axis in enumerate("XYZ"):
                rec.append((label(name),axis,score["r2"][i],score["mse"][i],score["mae"][i]))
            rec.append((label(name),"Mean XYZ",mean(score["r2"]),mean(score["mse"]),mean(score["mae"])))
        return view("Axis metrics",["Model","Axis","R²","MSE","MAE"],rec,[0,1],default_chart(0,2,1),0,
                    "The mean is the arithmetic mean of the three axis scores.", ["Mean XYZ","X","Y","Z"])

    original = "original-p1-gpu"
    correct = "pipeline-v2-corrected-baseline-p1-gpu-20260929T173627Z"
    six = "six-model-comparison-p1-gpu-20260930T095506Z"
    baseline = [
        study("Original P1 baseline", "The original CNN and CNN–BiLSTM are retained for reproduction.", [source(original)], [model_view(original)],
              limits="This original evaluation protocol differs from the corrected run. Compare scores within a protocol."),
        study("Corrected P1 baseline", "Whole-series splits, training-only scaling and windows kept within recording series.", [source(correct)], [model_view(correct)],
              [("Training and validation learning curves",f"{correct}/learning_curves.png")],
              "The displayed errors use training-standardized targets."),
    ]
    comparison = [study("Six corrected P1 reference models", "Compare the training mean, ridge, CNN, CNN + ELU, CNN–BiLSTM and EEGNet on the same P1 test split.",
                        [source(six)], [model_view(six)], limits="These fits use P1 and a single seed for the neural models.")]

    frozen = "frozen-p1-cross-participant-gpu-20260930T125956Z"
    r_transfer = [(r["participant"],label(r["model"]),{"all_evaluated_runs":"All evaluated series","primary_runs_8_9":"Test series 8–9"}[r["scope"]],int(r["windows"]),
                   *(float(r[f"r2_{a}"]) for a in "xyz"),mean(float(r[f"r2_{a}"]) for a in "xyz"))
                  for r in rows(frozen,"participant_metrics.csv")]
    transfer = [study("Frozen P1 transfer", "The P1 models and scalers were reused across participants without adaptation.",
                     [source(frozen)], [view("Per-participant transfer",["Participant","Model","Scope","Windows","R² X","R² Y","R² Z","Mean R²"],r_transfer,[0,1,2],default_chart(0,7,1),0,initial={"Scope":"Test series 8–9"})],
                     limits="P1 is the source reference. The two evaluation scopes cover different recording series.")]

    saved = "saved-decoder-analysis-p1-20261013T172654Z"
    r6 = "r6-population-error-models-20261111"
    r6follow = "t1-r6-followup-20261110"
    phase = [(label(r["model"]),label(r["stage"]),label(r["subset"]),int(r["n"]),float(r["mean_euclidean_error"]),
              *(float(r[f"mae_{a}"]) for a in "xyz"))
             for r in rows(saved,"phase_metrics.csv") if r["mean_euclidean_error"]]
    r6_errors = [(r["participant"],label(r["model"]),int(r["window_ms"]),label(r["features"]),int(r["trials"]),
                  float(r["mse"]),float(r["r2"]) if r["r2"] else None) for r in rows(r6,"participant_error_models.csv")]
    r6_corr = [(r["participant"],label(r["model"]),label(r["stage"]),int(r["trials"]),float(r["rho"]))
               for r in rows(r6,"participant_stage_correlations.csv")]
    stages = [
        study("Saved P1 decoder errors", "Compare error by movement phase and by the retained sample subset.",
              [source(saved)], [view("Phase error",["Model","Stage","Subset","Targets","Mean error norm","MAE X","MAE Y","MAE Z"],
              phase,[0,1,2],default_chart(1,4,0),initial={"Subset":"All targets"})], [("Phase comparison",f"{saved}/phase_error_comparison.png")]),
        study("Population error explanation", "Compare the fit of retrospective error models and the within-stage association between speed and decoder error.",
              [source(r6),source(r6follow,caption="Population streak follow-up")], [
                  view("Error models",["Participant","Decoder","Smoothing (ms)","Features","Trials","Error-model MSE","Error-model R²"],r6_errors,[0,1,2,3]),
                  view("Speed–error correlation",["Participant","Decoder","Stage","Trials","Spearman ρ"],r6_corr,[0,1,2],default_chart(2,4,1),0,initial={"Participant":"P1"})],
              [("Population smoothing sensitivity",f"{r6}/smoothing_comparison.png")],
              "These scores describe how well observed kinematics and stages explain saved decoder errors."),
    ]

    pilot = "weight-classification-p1-gpu-20261014T115817Z"
    c10 = "c10-stage-weight-20261114"
    full = "c10-stage-weight-full-stage-20261114"
    condition = condition_studies(c10, full, data, rows, view, study, source)

    p_rows = [(label(m),split.capitalize(),pct(r[split]["trial"]["balanced_accuracy"]),r[split]["trial"]["macro_f1"],r[split]["trial"]["n"])
              for m,r in data(pilot)["models"].items() for split in ("validation","test")]
    condition.append(study("P1 full-trial weight pilot","Three weight classes at fixed sandpaper. Both EEG classifiers scored below the held-out chance reference.",
                           [source(pilot)],[view("Trial classification",["Model","Split","Balanced accuracy (%)","Macro F1","Trials"],
                           p_rows,[1],default_chart(0,2,1),100/3,initial={"Split":"Test"})],
                           [("Trial confusion matrices",f"{pilot}/trial_confusion_matrices.png")]))

    perf = "inference-benchmark-p1-20261014"
    latency_rows = [(device.upper(),label(m),int(batch),val["median_ms"],val["p95_ms"],val["median_ms_per_window"],val["n_calls"])
                    for device in ("cpu","gpu") for m,r in data(perf,device+"_results.json")["models"].items()
                    for batch,val in r.items()]
    inference = [study("CPU and GPU inference latency","Compare warmed, synchronized calls at batch sizes 1 and 64.",[source(perf)],
        [view("Model-call latency",["Device","Model","Batch","Median (ms)","P95 (ms)","Median per window (ms)","Calls"],
        latency_rows,[0,1,2],default_chart(1,5,0),description="Use the batch filter to compare device latency. Includes transfer and synchronization, with preprocessing outside the timed call.",initial={"Batch":1})])]

    timing = "timing-decoder-preferences-p1-p12-20261111T113206Z"
    old_timing = "timing-alignment-population-gpu-20261028T133000Z"
    held = data(timing + "/heldout","population_summary.json")
    val_rows = [(r["participant"],label(r["decoder"]),int(r["gap_ms"]),float(r["validation_mse"]))
                for r in rows(timing,"validation_scores.csv")]
    assert len(val_rows)==288
    gap_data=data(timing,"gap_counts.json")
    gap_rows=[(label(m),int(g),n) for m,counts in gap_data["per_decoder"].items() for g,n in counts.items()]
    gap_rows += [("CNN + ELU (3 seeds)",int(g),n) for g,n in gap_data["cnn_elu_three_seed_reference"].items()]
    test_rows=[(r["participant"],label(r["model"]),r["selected_gap_ms"],r["baseline_mse"],r["selected_mse"],r["selected_minus_baseline"])
               for r in held["participant_metrics"]]
    pop=[(label(m),r["mean_baseline_mse"],r["mean_selected_mse"],r["mean_selected_minus_baseline"],*r["paired_mse_difference_95_interval"],
          r["participants_improved"],r["participants_unchanged"],r["participants_worse"]) for m,r in held["population"].items()]
    timing_studies = [
        study("Prediction-lead validation", "Compare the six prediction leads using the same EEG windows, eligible timestamps and participant-specific splits.",
              [source(timing),source(old_timing,caption="Original population timing report")], [
              view("Selected lead counts",["Model","Lead (ms)","Participants"],gap_rows,[0],default_chart(1,2,0),
                   description="CNN + ELU and the two newer CNNs use seed 42 here. The three-seed CNN + ELU row gives the sensitivity comparison."),
              view("Validation scores",["Participant","Model","Lead (ms)","Validation MSE"],val_rows,[0,1,2])]),
        study("Held-out prediction-lead effects", "Compare the validation-selected lead with 2 ms on identical test targets.",
              [source(timing+"/heldout","population_summary.json","Saved held-out summary"),source(timing+"/heldout","verification.json","Verification")], [
              view("Population effects",["Model","2 ms MSE","Selected MSE","Change","95% lower","95% upper","Improved","Same","Worse"],
                   pop,[0],default_chart(0,3,None,4,5),0),
              view("Participant effects",["Participant","Model","Selected lead (ms)","2 ms MSE","Selected MSE","MSE change"],test_rows,[0,1],default_chart(0,5,1),0)],
              limits="CNN + ELU averages three seeds here. CNN–BiLSTM and EEGNet each use seed 42. Negative MSE changes favour the selected lead."),
    ]

    history = "history-duration-p1-p12-gpu-20261028T132600Z"
    h_rows = [(r["participant"],label(r["model"]),int(r["history_ms"]),float(r["baseline_mse"]),float(r["longer_mse"]),
               float(r["relative_mse_reduction_pct"])) for r in rows(history,"participant_effects.csv")]
    histories = [study("History length at a fixed input shape","Compare longer histories with the 256 ms reference at a fixed 2 ms prediction lead.",
        [source(history),source(history,"verification.json","Verification")],
        [view("Longer-history effects",["Participant","Model","History (ms)","256 ms MSE","Longer-history MSE","Relative MSE reduction (%)"],
        h_rows,[0,1,2],default_chart(0,5,2),0,initial={"Model":"Ridge"})], [("Population effects",f"{history}/group_effects.png")],
        "Longer histories change context and time-bin width together.")]
    velocity = "position-velocity-p1-p12-20261110T171409Z"
    target_rows = [(r["participant"],label(r["model"]),label(r["target"]),float(r["skill"]),*(float(r[f"r2_{a}"]) for a in "xyz"))
                   for r in rows(velocity,"participant_metrics.csv")]
    targets = [study("Position and signed velocity", "Compare target-specific skill at a fixed 2 ms prediction lead across P1–P12.",
        [source(velocity),source(velocity,"verification.json","Verification")],
        [view("Participant and target scores",["Participant","Model","Target","Skill vs training mean","R² X","R² Y","R² Z"],
              target_rows,[0,1,2],default_chart(0,3,2),0,initial={"Model":"Ridge"})], [("Paired population target skill",f"{velocity}/population_target_skill.png")],
        "Skill is measured against the training mean of each target. Velocity labels use centered smoothing of the observed trace.")]
    explanations = (
        (("Recording series", "A series contains multiple trials. Series type describes whether weight, surface friction, or both vary within it."),),
        (("R²", "The fraction of observed variation explained by prediction. Zero matches the constant held-out mean. Negative values score worse than that reference."),
         ("MSE and MAE", "Mean squared error gives larger misses more weight. Mean absolute error averages the size of misses.")),
        (("R²", "Zero matches the constant held-out mean. Negative values score worse than that reference."),
         ("Mean XYZ", "The arithmetic mean of the three separately calculated wrist-axis scores.")),
        (("Frozen model", "The P1-trained model and scalers are reused as-is for each participant. P1 is the source reference."),),
        (("Error norm", "The three-dimensional distance between the predicted and observed wrist position."),),
        (),
        (("Batch", "The number of EEG windows processed in a single model call."), ("P95 latency", "The time that 95% of measured calls finished within.")),
        (("Prediction lead", "The time from the final sample in the EEG input window to the wrist-position target."),),
        (("EEG history", "The interval of EEG a model receives for each prediction. Every history is represented by 128 time bins."),),
        (("Skill vs training mean", "A score of zero matches a constant prediction based on the training mean. Positive scores improve on that reference."),),
    )
    categories = [audits, baseline, comparison, transfer, stages, condition, inference, timing_studies, histories, targets]
    for studies, terms in zip(categories, explanations):
        if terms:
            studies[0]["terms"] = terms
    return categories


def condition_studies(crop, full, data, rows, view, study, source):
    """Keep published cohort intervals and derive individual summaries from saved probabilities."""
    stages = ("Preparation", "Reach", "Loading", "Lift/hold")
    cohort = "Cohort (P1–P12)"
    representations = (("Full stage", full), ("Centered 64 ms crop", crop))
    controls = ("Previous weight", "Schedule lookup")
    accuracy_rows, margin_rows = [], []
    individuals = {}

    for representation, run in representations:
        for row in data(run, "report_participant_summary.json"):
            key = (representation, row["participant"], row["model"], row["stage"], row["group"])
            if key in individuals:
                raise ValueError(f"Duplicate participant summary: {key}")
            individuals[key] = (row["balanced_accuracy_pct"], row["switch_preference_pp"], row["trial_count"])

        saved = data(run, "summary.json")["models"]
        for model, metrics in saved.items():
            for stage, result in metrics["stages"].items():
                for group in ("Repeat", "Switch"):
                    matching = sorted((p, values) for (rep, p, m, s, g), values in individuals.items()
                                      if (rep, m, s, g) == (representation, model, stage, group))
                    lower, upper = (100 * v for v in result[group.lower() + "_balanced_accuracy"][1:])
                    score = 100 * result[group.lower() + "_balanced_accuracy"][0]
                    if len(matching) != 12 or not isclose(mean(v[0] for _, v in matching), score, abs_tol=1e-9):
                        raise ValueError(f"Participant scores do not reproduce the cohort: {model}, {stage}, {group}")
                    prefix = (representation, label(model), stage, group)
                    accuracy_rows.append((*prefix, cohort, score, lower, upper, sum(v[2] for _, v in matching)))
                    accuracy_rows.extend((*prefix, p, v[0], None, None, v[2]) for p, v in matching)
                    if group == "Switch":
                        score, lower, upper = (100 * v for v in result["actual_minus_previous"])
                        if not isclose(mean(v[1] for _, v in matching), score, abs_tol=1e-9):
                            raise ValueError(f"Participant preferences do not reproduce the cohort: {model}, {stage}")
                        prefix = (representation, label(model), stage)
                        margin_rows.append((*prefix, cohort, score, lower, upper, sum(v[2] for _, v in matching)))
                        margin_rows.extend((*prefix, p, v[1], None, None, v[2]) for p, v in matching)

    definitions = (
        ("Full stage and 64 ms crop", "Full stage averages each EEG channel into 32 ordered bins between that stage’s event markers. The crop uses 32 consecutive samples from the centre of the stage."),
        ("Repeat and switch trials", "A repeat uses the same object weight as the immediately preceding trial in that recording series. A switch changes the weight, for example from 165 g to 330 g."),
        ("Balanced accuracy", "The average of the three weight-class recalls, giving each weight equal influence. The chance reference is 33.3%."),
        ("Switch preference", "On switch trials, subtract the decoder’s probability for the previous weight from its probability for the current weight. For example, 60% for the current weight minus 25% for the previous weight gives +35 percentage points. Positive values favour the current weight. Each of the six possible weight transitions gets equal weight."),
        ("Duration control", "Duration only uses the elapsed time between the stage’s start and end events. EEG + duration gives logistic regression both full-stage EEG features and that duration."),
    )
    scope = "The cohort scores average P1–P12 equally. The 95% intervals resample participants and appear only in the cohort view. Past-weight and schedule controls are available in the model filter."
    chosen = {"EEG input": "Full stage", "Participant": cohort}
    references = [source(full, caption="Full-stage report"), source(crop, caption="64 ms crop report")]
    first = study("C_10 · Weight decoding by movement stage",
        "Explore the same 769 held-out trials with EEG from the entire stage or its central 64 ms. The trial set includes 494 repeats and 275 switches.", references, [
        view("Balanced accuracy", ["EEG input","Model","Stage","Trial group","Participant","Balanced accuracy (%)","95% lower","95% upper","Trials"],
             accuracy_rows, [0,4,1,2,3], default_chart(2,5,1,6,7,context_only=controls), 100/3,
             scope, stages, initial={**chosen, "Trial group":"Switch"}, selectors=(0,4,3)),
        view("Switch preference", ["EEG input","Model","Stage","Participant","Current minus previous (pp)","95% lower","95% upper","Trials"],
             margin_rows, [0,3,1,2], default_chart(2,4,1,5,6,context_only=controls), 0,
             scope, stages, initial=chosen, selectors=(0,3)),
    ], limits="Post-contact results may reflect movement, sensory feedback or artifacts. The schedule lookup uses the shared trial sequence.", terms=definitions)

    comparison_rows = []
    for model, by_stage in data(full, "paired_comparison.json")["comparisons"].items():
        for stage, by_group in by_stage.items():
            for group, result in by_group.items():
                group = group.capitalize()
                prefix = ("EEG + duration vs duration" if model == "incremental" else label(model), stage, group)
                metric = result["balanced_accuracy"]["delta"]
                comparison_rows.append((*prefix, cohort, *(100 * x for x in metric)))
                effects = []
                for participant in (f"P{i}" for i in range(1, 13)):
                    new_model = "logistic_plus_duration" if model == "incremental" else model
                    ref_model = "timing_only" if model == "incremental" else model
                    ref_rep = "Full stage" if model == "incremental" else "Centered 64 ms crop"
                    point = individuals["Full stage", participant, new_model, stage, group][0] - individuals[ref_rep, participant, ref_model, stage, group][0]
                    effects.append(point)
                    comparison_rows.append((*prefix, participant, point, None, None))
                if not isclose(mean(effects), 100 * metric[0], abs_tol=1e-9):
                    raise ValueError(f"Participant changes do not reproduce the cohort: {model}, {stage}, {group}")
    second = study("C_10 · Paired changes in accuracy",
        "Positive changes favour the full-stage EEG representation or the combined EEG and duration model.",
        references, [view("Paired changes", ["Model or comparison","Stage","Trial group","Participant","Accuracy change (pp)","95% lower","95% upper"],
        comparison_rows, [3,0,1,2], default_chart(1,4,0,5,6), 0, scope, stages,
        initial={"Participant":cohort, "Trial group":"Switch"}, selectors=(3,2))],
        [("Paired accuracy changes across all participants", f"{full}/full_stage_comparison.png")],
        terms=(("Paired change", "For each participant, subtract accuracy with the 64 ms crop from accuracy with the full stage on the same trials. The EEG + duration comparison subtracts accuracy with duration alone."),
               ("Percentage points (pp)", "An increase from 40% to 55% is a gain of 15 percentage points.")))
    return [first, second]


def default_chart(category,value,series=None,low=None,high=None,context_only=()):
    return dict(category=category,value=value,series=series,low=low,high=high,contextOnly=context_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source / "outputs" / "reports"
    output.mkdir(parents=True,exist_ok=True)
    template = Path(__file__).with_name("web_report.html").read_text(encoding="utf-8")
    categories = [dict(slug=slug,title=title,summary=summary) for slug,title,summary in CAT]
    def emit(path, text, payload):
        encoded = json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(",",":")).replace("<",r"\u003c")
        path.parent.mkdir(parents=True,exist_ok=True)
        rendered = template.replace("<!--TITLE-->",html.escape(text)).replace("<!--DATA-->",html.escape(encoded, quote=False))
        path.write_text(rendered,encoding="utf-8")
    reports = build_data(args.source)
    for cat, studies in zip(categories,reports):
        emit(output / cat["slug"] / "index.html",cat["title"],dict(categories=categories,category=cat,studies=studies))
    emit(output / "index.html","EEG-to-Kinematics · Interactive results",dict(categories=categories))
    print(f"Generated {len(categories)} family reports and one index in {output}")


if __name__ == "__main__":
    main()
