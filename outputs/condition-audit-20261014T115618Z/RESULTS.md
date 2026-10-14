# C1: trial-condition audit

P1–P12 metadata, with named columns and trial-level current/previous conditions.
All P1 WS trials verify physical weight strings and surface-code labels. Numeric surface codes are not series types.

| Participant | Run types 1–6 | Runs 7–9 | V2 held-out joint conditions absent from training |
|---|---|---|---|
| P1 | weight, friction, mixed, weight, mixed, friction | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P2 | mixed, friction, weight, friction, weight, mixed | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P3 | weight, mixed, friction, mixed, friction, weight | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P4 | friction, mixed, weight, mixed, weight, friction | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P5 | mixed, weight, friction, weight, friction, mixed | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P6 | weight, friction, mixed, weight, mixed, friction | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P7 | friction, weight, mixed, friction, mixed, weight | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P8 | mixed, friction, weight, friction, weight, mixed | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P9 | weight, mixed, friction, mixed, friction, weight | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P10 | friction, mixed, weight, mixed, weight, friction | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P11 | mixed, weight, friction, weight, friction, mixed | Weight, fixed sandpaper | {'validation': [], 'test': []} |
| P12 | weight, friction, mixed, weight, mixed, friction | Weight, fixed sandpaper | {'validation': [], 'test': []} |

## C2 setup supported by the audit

P1 weight runs 1 and 4 train (68 trials), run 7 validates (34), runs 8–9 test (68). Every selected run contains all three weights at fixed sandpaper.
Weight varies between trials within these runs. Previous conditions and repeated/changed flags are retained in trial_conditions.csv.
The v2 training split mixes series types while held-out runs 7–9 are weight series. Matching joint conditions alone does not make their task contexts identical.
All trial conditions and per-run counts are in results.json and trial_conditions.csv. Existing model artifacts are untouched.
