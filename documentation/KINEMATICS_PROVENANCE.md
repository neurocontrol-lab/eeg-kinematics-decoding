# Identity of the three kinematic targets

The reduced `way_eeg_clean/kin_P1.mat` and `kin_P6.mat` matrices contain three
unlabelled columns. Direct, sample-aligned comparison with the original
WAY-EEG-GAL `HS` series identifies them as **wrist position X, Y, and Z**.
In the original files, the relevant named columns are `Px4`, `Py4`, and `Pz4`
(position sensor 4). Luciw et al. identify sensor 4 as the wrist tracker in
Figure 1 of [the dataset paper](https://doi.org/10.1038/sdata.2014.47).

The nine original `HS_Px_S1` through `HS_Px_S9` series were concatenated in
the order specified by each reduced `session_start_Px.mat`. All nine segment
lengths matched exactly for both P1 and P6. The following are Pearson
correlations at corresponding sample indices, using every sample:

| Participant | Reduced column | Original wrist channel | Pearson r |
|---|---:|---|---:|
| P1 | 1 | `Px4` (X) | 0.9999508 |
| P1 | 2 | `Py4` (Y) | 0.9999969 |
| P1 | 3 | `Pz4` (Z) | 0.9999709 |
| P6 | 1 | `Px4` (X) | 0.9999516 |
| P6 | 2 | `Py4` (Y) | 0.9999974 |
| P6 | 3 | `Pz4` (Z) | 0.9999763 |

For each axis, the wrist channel correlated more strongly with the reduced
column than the corresponding object, index-finger, or thumb position
channel. For example, P1 reduced X correlated 0.9999508 with `Px4` and
0.9973302 with the next closest X channel, `Px3`. The full four-sensor
comparison is in [the machine-readable results](kinematics_verification.json).
These comparisons establish sensor identity without relying on the notebook's
incorrect “glove” or “finger” labels.

## What the scaling evidence supports

Each reduced column spans exactly **0 to 1** in both participants. Its mean is
not zero and its standard deviation is not one, so the values are **not a
plain `(x - mean) / standard deviation` z-score**. Away from the first and
last 100 samples of each series, a fitted positive affine map from the
original wrist channel to the reduced column has mean absolute residuals of
only 0.000054–0.000076 on the reduced 0–1 scale. This is consistent with
axis-wise range scaling of a closely related or filtered wrist signal.

The transformation is **not exactly min–max scaling of the original raw
channels**: X and Z extrema in the reduced matrices do not correspond to the
original raw extrema. Edge samples differ more than interior samples. This
suggests additional processing, but these files do not identify its exact
filter, padding, or scaling sequence. Correlation alone cannot recover that
missing recipe. The notebook then applies its own global min–max transform,
which maps these 0–1 targets to [-1, 1] for model training.

Run [the verification script](verify_kinematics.py) from the repository root
with the original `P1/` and `P6/` directories alongside the repository:

```bash
python documentation/verify_kinematics.py
```

The script rewrites `kinematics_verification.json` with the sample counts,
channel comparisons, ranges, and approximate affine fits. It does not change
the underlying recordings.
