# T7 — Wrist position and velocity decoding

**Interactive results:** [Open online](https://htmlpreview.github.io/?https://github.com/neurocontrol-lab/eeg-kinematics-decoding/blob/main/outputs/reports/movement-targets/index.html) · [Open locally](../../outputs/reports/movement-targets/index.html)

[Protocol](../../documentation/methods/POSITION_VELOCITY.md) · [Experiment registry](../../documentation/EXPERIMENT_REGISTRY.md)

Run from the repository root in the existing scientific Python environment
with a TensorFlow GPU. Use the verified T1 population outputs and their
local weights and indices as inputs.

```sh
python experiments/09_movement_targets/run_position_velocity.py --output-dir outputs/position-velocity-p1-p12-<timestamp>
```

Repeat the same command to resume an interrupted run. The runner checkpoints
individual fits, freezes all validation selections, verifies the held-out
predictions in isolated workers and writes a participant-level report.

