# Coordinate system

Verified 20 October 2026 against the dataset authors' event detector, paper Figure 1, and all 3,528 supplied trials across P1–P12.

| Signal | Interpretation |
|---|---|
| `Px4`, `Py4` | Wrist position in the horizontal plane; anatomical forward/lateral orientation remains unresolved. |
| `Pz4` | Wrist vertical position, positive upward. Gravity points approximately toward negative position Z. |
| `Pz1` | Vertical object position, used by the authors to identify lift-off and set-down. |
| `FX1`, `FX2` | Contact-plate lifting/load force components; their sum is `LF`. |
| `FZ1`, `FZ2` | Contact-plate normal/grip components, in a different frame from tracker position. |

The model's third reduced wrist target is therefore **Z: vertical (+ upward)**. Reduced model targets remain normalized values, not raw centimetres. This axis verification does not recover the undocumented upstream filtering/scaling process.

Object Z rose and fell in all 3,528 checked trials. Median hold-minus-prelift displacement was 6.244 cm for the object and 5.229 cm for the wrist. The event landmarks were originally detected using object Z, so this is corroboration of the authors' convention and sign rather than an independent statistical discovery.

See the [full audit](../../outputs/vertical-axis-audit-20261020/REPORT.md), [numeric evidence](../../outputs/vertical-axis-audit-20261020/results.json), and [target provenance](KINEMATICS_PROVENANCE.md).

Sources: [authors' event detector](https://github.com/luciw/way-eeg-gal-utilities/blob/master/Utilities/WEEG_FindEvents.m) and [Luciw et al. dataset paper](https://doi.org/10.1038/sdata.2014.47), Figure 1. A force magnitude alone cannot establish the position gravity axis: grip force may exceed load force, and plate/world axes differ.
