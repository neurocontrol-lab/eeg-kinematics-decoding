"""Plot the balanced C_5 model comparison as separate scope panels."""
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = Path(sys.argv[1])
summary = json.loads((out/"summary.json").read_text(encoding="utf-8"))
labels = {"ridge":"Ridge","cnn_elu":"CNN + ELU","eegnet_regression":"EEGNet",
          "cnn_bilstm":"CNN–BiLSTM"}
colors = {"ridge":"#3e647f","cnn_elu":"#127681","eegnet_regression":"#784d84",
          "cnn_bilstm":"#a4552e","mean":"#687885"}
panels = (("Stage-specific",("Specialist","Stage mean")),("Full pooled",("Pooled","Pooled mean")),
          ("Sample-matched pooled",("Matched pooled",)))
fig, axes = plt.subplots(2,3,figsize=(14,7),layout="constrained",sharex="row")
for c,(name,scopes) in enumerate(panels):
    rows=[r for r in summary["balanced"] if r["scope"] in scopes]
    names=[labels.get(r["architecture"],r["scope"]) for r in rows]
    for j,metric in enumerate(("mse","r2")):
        ax=axes[j,c]
        vals=[r[metric] for r in rows]
        y=np.arange(len(rows))
        ax.barh(y,vals,color=[colors[r["architecture"]] for r in rows])
        ax.set_yticks(y,names)
        ax.invert_yaxis()
        ax.grid(axis="x",alpha=.2)
        ax.axvline(0,color="#5f7383",lw=.9)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
        for i,value in enumerate(vals):
            ax.annotate(f"{value:.3f}" if j==0 else f"{value:.2f}",(value,i),
                        xytext=(4 if value>=0 else -4,0),textcoords="offset points",
                        ha="left" if value>=0 else "right",va="center",fontsize=9)
        if j==0:
            ax.set_title(name,weight="bold")
        ax.set_xlabel("Trial-balanced standardized XYZ MSE" if j==0 else "Mean across stage-wise XYZ R²")
for j in range(2):
    lo,hi=axes[j,0].get_xlim()
    axes[j,0].set_xlim(lo-(hi-lo)*.13 if j else 0,hi+(hi-lo)*.14)
fig.suptitle("C_5 · Balanced scores across P1–P12 and five common stages",weight="bold")
fig.savefig(out/"balanced_model_scores.png",dpi=160,facecolor="white")
