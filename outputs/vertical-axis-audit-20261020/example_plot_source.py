from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
root=Path(__file__).parent;out=root/'vertical-axis-audit'
a=SimpleNamespace(output=out)
w=loadmat('/mnt/c/Users/satya/OneDrive/Documents/[S. Nasuto] EEG-EMG-Kinematics/P1/WS_P1_S8.mat',squeeze_me=True,struct_as_record=False)['ws']
q=w.win[0];names=list(map(str,w.names.kin));k=q.kin.astype(float);t=(np.arange(len(k))+1)/500
import csv
with (out/'trial_displacements.csv').open() as f:r=next(row for row in csv.DictReader(f) if row['participant']=='1' and row['run']=='8' and row['trial']=='1')
for key in ['lift_time','lower_cue_time','replace_time','expected_mg_N']:r[key]=float(r[key])
example=(t,k,names,r)
source=(root/'verify_vertical_axis.py').read_text()
plot=source.split("plt.rcParams.update",1)[1].split("print(json.dumps",1)[0]
exec('plt.rcParams.update'+plot)
axs[2].get_legend().remove();axs[2].legend(ncol=4,loc='lower right',fontsize=10)
fig.savefig(out/'axis_check_example.png',dpi=160)
