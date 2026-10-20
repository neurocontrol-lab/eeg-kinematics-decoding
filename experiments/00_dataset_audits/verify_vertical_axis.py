"""Read-only raw-data audit: tracker Z versus plate force X during lift/hold."""
import argparse, csv, json, hashlib
from pathlib import Path
import sys
_repo = next(p for p in Path(__file__).resolve().parents if (p / "src/eeg_kinematics/paths.py").is_file())
sys.path.insert(0, str(_repo / "src"))
from eeg_kinematics.paths import RAW_DATA_DIR
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=RAW_DATA_DIR);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=True,parents=True)
records=[];example=None;source_files=[]
for participant in range(1,13):
    folder=a.data/f'P{participant}'
    p=loadmat(folder/f'P{participant}_AllLifts.mat',squeeze_me=True,struct_as_record=False)['P']; cols={str(k):i for i,k in enumerate(p.ColNames)}
    for run in range(1,10):
        path=folder/f'WS_P{participant}_S{run}.mat';source_files.append({'path':str(path),'size_bytes':path.stat().st_size})
        w=loadmat(path,squeeze_me=True,struct_as_record=False)['ws'];names=list(map(str,w.names.kin));idx={n.split(' - ')[0]:i for i,n in enumerate(names)}
        rows=p.AllLifts[p.AllLifts[:,cols['Run']]==run];wins=np.atleast_1d(w.win);assert len(rows)==len(wins)
        for j,(q,row) in enumerate(zip(wins,rows)):
            assert abs(q.trial_start_time-row[cols['StartTime']])<1e-7
            assert abs(q.LEDon-row[cols['LEDOn']])<.004 and abs(q.LEDoff-row[cols['LEDOff']])<.004
            k=q.kin.astype(float);t=(np.arange(len(k))+1)/500
            lift=row[cols['tLiftOff']];off=row[cols['LEDOff']];replace=row[cols['tReplace']]
            windows=[(lift-.25,lift-.05),(off-.6,off-.2),(replace+.2,replace+.4)]
            if not(np.isfinite([lift,off,replace]).all() and off-.6>lift and replace+.4<t[-1]):raise ValueError((participant,run,j,'invalid windows'))
            masks=[(t>=l)&(t<r) for l,r in windows];assert all(m.sum()>=95 for m in masks)
            r={'participant':participant,'run':run,'trial':j+1,'weight_g':{1:165,2:330,4:660}[int(row[cols['CurW']])],'lift_time':float(lift),'lower_cue_time':float(off),'replace_time':float(replace)}
            for sensor,label in [(1,'object'),(4,'wrist')]:
                xyz=k[:,[idx[f'P{d}{sensor}'] for d in 'xyz']]
                med=np.array([np.median(xyz[m],axis=0) for m in masks]);rise=med[1]-med[0];fall=med[2]-med[1]
                for d,up,dn in zip('xyz',rise,fall):r[f'{label}_{d}_hold_minus_prelift_cm']=float(up);r[f'{label}_{d}_postreplace_minus_hold_cm']=float(dn)
                r[f'{label}_dominant_axis']='xyz'[np.argmax(abs(rise))]
            fx=k[:,idx['FX1']]+k[:,idx['FX2']];lf=k[:,idx['LF']]
            assert np.max(np.abs(fx-lf))<2e-5
            baseline=t<=1
            for d in 'XYZ':
                v=k[:,idx[f'F{d}1']]+k[:,idx[f'F{d}2']]
                r[f'force_{d.lower()}_sum_hold_N']=float(np.median(v[masks[1]])-np.median(v[baseline]))
            r['load_force_hold_N']=float(np.median(lf[masks[1]]));r['expected_mg_N']=r['weight_g']/1000*9.80665
            records.append(r)
            if participant==1 and run==8 and j==0:example=(t,k,names,r)
        print('checked',participant,run,len(rows),flush=True)

def summarize(rows):
    out={'trials':len(rows)}
    for sensor in ['object','wrist']:
        out[sensor]={}
        for d in 'xyz':
            up=np.array([r[f'{sensor}_{d}_hold_minus_prelift_cm'] for r in rows]);down=np.array([r[f'{sensor}_{d}_postreplace_minus_hold_cm'] for r in rows])
            out[sensor][d]={'hold_minus_prelift_cm_median':float(np.median(up)),'hold_minus_prelift_cm_IQR':np.quantile(up,[.25,.75]).tolist(),'positive_lift_count':int((up>0).sum()),'postreplace_minus_hold_cm_median':float(np.median(down)),'negative_lower_count':int((down<0).sum()),'dominant_lift_displacement_count':sum(r[f'{sensor}_dominant_axis']==d for r in rows)}
    out['force_by_weight']={}
    for g in [165,330,660]:
        rs=[r for r in rows if r['weight_g']==g];out['force_by_weight'][g]={'n':len(rs),'sum_FX_hold_N_median':float(np.median([r['load_force_hold_N'] for r in rs])),'expected_mg_N':g/1000*9.80665}
    return out
summary={'method':{'baseline':'median in [tLiftOff-0.25,tLiftOff-0.05) s','hold':'median in [LEDOff-0.6,LEDOff-0.2) s','postreplace':'median in [tReplace+0.2,tReplace+0.4) s','clock':'WS trial-relative events and t=(index+1)/500. No continuous HS clock conversion needed. WS LED/start times checked against AllLifts.','position_units':'raw tracker cm, per dataset paper Figures 2 and hand-velocity cm/s; reduced model targets remain normalized','force_check':'derived LF must match FX1+FX2 per sample within 2e-5 N','caution':'lift/replace landmarks were derived using object Z; displacement corroborates author convention and sign but is not an independent inference of the axis.'},'all':summarize(records),'P1':summarize([r for r in records if r['participant']==1]),'P1_runs8_9':summarize([r for r in records if r['participant']==1 and r['run'] in [8,9]]),'participants':{f'P{p}':summarize([r for r in records if r['participant']==p]) for p in range(1,13)},'source_files':source_files,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(a.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
with (a.output/'trial_displacements.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
plt.rcParams.update({'font.size':11})
t,k,names,r=example;idx={n.split(' - ')[0]:i for i,n in enumerate(names)};fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
for ax,sensor,label in [(axs[0],1,'Object'),(axs[1],4,'Wrist')]:
    base=(t>=r['lift_time']-.25)&(t<r['lift_time']-.05)
    for d,col in zip('xyz',['#4477aa','#ee7733','#228833']):
        v=k[:,idx[f'P{d}{sensor}']];ax.plot(t,v-np.median(v[base]),label=d.upper(),color=col)
    ax.set_ylabel(label+' displacement\n(raw cm)');ax.legend(loc='upper left',ncol=3);ax.grid(alpha=.2)
for d,col in zip('XYZ',['#4477aa','#ee7733','#228833']):
    v=k[:,idx[f'F{d}1']]+k[:,idx[f'F{d}2']];v-=np.median(v[t<=1]);axs[2].plot(t,v,label='Sum F'+d,color=col)
axs[2].axhline(r['expected_mg_N'],color='black',ls=':',label='mg');axs[2].set_ylabel('Contact-plate forces\n(baseline-subtracted N)');axs[2].legend(ncol=4,loc='lower right',fontsize=10);axs[2].set_xlabel('Time within WS trial (s)')
for ax in axs:
    for tt,lab in [(r['lift_time'],'Lift-off'),(r['lower_cue_time'],'Lowering cue'),(r['replace_time'],'Set-down')]:ax.axvline(tt,color='#666666',ls='--',lw=1);ax.text(tt+.03,.95,lab,transform=ax.get_xaxis_transform(),va='top',fontsize=9)
    ax.set_xlim(1,t[-1]);ax.grid(alpha=.2)
fig.suptitle('P1, run 8, trial 1 (330 g): position Z rises, plate force X carries load')
fig.tight_layout();fig.savefig(a.output/'axis_check_example.png',dpi=160);plt.close(fig)
print(json.dumps({k:summary[k] for k in ['all','P1','P1_runs8_9']},indent=2),flush=True)
