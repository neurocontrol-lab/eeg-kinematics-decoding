"""Serialize T1 after T2 and bound RAM by resuming unchanged fits in small batches."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

def stamp():return datetime.now(timezone.utc).isoformat()

def read(path):return json.loads(path.read_text())

def write(path,value):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def other_history_jobs():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            if any(b'experiments/08_history_duration/' in a for a in args):found.append(int(p.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return found

def available_gib():
    values=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0])/1024**2

def completed_gaps(d):
    return sum('ridge' in g and len(g.get('cnn_elu',{}))==3 for p in d['participant_results'].values() for g in p['gaps'].values())

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();repo=a.repo.resolve();out=(repo/a.output_dir).resolve();assert out.parent==(repo/'outputs').resolve()
    status=out/'pipeline_status.json';log=out/'training.log';lock=out/'queue.lock'
    if lock.exists():
        old=int(lock.read_text())
        if Path(f'/proc/{old}').exists():raise RuntimeError(f'Queue process {old} already exists')
        lock.unlink()
    lock.write_text(str(os.getpid()))
    (out/'queue_source.py').write_bytes(Path(__file__).read_bytes())
    base={'experiment_id':'EXP-042','supervisor_pid':os.getpid(),'created_at':stamp(),'runtime_policy':'Wait for separate T2 jobs, then one newly completed gap per process. Resume unchanged source/protocol and checkpoints. No model or analysis changes.','queue_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def state(stage,**extra):write(status,{**base,'stage':stage,'updated_at':stamp(),**extra})
    def note(message):
        with log.open('a') as f:f.write(stamp()+' '+message+'\n')
        print(message,flush=True)
    env=dict(os.environ,TF_CPP_MIN_LOG_LEVEL='2',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MPLBACKEND='Agg')
    try:
        if not (out/'results.json').exists():raise RuntimeError('Expected existing checkpoint before queue resume')
        idle_since=None
        while read(out/'results.json')['status']!='complete':
            jobs=other_history_jobs();mem=available_gib()
            if jobs or mem<5:
                idle_since=None;state('waiting_for_T2_or_RAM',other_history_job_pids=jobs,available_ram_gib=round(mem,2),completed_gap_conditions=completed_gaps(read(out/'results.json')))
                note(f'WAITING other_history_jobs={jobs} available_RAM_GiB={mem:.2f}')
                time.sleep(30);continue
            if idle_since is None:idle_since=time.monotonic()
            if time.monotonic()-idle_since<30:time.sleep(5);continue
            before=completed_gaps(read(out/'results.json'))
            command=[sys.executable,'-u','experiments/07_timing_alignment/run_population_timing.py','--output-dir',str(a.output_dir),'--resume']
            note(f'RESUME with {before}/72 complete gap conditions')
            controlled=False
            with log.open('a',buffering=1) as stream:
                proc=subprocess.Popen(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
                state('training',child_pid=proc.pid,completed_gap_conditions=before)
                while proc.poll() is None:
                    time.sleep(.5)
                    d=read(out/'results.json');count=completed_gaps(d)
                    if count>before and d['status']!='complete':
                        # Atomic JSON checkpoint already contains the complete gap.
                        # Stop only this owned child before another gap retains RAM.
                        proc.send_signal(signal.SIGINT);controlled=True
                        try:proc.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            proc.terminate();proc.wait(timeout=15)
                        note(f'CHECKPOINT_BATCH_DONE {count}/72; child RAM released')
                        break
                code=proc.wait()
            if code!=0 and not controlled:
                state('failed',failed_stage='training',exit_code=code,completed_gap_conditions=completed_gaps(read(out/'results.json')))
                raise RuntimeError(f'Training exited {code}; inspect training.log')
            # Recheck T2/RAM before every new batch. No change to fitted choices.
            idle_since=None
        stages=[('saved_weight_verification','verify_population_timing.py'),('population_report','report_population_timing.py'),('population_summary_verification','verify_population_summary.py')]
        for stage,script in stages:
            state(stage);note('START '+stage)
            with log.open('a',buffering=1) as stream:
                code=subprocess.call([sys.executable,'-u',f'experiments/07_timing_alignment/{script}',str(a.output_dir)],cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
            if code:
                state('failed',failed_stage=stage,exit_code=code);raise RuntimeError(f'{stage} exited {code}; inspect training.log')
        assert read(out/'verification.json')['status']==read(out/'population_verification.json')['status']=='passed'
        state('verified',completed_gap_conditions=72,report='RESULTS.md',population_plot='meeting_summary.png')
        note('POPULATION_T1_VERIFIED')
    except BaseException as e:
        note('QUEUE_ERROR '+repr(e))
        if not status.exists() or read(status).get('stage')!='failed':state('failed',error=repr(e))
        raise
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()

if __name__=='__main__':main()
