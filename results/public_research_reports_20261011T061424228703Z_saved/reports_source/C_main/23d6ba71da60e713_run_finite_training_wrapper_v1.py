"""Persist actual child PID/argv and actual exit status, no SSH credentials."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
command=[sys.executable,str(a.root/'train_finite_task_risk_v1.py'),'--root',str(a.root),'--base-root','/data/coding/soft_vector_research_20261005T1220Z','--out',str(a.root/'run'),'--mode',a.mode,'--expected-uuid',a.expected_uuid]
with (a.root/'training.log').open('x') as f:
    child=subprocess.Popen(command,cwd=a.root,stdout=f,stderr=subprocess.STDOUT)
    launch=dict(pid=child.pid,wrapper_pid=os.getpid(),argv=command,mode=a.mode,expected_gpu_uuid=a.expected_uuid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (a.root/'launch.json').open('x') as out:json.dump(launch,out,indent=2)
    code=child.wait()
record=dict(exit_code=code,pid=child.pid,argv=command,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
pending=a.root/'exit.pending.json';pending.write_text(json.dumps(record,indent=2));pending.replace(a.root/'exit.json')
sys.exit(code)
