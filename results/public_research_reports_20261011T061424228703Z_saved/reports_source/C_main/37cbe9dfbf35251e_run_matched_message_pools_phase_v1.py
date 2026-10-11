from pathlib import Path
import argparse,datetime,json,subprocess,sys,os
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['precheck','execute'],required=True);a=p.parse_args()
argv=[sys.executable,'-u',str(a.root/'diagnose_matched_message_pools_v1.py'),'--plan',str(a.root/'plan.json'),'--phase',a.phase]
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
with (a.root/(a.phase+'.log')).open('x') as handle:
    child=subprocess.Popen(argv,stdout=handle,stderr=subprocess.STDOUT)
    proc=(Path('/proc')/str(child.pid)/'cmdline').read_bytes().decode().split('\0')[:-1]
    (a.root/(a.phase+'_launch.json')).write_text(json.dumps(dict(utc=now(),wrapper_pid=os.getpid(),child_pid=child.pid,argv=argv,actual_proc_argv=proc),indent=2))
    code=child.wait()
(a.root/(a.phase+'_exit.json')).write_text(json.dumps(dict(utc=now(),child_pid=child.pid,exit_code=code),indent=2))
print('MATCHED_POOLS_NATURAL_EXIT',a.phase,code,flush=True)
raise SystemExit(code)
