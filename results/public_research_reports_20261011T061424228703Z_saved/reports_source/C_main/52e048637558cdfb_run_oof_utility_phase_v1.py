from pathlib import Path
import argparse,datetime,json,subprocess,sys,os
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['precheck','execute'],required=True);a=p.parse_args()
q=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
argv=[sys.executable,'-u',str(a.root/'diagnose_oof_teacher_utility_v1.py'),'--plan',str(a.root/'plan.json'),'--phase',a.phase]
log=a.root/(a.phase+'.log')
with log.open('x') as handle:
 child=subprocess.Popen(argv,stdout=handle,stderr=subprocess.STDOUT)
 (a.root/(a.phase+'_launch.json')).write_text(json.dumps(dict(utc=q(),wrapper_pid=os.getpid(),child_pid=child.pid,argv=argv,actual_proc_argv=(Path('/proc')/str(child.pid)/'cmdline').read_bytes().decode().split('\0')[:-1]),indent=2))
 code=child.wait()
(a.root/(a.phase+'_exit.json')).write_text(json.dumps(dict(utc=q(),child_pid=child.pid,exit_code=code),indent=2))
print('PHASE_NATURAL_EXIT',a.phase,code,flush=True)
sys.exit(code)
