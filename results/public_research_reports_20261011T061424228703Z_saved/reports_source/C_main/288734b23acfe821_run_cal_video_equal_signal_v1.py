from pathlib import Path
import argparse,datetime,json,subprocess,sys,os
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['precheck','execute'],required=True);a=p.parse_args()
cmd=[sys.executable,str(a.root/'calibrate_video_equal_residual_v1.py'),'--root',str(a.root),'--phase',a.phase]
with (a.root/(a.phase+'.log')).open('x',encoding='utf-8') as f:
    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);code=child.wait()
receipt=dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wrapper_pid=os.getpid(),child_pid=child.pid,argv=cmd,exit_code=code)
runtime_path=a.root/a.phase/'runtime.json'
if runtime_path.exists():
    runtime=json.loads(runtime_path.read_text());assert runtime['pid']==child.pid and runtime['actual_python_argv']==cmd[1:]
    receipt['original_runtime_argv_and_pid_match']=True
(a.root/(a.phase+'_exit.json')).write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print('CAL_SIGNAL_ACTUAL_EXIT',a.phase,code,child.pid);raise SystemExit(code)
