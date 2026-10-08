from pathlib import Path
import argparse,datetime,json,subprocess,sys,time,os
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
argv=[sys.executable,'-u',str(a.root/'diagnose_native_radius_cal_v1.py'),'--plan',str(a.root/'plan.json')]
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
with (a.root/'mechanism.log').open('x') as f:
    child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT);t=time.monotonic();actual=[]
    while time.monotonic()-t<2:
        actual=(Path('/proc')/str(child.pid)/'cmdline').read_bytes().decode().split('\0')[:-1]
        if actual==argv:break
        time.sleep(.01)
    assert actual==argv
    (a.root/'mechanism_launch.json').write_text(json.dumps(dict(utc=now(),wrapper_pid=os.getpid(),child_pid=child.pid,argv=argv,actual_proc_argv=actual),indent=2));code=child.wait()
(a.root/'mechanism_exit.json').write_text(json.dumps(dict(utc=now(),child_pid=child.pid,exit_code=code),indent=2))
print('CAL_NATIVE_RADIUS_NATURAL_EXIT',code,flush=True);raise SystemExit(code)
