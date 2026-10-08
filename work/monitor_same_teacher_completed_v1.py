from pathlib import Path
import argparse, datetime, hashlib, json, shutil, subprocess
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
read=lambda p:json.loads(Path(p).read_text())
plan=read(a.root/'same_teacher_collection_plan_v2.json')
exits={phase:read(a.root/(phase+'_exit.json')) for phase in ['precheck','execute']}
assert all(e['exit_code']==0 for e in exits.values())
for e in exits.values():assert not (Path('/proc')/str(e['child_pid'])).exists()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip()
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
fold=exits['execute']['fold'];assert gpu.split(',')[0].strip()==plan['folds'][fold]['expected_uuid'] and not compute
processes=[]
for child in Path('/proc').iterdir():
    if not child.name.isdigit():continue
    try:argv=(child/'cmdline').read_bytes().decode().split('\0')[:-1]
    except (OSError,UnicodeError):continue
    if argv:processes.append({'pid':int(child.name),'argv':argv})
out=a.root/'completed_monitor.json';assert not out.exists()
out.write_text(json.dumps({'status':'ACTUAL_COLLECTION_BOTH_PHASES_NATURAL_EXIT0_RETIRED_GPU_EMPTY',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fold':fold,'gpu':gpu,'compute':compute,
    'exits':exits,'processes_full_argv':processes,'remote_free_bytes':shutil.disk_usage(a.root).free,
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
print('ACTUAL_COLLECTION_COMPLETED_MONITOR',fold,flush=True)
