"""New root capture, old capture19 kept unchanged; no legacy dummy assets."""
import argparse,hashlib,json,os,shutil,subprocess,zipfile
from pathlib import Path
from datetime import datetime,timezone
p=argparse.ArgumentParser();p.add_argument('--asset-root',required=True);p.add_argument('--bundle',required=True);p.add_argument('--run-root',required=True);p.add_argument('--stamp',required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
asset=Path(a.asset_root).resolve();bundle=Path(a.bundle).resolve();run=Path(a.run_root).resolve()
for r in (asset,bundle,run):assert r.is_relative_to(Path('/data/coding').resolve()) and r.is_dir()
out=run.parent/('capture_'+a.stamp);out.mkdir(exist_ok=False)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
q=lambda cmd:subprocess.check_output(cmd,text=True).strip()
actual=q(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']);assert actual==a.expected_uuid
processes=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:args=(proc/'cmdline').read_bytes().decode().split('\0')[:-1]
 except (OSError,UnicodeError):continue
 if any(str(bundle) in x or str(run) in x for x in args):processes.append({'pid':int(proc.name),'full_argv':args})
inv={'actual_utc':datetime.now(timezone.utc).isoformat(),'gpu_uuid':actual,'compute':q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),'full_matching_processes':processes,'launch':json.loads((run/'launch.json').read_text()),'natural_exit':json.loads((run/'natural_exit.json').read_text()) if (run/'natural_exit.json').exists() else None,'asset_root':str(asset),'bundle':str(bundle),'run_root':str(run),'free_bytes':shutil.disk_usage('/data').free,'capture_source_sha256':sha(__file__),'parent_capture19_sha256':sha(bundle/'capture_soft_vector_v19.py'),'old_capture19_executed_on_new_root':False}
assert inv['parent_capture19_sha256']=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
members={};large={};pending=out/'snapshot.pending';final=out/'snapshot.zip'
with zipfile.ZipFile(pending,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 def add_bytes(data,name):
  if name in members:raise ValueError('DUPLICATE_CAPTURE_MEMBER')
  z.writestr(name,data);members[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 def add_file(path,name):
  if path.is_symlink():raise ValueError('NO_SYMLINK_CAPTURE_SOURCE')
  if path.stat().st_size>8*1024**2:
   with path.open('rb') as f:
    before=os.fstat(f.fileno());h=hashlib.sha256()
    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    after=os.fstat(f.fileno())
   if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('LARGE_OPEN_INODE_CHANGED')
   large[name]={'path':str(path),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'sha256':h.hexdigest(),'only_SHA_reference_not_fullweight_download':True}
  else:add_bytes(path.read_bytes(),name)
 add_bytes(json.dumps(inv,indent=2).encode(),'inventory.json')
 for folder,prefix in ((bundle,'source'),(run,'run')):
  for path in sorted(folder.rglob('*')):
   if not path.is_file() or '__pycache__' in path.parts or path.name.endswith('.pending'):continue
   if path.suffix in ('.py','.json','.npz','.npy','.pt','.log','.txt'):add_file(path,prefix+'/'+path.relative_to(folder).as_posix())
 for path in sorted(asset.glob('*')):
  if path.is_file() and path.suffix in ('.json','.log'):add_file(path,'asset_records/'+path.name)
 runtime=json.loads((bundle/'runtime_candidate_plan.json').read_text())
 for name,digest in runtime['asset_sha256'].items():
  path=asset/name;assert sha(path)==digest;add_file(path,'public_assets/'+name)
 required={'source/second_lease_precheck_plan.json','source/deployment_execution_plan_v1.json','source/second_lease_precheck_wrapper_v2.py','source/precheck_minimal_fixed_v4.py','source/precheck_gates_v2.py','source/donor_terminal_mechanism_precheck_v1.py','source/audit_second_lease_full_precheck_cpu_v1.py','source/capture_new_fixed_precheck_v21.py','run/launch.json'}
 assert required.issubset(members)
 add_bytes(json.dumps(large,indent=2).encode(),'large_file_manifest.json')
 add_bytes(json.dumps(members,indent=2).encode(),'member_manifest.json')
pending.rename(final)
with zipfile.ZipFile(final) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(members)
 for name,m in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==m['sha256']
receipt={'status':'ACTUAL_NEW_FIXED_ROOT_CAPTURE_COMPLETE','actual_utc':datetime.now(timezone.utc).isoformat(),'sha256':sha(final),'bytes':final.stat().st_size,'members':len(members),'large_references':len(large),'source_and_new_root_and_launch_argv_covered':True,'capture_source_sha256':sha(__file__),'old_capture19_unchanged_parent_reference':True,'large_weights_downloaded':False}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('CAPTURE_COMPLETE '+json.dumps(receipt),flush=True)
