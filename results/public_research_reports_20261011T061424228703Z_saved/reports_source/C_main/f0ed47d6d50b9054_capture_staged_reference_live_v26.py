"""Actual live small-file capture; this is not an intermediate model checkpoint."""
import argparse,datetime,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--bundle',required=True);p.add_argument('--preservation-evidence',required=True);p.add_argument('--dest',required=True);a=p.parse_args()
run=Path(a.run).resolve();bundle=Path(a.bundle).resolve();dest=Path(a.dest).resolve();evidence=Path(a.preservation_evidence).resolve()
assert str(run).startswith('/data/coding/minimal_fixed_fold0_continue100_actual_')
assert str(bundle).startswith('/data/coding/minimal_fixed_staged_reference_')
assert dest.parent==Path('/data/coding') and dest.name.startswith('capture_staged_reference_live_')
assert (run/'actual_child_launch.json').is_file() and (run/'out/actual_training_start.json').is_file()
assert json.loads(evidence.read_text())['status']=='ACTUAL_SHARED10_GPU_D_B_CPU_NEXT_UPDATE_AND_FRESH_REPLAY_JOINT_PASSED'
dest.mkdir(exist_ok=False)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
members={};large={}
for label,root in (('run',run),('source',bundle)):
 for path in sorted(root.rglob('*')):
  if not path.is_file() or '__pycache__' in path.parts:continue
  before=path.stat();digest=sha(path);after=path.stat()
  if (before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns):raise RuntimeError('LIVE_FILE_CHANGED_DURING_CAPTURE '+str(path))
  record={'original_path':str(path),'sha256':digest,'bytes':before.st_size,'inode':before.st_ino,'device':before.st_dev,'mtime_ns':before.st_mtime_ns,'stable_before_after':True}
  name=label+'/'+path.relative_to(root).as_posix()
  if before.st_size>16*1024**2:large[name]=record
  else:members[name]=record
for name,path in (('preservation_inputs/'+evidence.name,evidence),('capture_tool/'+Path(__file__).name,Path(__file__).resolve())):
 before=path.stat();digest=sha(path);after=path.stat()
 assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 members[name]={'original_path':str(path),'sha256':digest,'bytes':before.st_size,'inode':before.st_ino,'device':before.st_dev,'mtime_ns':before.st_mtime_ns,'stable_before_after':True}
start=json.loads((run/'out/actual_training_start.json').read_text());assert start['start_epoch']==10 and start['optimizer_steps']==220 and start['phase']=='continue100'
launch=json.loads((run/'actual_child_launch.json').read_text())
progress=json.loads((run/'out/progress.json').read_text()) if (run/'out/progress.json').is_file() else None
get=lambda args:subprocess.check_output(args,text=True).strip()
inventory={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':get(['nvidia-smi','--query-gpu=uuid,name,memory.used','--format=csv,noheader']),'compute':get(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader']),'full_process_argv':get(['ps','-eo','pid,args','--width','2000'])}
manifest={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'run':str(run),'source':str(bundle),'capture_source_sha256':sha(__file__),'small_members':members,'large_references_not_downloads':large,'actual_inventory':inventory,'actual_child_launch':launch,'resumed_from_epoch':start['start_epoch'],'resumed_from_steps':start['optimizer_steps'],'progress_at_capture':{k:v for k,v in progress.items() if k!='history'} if progress else None,'new_source_and_run_and_full_argv_covered':True,'formal100_complete':False,'scope':'Live small files and process evidence, not an intermediate model/Adam/RNG checkpoint or training completion; files stable individually, not an atomic global model state.'}
mf=dest/'member_manifest.json';mf.write_text(json.dumps(manifest,indent=2)+'\n')
pkg=dest/'snapshot.zip'
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for name,r in members.items():z.write(r['original_path'],name)
 z.write(mf,'member_manifest.json')
with zipfile.ZipFile(pkg) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for name,r in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256']
receipt={'status':'ACTUAL_STAGED_REFERENCE_LIVE_SMALL_FILES_CAPTURE_COMPLETE_NOT_MODEL_CHECKPOINT','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'capture_source_sha256':sha(__file__),'snapshot_sha256':sha(pkg),'snapshot_bytes':pkg.stat().st_size,'members':len(members)+1,'formal100_complete':False,'weights_downloaded_this_capture':False}
(dest/'capture_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('CAPTURE_COMPLETE '+json.dumps(receipt),flush=True)
