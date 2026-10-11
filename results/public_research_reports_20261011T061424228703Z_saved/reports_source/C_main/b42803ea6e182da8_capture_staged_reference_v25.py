"""Completed new staged run/source capture with stable large file references."""
import argparse,datetime,hashlib,json,os,stat,sys,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--bundle',required=True);p.add_argument('--dest',required=True);a=p.parse_args()
run=Path(a.run).resolve();bundle=Path(a.bundle).resolve();dest=Path(a.dest).resolve()
assert str(run).startswith('/data/coding/minimal_fixed_fold0_') and str(bundle).startswith('/data/coding/minimal_fixed_staged_reference_')
assert dest.parent==Path('/data/coding') and dest.name.startswith('capture_staged_reference_')
exit=json.loads((run/'natural_exit.json').read_text());assert exit['natural_wait_verified']
if exit['exit_code']==0:assert (run/'out/actual_training_receipt.json').is_file()
dest.mkdir(exist_ok=False)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
members={};large={};states={}
for label,root in (('run',run),('source',bundle)):
 for path in sorted(root.rglob('*')):
  if not path.is_file() or '__pycache__' in path.parts:continue
  before=path.stat();digest=sha(path);after=path.stat()
  if (before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns):raise RuntimeError('FILE_CHANGED_DURING_CAPTURE '+str(path))
  record={'original_path':str(path),'sha256':digest,'bytes':before.st_size,'inode':before.st_ino,'device':before.st_dev,'mtime_ns':before.st_mtime_ns,'stable_before_after':True}
  name=label+'/'+path.relative_to(root).as_posix()
  if before.st_size>16*1024**2:large[name]=record
  else:members[name]=record
tool=Path(__file__).resolve();before=tool.stat();digest=sha(tool);after=tool.stat()
assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
members['capture_tool/capture_staged_reference_v25.py']={'original_path':str(tool),'sha256':digest,'bytes':before.st_size,'inode':before.st_ino,'device':before.st_dev,'mtime_ns':before.st_mtime_ns,'stable_before_after':True}
manifest={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'run':str(run),'source':str(bundle),'capture_source_sha256':sha(__file__),'natural_training_exit_code':exit['exit_code'],'training_complete100':exit['formal100_complete'],'small_members':members,'large_references_not_downloads':large,'new_source_and_run_and_full_argv_covered':True,'old_capture19_23_not_reexecuted':True}
mf=dest/'member_manifest.json';mf.write_text(json.dumps(manifest,indent=2)+'\n')
pkg=dest/'snapshot.zip'
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for name,r in members.items():z.write(r['original_path'],name)
 z.write(mf,'member_manifest.json')
with zipfile.ZipFile(pkg) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for name,r in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256']
receipt={'status':'ACTUAL_STAGED_REFERENCE_CAPTURE_COMPLETE','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'capture_source_sha256':sha(__file__),'snapshot_sha256':sha(pkg),'snapshot_bytes':pkg.stat().st_size,'members':len(members)+1,'large_references':large,'natural_training_exit_code':exit['exit_code'],'formal100_complete':exit['formal100_complete'],'weights_downloaded_this_capture':False}
(dest/'capture_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('CAPTURE_COMPLETE '+json.dumps(receipt),flush=True)
