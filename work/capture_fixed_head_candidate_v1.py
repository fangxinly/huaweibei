"""Completed original GPU/CPU candidate collection capsule, no model execution."""
import argparse,datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--dest',required=True);p.add_argument('--kind',choices=['gpu','cpu'],required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
root=Path(a.root).resolve();dest=Path(a.dest).resolve();assert root.parent==Path('/data/coding') and root.name.startswith('fixed_head_candidate_')
assert dest.parent==Path('/data/coding') and dest.name.startswith('capture_fixed_head_candidate_') and not dest.exists()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
receipt_path=root/('out/actual_candidate_collection_receipt.json' if a.kind=='gpu' else 'cpu_original_receipt.json')
exit_path=root/('natural_exit.json' if a.kind=='gpu' else 'cpu_original_receipt.actual_exit.json')
r=read(receipt_path);e=read(exit_path);assert e['exit_code']==0 and e['natural_wait_verified'] and e['child_pid']==r['pid'] and e['child_full_argv'][1:]==r['argv'] and e['original_receipt_sha256']==sha(receipt_path)
checkpoint=Path(a.checkpoint).resolve();plan=read(root/'source/candidate_collection_plan.json');assert sha(checkpoint)==plan['checkpoint_file_sha256']
dest.mkdir(exist_ok=False);small={};large={}
paths=[('run/'+f.relative_to(root).as_posix(),f) for f in sorted(root.rglob('*')) if f.is_file() and '__pycache__' not in f.parts]
paths += [('reference/selected_best_full.pt',checkpoint),('capture_tool/'+Path(__file__).name,Path(__file__).resolve())]
for n,f in paths:
 b=f.stat();h=sha(f);c=f.stat();assert (b.st_ino,b.st_dev,b.st_size,b.st_mtime_ns)==(c.st_ino,c.st_dev,c.st_size,c.st_mtime_ns)
 item={'original_path':str(f),'bytes':b.st_size,'sha256':h,'inode':b.st_ino,'device':b.st_dev,'mtime_ns':b.st_mtime_ns,'stable_before_after':True}
 (small if b.st_size<=16*1024**2 else large)[n]=item
manifest={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'kind':a.kind,'small_members':small,'large_references_not_downloads':large,'original_receipt_sha256':sha(receipt_path),'original_exit_sha256':sha(exit_path),'task_labels_read':False}
mf=dest/'member_manifest.json';mf.write_text(json.dumps(manifest,indent=2)+'\n');pkg=dest/'snapshot.zip'
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for n,item in small.items():z.write(item['original_path'],n)
 z.write(mf,'member_manifest.json')
with zipfile.ZipFile(pkg) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,item in small.items():assert hashlib.sha256(z.read(n)).hexdigest()==item['sha256']
record={'status':'ACTUAL_COMPLETED_HEAD_CANDIDATE_'+a.kind.upper()+'_CAPTURE_COMPLETE','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),'members':len(small)+1,'snapshot_sha256':sha(pkg),'snapshot_bytes':pkg.stat().st_size,'large_references':large,'reference_full_file_new_download_this_capture':False,'no_task_labels_or_new_fit':True}
(dest/'capture_receipt.json').write_text(json.dumps(record,indent=2)+'\n');print('CAPTURE_COMPLETE '+json.dumps(record),flush=True)
