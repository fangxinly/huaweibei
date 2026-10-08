"""Actual completed CPU audit root capture, full-file references distinct from transfers."""
import argparse,datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--dest',required=True);a=p.parse_args()
root=Path(a.root).resolve();dest=Path(a.dest).resolve()
assert root.parent==Path('/data/coding') and (root.name.startswith('staged_reference_') or root.name=='staged_reference100_cpu_audit_20261006T184140Z')
assert dest.parent==Path('/data/coding') and dest.name.startswith('capture_staged_CPU_audit_')
r=json.loads((root/'cpu_original_receipt.json').read_text());ex=json.loads((root/'cpu_original_receipt.actual_exit.json').read_text())
assert ex['exit_code']==0 and ex['natural_wait_verified'] and ex['child_pid']==r['pid'] and ex['child_full_argv'][1:]==r['argv']
assert not r['GPU_used'] and not r['CPU_model_forward']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
assert ex['original_receipt_sha256']==sha(root/'cpu_original_receipt.json')
dest.mkdir(exist_ok=False);members={};large={}
paths=[('audit_root/'+p.relative_to(root).as_posix(),p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
paths.append(('capture_tool/'+Path(__file__).name,Path(__file__).resolve()))
for n,p in paths:
 before=p.stat();h=sha(p);after=p.stat();b=(before.st_ino,before.st_dev,before.st_size,before.st_mtime_ns)
 assert b==(after.st_ino,after.st_dev,after.st_size,after.st_mtime_ns)
 record={'original_path':str(p),'sha256':h,'bytes':before.st_size,'inode':before.st_ino,'device':before.st_dev,'mtime_ns':before.st_mtime_ns,'stable_before_after':True}
 (large if before.st_size>16*1024**2 else members)[n]=record
manifest={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'root':str(root),'small_members':members,'large_references_not_downloads':large,'original_CPU_receipt_sha256':sha(root/'cpu_original_receipt.json'),'original_CPU_natural_exit_sha256':sha(root/'cpu_original_receipt.actual_exit.json'),'capture_source_sha256':sha(__file__),'CPU_model_forward':False,'GPU_used':False}
mf=dest/'member_manifest.json';mf.write_text(json.dumps(manifest,indent=2)+'\n');pkg=dest/'snapshot.zip'
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for n,r in members.items():z.write(r['original_path'],n)
 z.write(mf,'member_manifest.json')
with zipfile.ZipFile(pkg) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,r in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==r['sha256']
receipt={'status':'ACTUAL_COMPLETED_ORIGINAL_CPU_AUDIT_CAPTURE_COMPLETE','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'snapshot_sha256':sha(pkg),'snapshot_bytes':pkg.stat().st_size,'members':len(members)+1,'capture_source_sha256':sha(__file__),'large_references':large,'CPU_model_forward':False,'GPU_used':False,'new_weight_download_this_capture':False}
(dest/'capture_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('CAPTURE_COMPLETE '+json.dumps(receipt),flush=True)
