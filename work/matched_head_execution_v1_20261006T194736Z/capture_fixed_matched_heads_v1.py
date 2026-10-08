"""Future complete stage capsule: original supervision is retained honestly."""
import argparse,datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
p=argparse.ArgumentParser()
for n in ['root','dest','kind']:p.add_argument('--'+n,required=True)
p.add_argument('--checkpoint');a=p.parse_args();root=Path(a.root).resolve();dest=Path(a.dest).resolve()
assert root.parent==Path('/data/coding') and root.name.startswith('fixed_head_matched_') and dest.parent==Path('/data/coding') and dest.name.startswith('capture_fixed_head_matched_') and not dest.exists()
choices={'fit':'out/actual_matched_head_fit_receipt.json','predict':'out/actual_candidate_collection_receipt.json','score':'out/actual_matched_head_evaluation_receipt.json','cpu':'cpu_original_receipt.json'}
assert a.kind in choices
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
receipt=root/choices[a.kind];ex=root/'natural_exit.json';r=read(receipt);e=read(ex);plan=read(root/'source/matched_head_execution_plan.json')
assert plan['status']=='COMPLETE_MATCHED_HEAD_EXECUTION_PROTOCOL_FROZEN' and e['exit_code']==0 and e['natural_wait_verified'] and e['child_pid']==r['pid'] and e['child_full_argv'][1:]==r['argv'] and e['original_receipt_sha256']==sha(receipt)
assert r['source_sha256']==plan['source_and_role_sha256'][Path(r['argv'][0]).name] and sha(__file__)==plan['source_and_role_sha256'][Path(__file__).name]
for n,h in plan['source_and_role_sha256'].items():assert sha(root/'source'/n)==h
dest.mkdir(exist_ok=False);small={};large={};paths=[('run/'+p.relative_to(root).as_posix(),p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
paths.append(('capture_tool/'+Path(__file__).name,Path(__file__).resolve()))
if a.checkpoint:
 checkpoint=Path(a.checkpoint).resolve();assert sha(checkpoint)==plan['checkpoint_file_sha256'];paths.append(('reference/selected_best_full.pt',checkpoint))
for n,path in paths:
 b=path.stat();h=sha(path);c=path.stat();assert (b.st_ino,b.st_dev,b.st_size,b.st_mtime_ns)==(c.st_ino,c.st_dev,c.st_size,c.st_mtime_ns)
 item={'original_path':str(path),'bytes':b.st_size,'sha256':h,'inode':b.st_ino,'device':b.st_dev,'mtime_ns':b.st_mtime_ns,'stable_before_after':True};(small if b.st_size<=16*1024**2 else large)[n]=item
manifest={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'kind':a.kind,'small_members':small,'large_references_not_downloads':large,'original_receipt_sha256':sha(receipt),'original_exit_sha256':sha(ex),'capture_tool_indexes_no_new_task_labels':True,'supervised_FIT_or201_original_arrays_retained_if_present':True}
mf=dest/'member_manifest.json';mf.write_text(json.dumps(manifest,indent=2)+'\n');pkg=dest/'snapshot.zip'
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for n,item in small.items():z.write(item['original_path'],n)
 z.write(mf,'member_manifest.json')
with zipfile.ZipFile(pkg) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,item in small.items():assert hashlib.sha256(z.read(n)).hexdigest()==item['sha256']
record={'status':'ACTUAL_COMPLETED_MATCHED_HEAD_'+a.kind.upper()+'_CAPTURE_COMPLETE','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),'kind':a.kind,'members':len(small)+1,'snapshot_sha256':sha(pkg),'snapshot_bytes':pkg.stat().st_size,'large_references':large,'full_checkpoint_new_download_this_capture':False,'original_receipt_sha256':sha(receipt),'original_exit_sha256':sha(ex),'plan_sha256':sha(root/'source/matched_head_execution_plan.json')}
(dest/'capture_receipt.json').write_text(json.dumps(record,indent=2)+'\n');print('CAPTURE_COMPLETE '+json.dumps(record),flush=True)
