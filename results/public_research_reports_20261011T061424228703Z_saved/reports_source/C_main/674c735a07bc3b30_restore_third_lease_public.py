"""Fresh public assets + private offline runtime; no task training/label decode."""
import argparse,datetime,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--node',choices=['A','B','C'],required=True);a=p.parse_args()
plan=json.loads(a.plan.read_text());node=plan['nodes'][a.node]
root=Path(plan['remote_record_prefix']+'_'+a.node);root.mkdir()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
write=lambda p,r:p.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
assert sha(__file__)==plan['restore_source_SHA']
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==node['uuid']
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert not compute.strip()
write(root/'physical_preflight.json',dict(actual_utc=utc(),node=a.node,uuid=node['uuid'],compute=compute,
 fullargv=[sys.executable]+sys.argv,pid=os.getpid(),source_SHA=sha(__file__),space=shutil.disk_usage('/data')._asdict(),
 processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)))
assets=Path(plan['public_root']);assert not assets.exists();assets.mkdir()
common=Path('/data/coding/public_common_assets_9ba39814.zip');assert sha(common)==plan['public_archive_SHA']
with zipfile.ZipFile(common) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n in z.namelist():
  if not n.startswith('assets/'):continue
  target=(assets/n).resolve();assert target.is_relative_to(assets.resolve());target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(z.read(n))
for n,h in plan['asset_sha256'].items():assert sha(assets/n)==h
write(root/'public_assets_verified.json',dict(actual_utc=utc(),archive_SHA=plan['public_archive_SHA'],
 public_root=str(assets),asset_sha256=plan['asset_sha256'],only_public_assets_extracted=True,task_weights_restored=False,labels_decoded=False))
wheels=Path('/data/coding/public_linux24_wheels_735be363.zip');assert sha(wheels)==plan['wheel_archive_SHA']
wheelroot=root/'wheels';wheelroot.mkdir()
with zipfile.ZipFile(wheels) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n in z.namelist():
  target=(wheelroot/n).resolve();assert target.is_relative_to(wheelroot.resolve());target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(z.read(n))
manifest=json.loads((wheelroot/'wheel_manifest.json').read_text())
assert json.loads((wheelroot/'linux_dependency_closure.json').read_text())['complete']
for n,m in manifest['wheel_files'].items():assert sha(wheelroot/n)==m['sha256']
venv=assets/'.venv';subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(venv)],check=True)
native=str(venv/'bin/python')
argv=[native,'-m','pip','install','--no-index','--find-links',str(wheelroot),'--no-deps','-r',str(wheelroot/'offline_requirements.txt')]
with (root/'dependency_stdout.log').open('wb') as out,(root/'dependency_stderr.log').open('wb') as err:
 child=subprocess.Popen(argv,stdout=out,stderr=err);write(root/'dependency_child.json',dict(actual_start_utc=utc(),pid=child.pid,fullargv=argv));code=child.wait()
write(root/'dependency_natural_exit.json',dict(actual_utc=utc(),pid=child.pid,fullargv=argv,natural_exit=code));assert code==0
check="import importlib.metadata as im,json,torch,numpy,transformers,sentencepiece,scipy,sklearn;print(json.dumps({n:im.version(n) for n in "+repr(list(plan['runtime_exact_versions']))+"}))"
versions=json.loads(subprocess.check_output([native,'-c',check],text=True));assert versions==plan['runtime_exact_versions']
write(root/'runtime_verified.json',dict(actual_utc=utc(),fullargv=[native,'-c',check],versions=versions,imports_natural_exit=0,private_venv=True,global_environment_changed=False))
shutil.copy2(__file__,root/Path(__file__).name);shutil.copy2(a.plan,root/a.plan.name)
files=[q for q in root.iterdir() if q.is_file()]
write(root/'capture_manifest.json',dict(actual_utc=utc(),fullargv=[sys.executable]+sys.argv,source_SHA=sha(__file__),
 member_sha256={q.name:sha(q) for q in files},scope='Public recovery only, no model training or label decode'))
archive=root/'complete_public_restore_capture.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for q in files+[root/'capture_manifest.json']:z.write(q,q.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
receipt=dict(status='THIRD_LEASE_PUBLIC_ASSETS_PRIVATE_RUNTIME_COMPLETE',actual_utc=utc(),node=a.node,uuid=node['uuid'],
 archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,pip_natural_exit=code,no_scientific_training_started=True)
write(root/'capture_receipt.json',receipt);print(json.dumps(receipt),flush=True)
