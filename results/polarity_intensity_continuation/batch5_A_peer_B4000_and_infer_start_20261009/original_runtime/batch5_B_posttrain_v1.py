"""New N2 execution identity, unchanged original scientific inference core."""
import argparse,base64,datetime,hashlib,importlib.metadata,json,os,pathlib,shutil,subprocess,sys,zipfile
P=pathlib.Path
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(P(p).read_bytes())
def write(p,v):P(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def contained(root,name):
 p=(root/name).resolve();assert p.is_relative_to(root.resolve());return p
def guard(plan,root,stage,versions=False,assets=False):
 assert sha(__file__)==plan['coordinator_SHA']
 for n,h in plan['source_sha256'].items():assert sha(P(plan['bundle'])/n)==h,n
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True)
 assert uuid==plan['GPU_UUID']['N2'] and not compute.strip()
 mem=dict(x.split(':',1) for x in P('/proc/meminfo').read_text().splitlines());available=int(mem['MemAvailable'].split()[0])*1024
 assert available>=6*1024**3 and shutil.disk_usage('/data').free>=8*1024**3
 left=(datetime.datetime.fromisoformat(plan['conservative_lease_end_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert left>plan['stage_budget_seconds'][stage]+7200
 runtime={n:importlib.metadata.version(n) for n in plan['runtime_versions']} if versions else {'base_python':sys.version}
 if versions:assert runtime==plan['runtime_versions']
 if assets:
  for n,h in plan['asset_sha256'].items():assert sha(P(plan['assets_root'])/n)==h,n
 observation=dict(actual_UTC=utc(),node='N2',UUID=uuid,compute=compute,fullargv=[sys.executable]+sys.argv,pid=os.getpid(),available_RAM=available,space=shutil.disk_usage('/data')._asdict(),remaining_seconds=left,runtime=runtime,processes=subprocess.check_output(['ps','-eo','pid,ppid,stat,args','--width','10000'],text=True),source_and_assets_SHA_verified=assets,conservative_human_lease_not_platform_confirmed=True)
 write(root/('fresh_'+stage+'_guard_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'),observation)
 return observation
def child(root,name,argv):
 with (root/(name+'_stdout.log')).open('wb') as o,(root/(name+'_stderr.log')).open('wb') as e:
  p=subprocess.Popen(argv,stdout=o,stderr=e,stdin=subprocess.DEVNULL);write(root/(name+'_dispatch.json'),dict(actual_UTC=utc(),pid=p.pid,fullargv=argv));code=p.wait()
 write(root/(name+'_natural_exit.json'),dict(actual_UTC=utc(),pid=p.pid,fullargv=argv,natural_exit=code));assert code==0,(name,code)
def restore(plan,root):
 guard(plan,root,'restore');assets=P(plan['assets_root']);assert not assets.exists();assets.mkdir()
 package=P(plan['common_archive']);assert sha(package)==plan['common_archive_SHA']
 with zipfile.ZipFile(package) as z:
  assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(plan['common_member_SHA'])|{'asset_manifest.json'}
  assert {v['archive']:v['sha256'] for v in json.loads(z.read('asset_manifest.json'))['files']}==plan['common_member_SHA']
  for n,h in plan['common_member_SHA'].items():
   take=n in plan['asset_sha256'];dest=contained(assets,n);digest=hashlib.sha256()
   if take:dest.parent.mkdir(parents=True,exist_ok=True)
   output=dest.open('xb') if take else None
   try:
    with z.open(n) as stream:
     for b in iter(lambda:stream.read(8*1024**2),b''):
      digest.update(b)
      if output:output.write(b)
   finally:
    if output:output.close()
   assert digest.hexdigest()==h,n
 for n,h in plan['asset_sha256'].items():assert sha(assets/n)==h,n
 write(root/'public_assets_result.json',dict(actual_UTC=utc(),archive_SHA=sha(package),all_member_SHA_CRC_unique_passed=True,assets_SHA=plan['asset_sha256'],task_weights_not_restored=True,labels_not_decoded=True))
 package=P(plan['wheel_archive']);assert sha(package)==plan['wheel_archive_SHA'];wheels=root/'wheels';wheels.mkdir()
 with zipfile.ZipFile(package) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for n in z.namelist():
   dest=contained(wheels,n);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
 manifest=read(wheels/'wheel_manifest.json');assert read(wheels/'linux_dependency_closure.json')['complete']
 for n,v in manifest['wheel_files'].items():assert sha(wheels/n)==v['sha256']
 child(root,'venv_setup',[sys.executable,'-m','venv','--system-site-packages',str(assets/'.venv')])
 native=str(assets/'.venv/bin/python');child(root,'offline_install',[native,'-m','pip','install','--no-index','--find-links',str(wheels),'--no-deps','-r',str(wheels/'offline_requirements.txt')])
 check="import importlib.metadata as im,json,torch,numpy,transformers,sentencepiece,scipy,sklearn;print(json.dumps({n:im.version(n) for n in "+repr(list(plan['runtime_versions']))+"}))"
 actual=json.loads(subprocess.check_output([native,'-B','-c',check],text=True));assert actual==plan['runtime_versions']
 child(root,'original_candidate_synthetic',[native,'-B',str(P(plan['bundle'])/'qualify_candidate_posttrain_v1.py'),'--out',str(root/'candidate_synthetic_result.json')])
 write(root/'runtime_result.json',dict(status='NEW_BATCH5_N2_ORIGINAL_EXACT24_ISOLATED_RUNTIME_AND_PUBLIC_ASSETS_VERIFIED',actual_UTC=utc(),node='N2',UUID=plan['GPU_UUID']['N2'],interpreter=native,versions=actual,original_candidate_both_modes_synthetic_passed=True,global_runtime_not_changed=True,no_inference_training_scoring=True))
def infer(plan,root):
 assert sys.executable==str(P(plan['assets_root'])/'.venv/bin/python')
 gate=read(P(plan['restore_root'])/'capture_receipt.json');assert gate['natural_exit']==0 and gate['stage']=='restore'
 guard(plan,root,'infer',versions=True,assets=True)
 audit=read(plan['audit_result_path']);assert audit==plan['training_Release_B_CPU_gate']['peer_original_CPU'] and audit['all_Adam_steps']==4000 and audit['best_epoch']==92
 assert sha(plan['training_archive'])==plan['training_original_reference']['archive_SHA']
 inputroot=root/'input';final=inputroot/'out/complete_final_and_selected.pt';final.parent.mkdir(parents=True)
 with zipfile.ZipFile(plan['training_archive']) as z:
  name='out/complete_final_and_selected.pt';assert z.getinfo(name).file_size==plan['final_checkpoint_bytes']
  h=hashlib.sha256()
  with z.open(name) as source,final.open('xb') as target:
   for b in iter(lambda:source.read(8*1024**2),b''):h.update(b);target.write(b)
  assert h.hexdigest()==plan['training_original_reference']['member_SHA'][name]
 source=P(plan['verified_original_root'])/'out/selected_DEV_frozen_prediction.npz';assert sha(source)==plan['training_original_reference']['member_SHA']['out/selected_DEV_frozen_prediction.npz'];shutil.copyfile(source,inputroot/'out/selected_DEV_frozen_prediction.npz')
 write(root/'temporary_final_PT_verified.json',dict(actual_UTC=utc(),path=str(final),SHA=sha(final),original_full_archive_permanently_retained=plan['training_archive']))
 guard(plan,root,'infer',versions=True,assets=True)
 once=P(plan['inference_once_token']);assert not once.exists();once.write_text(plan['selected_state_SHA'])
 argv=[sys.executable,'-B',str(P(plan['bundle'])/'infer_official.py'),'--plan',plan['plan_path'],'--plan-sha',sha(plan['plan_path']),'--bundle',plan['bundle'],'--assets',plan['assets_root'],'--out',str(root/'out'),'--input-root',str(inputroot)]
 child(root,'original_B_official_inference',argv)
 result=read(root/'out/inference_result.json');assert result['selected_state_SHA']==plan['selected_state_SHA'] and result['candidate_mode']=='regression_aux' and result['scalar_VAL_TEST_true_labels_not_indexed'] and result['all_parameters_buffers_RNG_unchanged']
 assert sha(final)==plan['training_original_reference']['member_SHA']['out/complete_final_and_selected.pt'];final.unlink()
 write(root/'temporary_PT_removal_receipt.json',dict(actual_UTC=utc(),removed=str(final),only_new_temporary_PT_after_actual_success=True,original_training_archive_retained=plan['training_archive']))
def capture(plan,root,stage):
 assert not root.exists();root.mkdir();write(root/'coordinator_dispatch.json',dict(actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,stage=stage,source_SHA=sha(__file__)))
 try:
  (restore if stage=='restore' else infer)(plan,root);code=0
 except Exception as e:
  import traceback;write(root/'coordinator_failure.json',dict(actual_UTC=utc(),type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));code=1
 write(root/'natural_exit.json',dict(actual_UTC=utc(),pid=os.getpid(),natural_exit=code,fullargv=[sys.executable]+sys.argv,stage=stage))
 original=root/'original_source';original.mkdir();shutil.copyfile(__file__,original/P(__file__).name);shutil.copyfile(plan['plan_path'],original/'official_new_N2_inference_protocol.json')
 for n in plan['source_sha256']:
  p=contained(original/'bundle',n);p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P(plan['bundle'])/n,p)
 files=[p for p in root.rglob('*') if p.is_file() and not p.is_relative_to(root/'wheels') and p.stat().st_size<40*1024**2];rows=[];archive=root/'complete_actual_small_original.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(files):
   n=p.relative_to(root).as_posix();z.write(p,n);rows.append(dict(name=n,bytes=p.stat().st_size,sha256=sha(p)))
  z.writestr('member_manifest.json',json.dumps(rows,indent=2))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for v in rows:assert hashlib.sha256(z.read(v['name'])).hexdigest()==v['sha256']
 write(root/'capture_receipt.json',dict(status='NEW_BATCH5_N2_'+stage.upper()+'_ACTUAL_CAPTURED',actual_UTC=utc(),stage=stage,node='N2',natural_exit=code,archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,all_member_SHA_CRC_unique_passed=True,large_public_archives_assets_and_training_archive_permanently_retained=True))
 return code
def qualify(root):
 root.mkdir();good=contained(root,'a/b.json');assert good.is_relative_to(root.resolve())
 try:contained(root,'../escape')
 except AssertionError:pass
 else:raise AssertionError('illegal path accepted')
 a=root/'synthetic.zip';data=b'synthetic only'
 with zipfile.ZipFile(a,'x') as z:z.writestr('x',data)
 with zipfile.ZipFile(a) as z:
  assert z.testzip() is None and hashlib.sha256(z.read('x')).hexdigest()==hashlib.sha256(data).hexdigest()
  assert hashlib.sha256(z.read('x')).hexdigest()!='0'*64
 write(root/'qualification_result.json',dict(actual_UTC=utc(),source_SHA=sha(__file__),synthetic_path_and_SHA_CRC_only=True,not_real_inference_or_runtime_qualification=True))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['restore','infer','qualify'],required=True);ap.add_argument('--plan',type=P);ap.add_argument('--plan-sha');ap.add_argument('--root',type=P,required=True);a=ap.parse_args()
 if a.stage=='qualify':qualify(a.root)
 else:
  assert sha(a.plan)==a.plan_sha;p=read(a.plan);assert str(a.plan)==p['plan_path'];sys.exit(capture(p,a.root,a.stage))
