"""Preserve naturally failed paired prechecks; never qualifies them for training."""
import argparse,datetime,hashlib,json,os,pathlib,shutil,subprocess,sys,zipfile
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser()
 for n in ('root','bundle','protocol','dest'):p.add_argument('--'+n,type=pathlib.Path,required=True)
 p.add_argument('--protocol-sha',required=True);p.add_argument('--action',choices=('capture','cpu'),required=True);p.add_argument('--method',required=True);a=p.parse_args()
 assert sha(a.protocol)==a.protocol_sha
 q=read(a.protocol);assert sha(pathlib.Path(__file__))==q['worker_sha256']
 sys.path.insert(0,str(a.bundle))
 from paired_fulltrain_evidence_candidate_v1 import plan_gate,stable_file,zip_audit
 plan=plan_gate(a.bundle,q['original_plan_sha256'])
 assert not a.dest.exists();a.dest.mkdir()
 raw={k:subprocess.run(c,check=True,capture_output=True,text=True).stdout for k,c in [('uuid',['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']),('compute',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader']),('full_ps_argv',['ps','-eo','pid,ppid,lstart,args','--width','10000'])]}
 assert not raw['compute'].strip() and shutil.disk_usage(a.dest).free>=4294967296
 assert raw['uuid'].strip()==(plan['assigned_gpu_UUID'][a.method] if a.action=='capture' else plan['original_CPU_gpu_UUID'])
 ex,launch,wrap=read(a.root/'natural_exit.json'),read(a.root/'actual_child_launch.json'),read(a.root/'wrapper_actual_start.json')
 assert ex['exit_code']==1 and ex['natural_exit'] is True and ex['receipt_sha256'] is None
 assert ex['full_argv']==launch['full_argv']==wrap['child_full_argv'] and ex['child_pid']==launch['pid']
 assert wrap['child_source_sha256']==plan['source_sha256']['paired_fulltrain_runtime_candidate_v1.py']
 assert ex['full_argv'][ex['full_argv'].index('--method')+1]==a.method
 assert 'Cumulative GPU peak exceeded declared budget' in (a.root/'child.stderr.log').read_text()
 assert not (a.root/'out/actual_stage_receipt.json').exists()
 result={'actual_utc':now(),'pid':os.getpid(),'argv':sys.argv,'method':a.method,'raw_actual_checks':raw,'free_bytes':shutil.disk_usage(a.dest).free,'original_plan_sha256':q['original_plan_sha256'],'failure_protocol_sha256':a.protocol_sha,'original_exit_sha256':sha(a.root/'natural_exit.json'),'original_failed_child_argv':ex['full_argv'],'failed_precheck_is_not_passed':True,'training100_started':False,'DEV_labels_read':False,'final_TEST_read':False}
 if a.action=='capture':
  small={};large={}
  for label,base in [('run',a.root),('source',a.bundle),('preservation',a.protocol.parent)]:
   for f in base.rglob('*'):
    if not f.is_file() or '__pycache__' in f.parts or f.suffix=='.zip':continue
    rel=label+'/'+f.relative_to(base).as_posix();record=stable_file(f)
    (large if f.stat().st_size>16*1024**2 else small)[rel]=record
  manifest={'small_members':small,'large_references_not_downloads':large,'failed_stage_exit_sha256':result['original_exit_sha256']}
  write(a.dest/'member_manifest.json',manifest)
  with zipfile.ZipFile(a.dest/'snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
   for n,r in small.items():z.write(r['original_path'],n)
   z.write(a.dest/'member_manifest.json','member_manifest.json')
  result.update(status='ACTUAL_FAILED_PAIRED_PRECHECK_CAPTURE_COMPLETE',snapshot=zip_audit(a.dest/'snapshot.zip'),manifest_sha256=sha(a.dest/'member_manifest.json'),large_references_not_downloads=large)
 else:
  original=a.root/'out/clean_initial_full.pt';record=zip_audit(original,q['clean_initial_sha256'][a.method])
  import torch
  from paired_fulltrain_CPU_audit_candidate_v1 import verify_parameters
  torch.set_num_threads(2);checkpoint=torch.load(original,map_location='cpu');check=verify_parameters(torch,checkpoint,checkpoint['model'],0)
  assert checkpoint['metadata']['method']==a.method and checkpoint['metadata']['plan_sha256']==q['original_plan_sha256']
  assert check['state_sha256']==checkpoint['metadata']['initial_state_sha256']
  result.update(status='ACTUAL_FAILED_PRECHECK_CLEAN_INITIAL_ORIGINAL_OTHER_CPU_COMPLETE',whole_clean_initial=record,original_model_Adam_scheduler_RNG_check=check,CPU_model_forward=False,failed_after_update_state_not_saved=True)
 write(a.dest/'actual_receipt.json',result);print('COMPLETE '+json.dumps({'status':result['status'],'receipt_sha256':sha(a.dest/'actual_receipt.json'),'actual_utc':result['actual_utc']}),flush=True)
if __name__=='__main__':main()
