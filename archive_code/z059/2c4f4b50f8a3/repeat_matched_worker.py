import argparse,datetime,hashlib,json,os,subprocess,sys,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--node',required=True);a=p.parse_args()
B=Path('/data/coding/selective_flow/strong_baselines');V=B/'matched_repeat_deployment_20261004T0650Z'
plan=json.loads((V/'plan.json').read_text());node=plan['nodes'][a.node];D=B/'own_flow_v8_identifiability/matched_repeat_20261004T0650Z'/a.node
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
assert not D.exists(),'retain attempts; never relaunch this assignment'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==node['gpu_uuid']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert shutil.disk_usage(B).free>4_000_000_000
inputproof=json.loads((B.parent/'provision_20261004T0647Z/destination_verification.json').read_text());assert inputproof['status']=='PORTABLE_INPUTS_ALL_FILES_SHA_AND_RUNTIME_VERIFIED'
manifest=json.loads((V/'deployment_manifest.json').read_text());assert all(sha(V/n)==h for n,h in manifest.items())
D.mkdir(parents=True,exist_ok=False)
state={'status':'PREPARING','pid':os.getpid(),'node':a.node,'seed':node['seed'],'started_at':now(),'attention':None,'completed':{},'current_job':None}
write(D/'status.json',state)
try:
 for case in plan['case_order']:
  seed=node['seed'];root=D/(case+'_seed'+str(seed));root.mkdir(exist_ok=False)
  caseV=V/case;ref=json.loads((caseV/'reference_protocol.json').read_text())
  source_manifest=json.loads((caseV/'source_manifest.json').read_text());assert all(sha(caseV/n)==h for n,h in source_manifest.items())
  assert sha(B/'mosi.pkl')==ref['data_sha256']
  for n,h in ref['author_source_sha256'].items():assert sha(B/'CaReFlow'/n)==h
  for n,h in ref['backbone_sha256'].items():assert sha(B/'deberta-v3-base'/n)==h
  auth={'independent_gpu_authorized':True,'gpu_uuid':node['gpu_uuid'],'frozen_inputs_verified':True,'endpoint':node['endpoint'],'human_fixed_five_seed_confirmation':True,'official_split_unchanged':True,'seed':seed,'job':case+'_seed'+str(seed),'maximum_training_processes':1,'deadline_beijing_conservative':node['deadline_beijing_estimate'],'deadline_is_estimate_from_remaining_time':True,'test_policy':'validation only in this training stage; test once after frozen structure','source_manifest':source_manifest}
  (root/'authorization.json').write_text(json.dumps(auth,indent=2))
  jobstate={'status':'PREPARING','pid':os.getpid(),'attention':None,'seed':seed,'case':case,'completed':{}}
  for stage in ['check','run']:
   parent=root/('gpu_check' if stage=='check' else 'source_views');parent.mkdir();out=parent/'attempt01'
   cmd=[str(B/'.venv/bin/python'),'-u',str(caseV/'run_source_validation.py'),'--stage',stage,'--baseline-root',str(B),'--out',str(out),'--feedback','none','--rho','0','--guard','on','--epochs','100','--seed',str(seed),'--device','cuda','--parallel-authorization',str(root/'authorization.json')]
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   with (parent/'attempt01.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=B,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    jobstate.update(status='RUNNING_CHECK' if stage=='check' else 'RUNNING_PILOT',child_pid=child.pid,command=cmd,current_attempt=str(out),current_epoch=0)
    state.update(status=jobstate['status'],current_job=case+'_seed'+str(seed),child_pid=child.pid,command=cmd,current_attempt=str(out),current_epoch=0)
    write(root/'status.json',jobstate);write(D/'status.json',state)
    while child.poll() is None:
     h=out/'history.json'
     if h.exists():state['current_epoch']=jobstate['current_epoch']=len(json.loads(h.read_text()))
     state['updated_at']=jobstate['updated_at']=now();write(root/'status.json',jobstate);write(D/'status.json',state);time.sleep(10)
    assert child.returncode==0,('Retained failure',case,stage,child.returncode)
   if stage=='check':
    checks=json.loads((out/'checks.json').read_text());assert checks['status']=='GPU_BATCH32_AND_VALIDATION128_CHECK_PASSED'
   else:
    h=json.loads((out/'history.json').read_text());s=json.loads((out/'selection.json').read_text());r=json.loads((out/'results.json').read_text());pr=json.loads((out/'protocol.json').read_text())
    assert [x['epoch'] for x in h]==list(range(1,101)) and s==r['selection']
    best=min(h,key=lambda x:x['valid_mse']);assert (best['epoch'],best['valid_mse'])==(s['best_epoch'],s['valid_mse'])
    assert sha(out/'best.pt')==s['checkpoint_sha256'] and sha(out/'protocol.json')==s['protocol_sha256']
    for k in ['name','author_source_sha256','author_commit','backbone_sha256','data_sha256','pretrained_tensors_verified','train_samples','valid_samples','author_arguments','trainable_parameters','flow_parameters','selection','optimizer','test_policy','normalization_statistics','task_loss']:
     assert pr[k]==ref[k],k
    assert pr['seed']==seed and pr['epochs']==100 and not r['test_evaluated']
    receipt={'status':'REMOTE_COMPLETE_100_SELECTION_AND_WHOLE_CHECKPOINT_SHA_VERIFIED_LOCAL_BACKUP_PENDING','selection':s,'files_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()},'node':a.node,'job':case+'_seed'+str(seed),'verified_at':now(),'test_evaluated':False,'local_full_checkpoint_saved':False}
    (root/'remote_completion_receipt.json').write_text(json.dumps(receipt,indent=2))
   jobstate['completed'][stage]={'directory':str(out),'verified_at':now()};write(root/'status.json',jobstate)
  jobstate.update(status='COMPLETE',finished_at=now(),current_epoch=100);write(root/'status.json',jobstate)
  state['completed'][case]={'directory':str(root/'source_views/attempt01'),'verified_at':now()};write(D/'status.json',state)
 state.update(status='COMPLETE',finished_at=now(),child_pid=None,current_job=None,current_epoch=None);write(D/'status.json',state)
except Exception as e:
 state.update(status='NEEDS_ATTENTION',attention=repr(e),updated_at=now());write(D/'status.json',state)
 if 'root' in globals() and 'jobstate' in globals():jobstate.update(status='NEEDS_ATTENTION',attention=repr(e));write(root/'status.json',jobstate)
 raise
