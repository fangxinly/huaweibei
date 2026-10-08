"""Fixed selected20 diagnosis and requested official VAL/TEST descriptive evaluation; no updates or selection."""
import argparse,json,hashlib,sys,os,datetime,subprocess,zipfile,shutil,pickle,random,time
from pathlib import Path
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def shared(a):
 plan=json.loads(a.plan.read_text());assert sha(a.plan)==a.plan_sha
 assert sha(Path(__file__))==plan['diagnostic_source_sha256']
 parent=Path(plan['parent_roots'][a.node]);bundle=parent/'source' if a.node=='B' else Path(plan['A_bundle'])
 if a.node=='B':parent=parent/'run'
 original=json.loads((parent/'out/actual_stage_receipt.json').read_text())
 assert sha(parent/'out/actual_stage_receipt.json')==plan['parent_receipt_sha256']
 for n,h in original['source_sha256'].items():assert sha(bundle/n)==h
 assert sha(parent/'out/resume_and_selected_full.pt')==plan['checkpoint_sha256']
 sys.path.insert(0,str(bundle));return plan,parent,bundle,original
def arithmetic(pred,labels,metric):
 import numpy as np
 y=np.asarray(labels,dtype=np.float64).reshape(-1);b,f,p=[np.asarray(pred[k],dtype=np.float64) for k in ('b','f','p')]
 exact=(p-y)**2-(b-y)**2;identity=2*(b-y)*(p-b)+(p-b)**2
 error=float(np.max(np.abs(exact-identity)));assert error<1e-10
 result={k:metric(pred[k],y) for k in ('b','f','p')}
 result.update(p_minus_b_MSE=float(exact.mean()),algebra_max_error=error,
  rows=len(y),correction_mean=float((p-b).mean()),correction_rms=float(np.sqrt(np.mean((p-b)**2))))
 result['weak_strong']={}
 for name,mask in [('weak',np.abs(y)<=1),('strong',np.abs(y)>1)]:
  result['weak_strong'][name]={k:dict(rows=int(mask.sum()),MAE=float(np.mean(np.abs(v[mask]-y[mask]))),
   bias=float(np.mean(v[mask]-y[mask]))) for k,v in [('b',b),('f',f),('p',p)]}
 return result
def gpu(a):
 import importlib.metadata as md
 plan,parent,bundle,r=shared(a);assets=Path(plan['assets']);original_plan=json.loads((bundle/'pilot40_execution_protocol.json').read_text())
 versions={n:md.version(n) for n in original_plan['runtime_exact_versions']};assert versions==original_plan['runtime_exact_versions']
 for n,h in original_plan['asset_sha256'].items():assert sha(assets/n)==h
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().strip();assert uuid==plan['A_uuid']
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']).decode();assert not compute.strip()
 remaining=(datetime.datetime.fromisoformat(plan['lease_end'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert remaining>=plan['gpu_seconds']+7200
 assert shutil.disk_usage('/data').free>1000000000
 assert not a.root.exists();a.root.mkdir()
 write(a.root/'actual_preflight.json',dict(actual_utc=utc(),uuid=uuid,compute=compute,runtime=versions,
  fullargv=[sys.executable]+sys.argv,processes=subprocess.check_output(['ps','-eo','pid,args','--width','4000']).decode(),remaining_seconds=remaining))
 os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
 import torch,numpy as np
 from fold_session import construct,prediction
 from fixed_flow_components_candidate import tensor_sha
 from sentiment_metrics_careflow_v1 import metrics
 torch.set_num_threads(2);split=json.loads((bundle/'split.json').read_text());fold=split['folds'][0]
 session=construct('anchored_flow',bundle,assets,original_plan,split,0)
 state=torch.load(parent/'out/resume_and_selected_full.pt',map_location='cpu')
 assert tensor_sha(state['selected_model'])==r['selected_state_sha256']
 session.model.load_state_dict(state['selected_model'],strict=True);del state
 session.model.eval();flow=session.model.dberta.own_flow
 with (assets/'assets/mosi.pkl').open('rb') as handle:data=pickle.load(handle)
 official={role:data[key] for role,key in [('val','dev'),('test','test')]}
 role_ids={role:[str(row[2]) for row in rows] for role,rows in official.items()}
 for role,rows in official.items():
  session.inputs[role]=session.author.get_appropriate_dataset([(row[0],np.zeros_like(row[1]),row[2]) for row in rows])
 overlap={role:{part:len(set(ids)&set(fold['row_ids'][part])) for part in ('fit','inner','outer')} for role,ids in role_ids.items()}
 before=tensor_sha(session.model.state_dict())
 def rng():return pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().tolist(),[v.tolist() for v in torch.cuda.get_rng_state_all()]),protocol=4)
 before_rng=sha_bytes(rng());gain=float(flow.gain.detach().tanh())
 def forward(role,dummy):
  all_values={k:[] for k in ('b','f','p')};tensors=session.inputs[role].tensors
  with torch.no_grad():
   for start in range(0,len(tensors[0]),128):
    batch=[v[start:start+128].to(session.author.DEVICE) for v in tensors];batch[3]=torch.full_like(batch[3],dummy)
    p=prediction(session,'anchored_flow',tuple(batch))
    for k,v in [('b',flow.last_base),('f',flow.last_flow),('p',p)]:all_values[k].append(v.detach().cpu().numpy())
  return {k:np.concatenate(v).astype(np.float32) for k,v in all_values.items()}
 scores={};digests={}
 for role in ('fit','inner','val','test'):
  pred=forward(role,0);dummy=forward(role,7)
  for k in pred:assert np.array_equal(pred[k],dummy[k]) and np.isfinite(pred[k]).all()
  path=a.root/(role+'_bfp.npz');np.savez(path,**pred,row_ids=np.asarray(fold['row_ids'][role] if role in ('fit','inner') else role_ids[role]),selected_state_sha256=np.asarray(before))
  digests[role]=sha(path);write(a.root/(role+'_prediction_freeze.json'),dict(actual_utc=utc(),sha256=digests[role],state_sha256=before))
  if role=='inner':
   with np.load(parent/'out/selected_INNER_prediction_only.npz',allow_pickle=False) as z:assert np.array_equal(pred['p'],z['prediction'])
  if role in ('fit','inner'):
   label_path=parent/'out'/('original_FIT_supervision.npy' if role=='fit' else 'original_INNER_selection_labels.npy')
   labels=np.load(label_path,allow_pickle=False).reshape(-1)
  else:labels=np.asarray([row[1] for row in official[role]]).reshape(-1)
  np.save(a.root/(role+'_evaluation_targets.npy'),labels,allow_pickle=False)
  scores[role]=arithmetic(pred,labels,metrics)
 assert tensor_sha(session.model.state_dict())==before and sha_bytes(rng())==before_rng
 torch.cuda.synchronize();peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved());assert max(peak.values())<=6*1024**3
 counts={k:sum(v.numel() for n,v in session.model.named_parameters() if any(n.startswith(q) for q in names)) for k,names in {'donor_role':['dberta.own_flow.donor_feedback.','dberta.own_flow.role_head.'],'gain':['dberta.own_flow.gain']}.items()}
 write(a.root/'actual_stage_receipt.json',dict(status='FIXED_SELECTED20_BFP_GPU_COMPLETE_CPU_PENDING',actual_utc=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,
  plan_sha256=a.plan_sha,checkpoint_sha256=plan['checkpoint_sha256'],selected_state_sha256=before,gain_tanh=gain,
  scores=scores,prediction_sha256=digests,all_parameter_and_RNG_unchanged=True,INNER_original_final_replay_exact=True,dummy_all_bfp_exact=True,
  peak=peak,actual_parameter_counts=counts,no_updates=True,official_role_overlap=overlap,official_VAL_TEST_independent=False,
  scope='Human requested fixed20 official VAL/TEST descriptive scores; pooled fold FIT/INNER overlap disclosed; no checkpoint or method selection',
  pickle_materializes_all_original_label_objects=True))
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def cpu(a):
 import numpy as np
 plan,parent,bundle,r=shared(a)
 from sentiment_metrics_careflow_v1 import metrics
 original=a.input;raw=json.loads((original/'actual_stage_receipt.json').read_text());assert raw['plan_sha256']==a.plan_sha
 errors=[]
 for role in ('fit','inner','val','test'):
  p=original/(role+'_bfp.npz');assert sha(p)==raw['prediction_sha256'][role]
  with np.load(p,allow_pickle=False) as z:
   split=json.loads((bundle/'split.json').read_text())
   if role in ('fit','inner'):assert z['row_ids'].tolist()==split['folds'][0]['row_ids'][role]
   pred={k:z[k] for k in ('b','f','p')}
  lp=original/(role+'_evaluation_targets.npy')
  s=arithmetic(pred,np.load(lp,allow_pickle=False),metrics)
  assert s==raw['scores'][role]
  if role=='inner':
   with np.load(parent/'out/selected_INNER_prediction_only.npz',allow_pickle=False) as z:assert np.array_equal(pred['p'],z['prediction'])
 assert not a.root.exists();a.root.mkdir()
 write(a.root/'actual_stage_receipt.json',dict(status='FIXED_SELECTED20_BFP_CPU_ORIGINAL_ARITHMETIC_COMPLETE',actual_utc=utc(),pid=os.getpid(),
  fullargv=[sys.executable]+sys.argv,plan_sha256=a.plan_sha,original_receipt_sha256=sha(original/'actual_stage_receipt.json'),
  predictions=raw['prediction_sha256'],saved_metrics_exact_equal=True,CPU_model_forward=False,official_role_overlap=raw['official_role_overlap'],official_VAL_TEST_independent=False))
def run(a):
 assert not a.root.exists();a.root.mkdir();plan=json.loads(a.plan.read_text());assert sha(a.plan)==a.plan_sha
 cmd=[sys.executable,str(Path(__file__).resolve()),a.worker,'--plan',str(a.plan),'--plan-sha',a.plan_sha,'--root',str(a.output),'--node',a.node]
 if a.input:cmd+=['--input',str(a.input)]
 start=utc()
 with (a.root/'stdout.log').open('wb') as out,(a.root/'stderr.log').open('wb') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,cwd=Path(plan['A_bundle']) if a.node=='A' else Path(plan['parent_roots']['B'])/'source')
  write(a.root/'actual_child.json',dict(pid=child.pid,fullargv=cmd,actual_start_utc=start,plan_sha256=a.plan_sha))
  try:code=child.wait(timeout=plan['gpu_seconds'] if a.worker=='gpu' else 300)
  except subprocess.TimeoutExpired:child.terminate();code=child.wait();write(a.root/'budget_timeout.json',dict(actual_utc=utc(),limit=plan['gpu_seconds']))
 write(a.root/'natural_exit.json',dict(pid=child.pid,fullargv=cmd,natural_exit=code,actual_exit_utc=utc(),plan_sha256=a.plan_sha))
 if code:sys.exit(code)
def capture(a):
 plan=json.loads(a.plan.read_text());assert sha(a.plan)==a.plan_sha;assert not a.output.exists();a.output.mkdir()
 n=json.loads((a.root/'natural_exit.json').read_text());c=json.loads((a.root/'actual_child.json').read_text());r=json.loads((a.input/'actual_stage_receipt.json').read_text())
 assert n['natural_exit']==0 and c['pid']==r['pid']==n['pid'] and c['fullargv']==r['fullargv']==n['fullargv']
 payload=a.output/'payload';payload.mkdir();shutil.copytree(a.root,payload/'execution');shutil.copytree(a.input,payload/'original')
 shutil.copy2(Path(__file__),payload/Path(__file__).name);shutil.copy2(a.plan,payload/'protocol.json')
 write(payload/'manifest.json',dict(actual_capture_utc=utc(),member_sha256={str(p.relative_to(payload)).replace('\\','/'):sha(p) for p in payload.rglob('*') if p.is_file()},
  parent_full_checkpoint_sha_ref=plan['checkpoint_sha256'],original_receipt_sha256=sha(a.input/'actual_stage_receipt.json'),plan_sha256=a.plan_sha))
 archive=a.output/'complete_capture.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in payload.rglob('*'):
   if p.is_file():z.write(p,str(p.relative_to(payload)).replace('\\','/'))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 write(a.output/'capture_receipt.json',dict(actual_utc=utc(),sha256=sha(archive),bytes=archive.stat().st_size))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['gpu','cpu','run','capture'])
 for name in ['plan','root','input','output']:p.add_argument('--'+name,type=Path,required=name in ['plan','root'])
 p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=['A','B'],default='A');p.add_argument('--worker',choices=['gpu','cpu'])
 a=p.parse_args();globals()[a.action](a)
