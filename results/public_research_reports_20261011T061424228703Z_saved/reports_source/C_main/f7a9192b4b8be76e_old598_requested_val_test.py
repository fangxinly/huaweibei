"""Requested saved old A/B/C2 official VAL/TEST evaluation; no training or selection."""
import argparse,json,hashlib,sys,os,datetime,subprocess,zipfile,shutil,pickle,random
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def gpu(a):
 plan=json.loads(a.plan.read_text());assert sha(a.plan)==a.plan_sha
 assert sha(__file__)==plan['source_sha256'][Path(__file__).name]
 bundle=a.plan.parent
 for n,h in plan['source_sha256'].items():assert sha(bundle/n)==h
 remaining=(datetime.datetime.fromisoformat(plan['lease_end'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
 assert remaining>=plan['gpu_seconds']+7200
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().strip();assert uuid==plan['A_uuid']
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']).decode();assert not compute.strip()
 assert shutil.disk_usage('/data').free>1000000000
 import importlib.metadata as md
 for n,v in plan['runtime_exact_versions'].items():assert md.version(n)==v
 assets=Path(plan['assets'])
 for n,h in plan['asset_sha256'].items():assert sha(assets/n)==h
 item=plan['models'][a.model];assert sha(item['checkpoint'])==item['sha256']
 assert not a.root.exists();a.root.mkdir()
 write(a.root/'preflight.json',dict(actual_utc=utc(),remaining=remaining,uuid=uuid,compute=compute,full_processes=subprocess.check_output(['ps','-eo','pid,args','--width','4000']).decode()))
 os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
 import torch,numpy as np
 from types import MethodType,SimpleNamespace
 sys.path.insert(0,str(bundle));sys.path.append(plan['author_bundle'])
 from paired_fulltrain_session_candidate import load_author_components
 from counterfactual_flow_model import WholeStateFlow
 from encoder_adapter import install,forward_v6,forward_batch
 from finite_task_risk_feedback_v1 import FiniteTaskRiskFeedback
 from finite_task_risk_runtime_v1 import install_finite_inference
 from sentiment_metrics_careflow_v1 import metrics
 torch.set_num_threads(2)
 state=torch.load(item['checkpoint'],map_location='cpu');assert len(state)==461
 author,_=load_author_components(Path(plan['author_bundle']),assets);author.set_random_seed(91814)
 model,opt,sch=author.prep_for_training(4000);del opt,sch
 core=model.dberta
 for n in ('reflow_a','reflow_v','reflow_a_b','reflow_v_b','rf_a','rf_v','rf_a_b','rf_v_b'):delattr(core,n)
 core.own_flow=WholeStateFlow('none');core.own_flow.vector_feedback=FiniteTaskRiskFeedback(item['mode'])
 stats={name:{field:state['dberta.v6_'+name+'_'+field] for field in ('mean','std','active')} for name in ('audio','visual')}
 install(core,stats);core.forward=MethodType(forward_v6,core)
 model.load_state_dict(state,strict=True);del state;model.to(author.DEVICE);core.own_flow.set_epoch(41)
 if a.model=='C2':
  from finite_single_token_reader_v1 import install_for_flow
  install_for_flow(core.own_flow)
 install_finite_inference(core,SimpleNamespace(donor=core.own_flow.donor_feedback,feedback=core.own_flow.vector_feedback),item['mode'])
 model.eval();model.requires_grad_(False)
 def tensor_sha():
  h=hashlib.sha256()
  for n,t in sorted(model.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
  return h.hexdigest()
 def rng():return pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().tolist(),[v.tolist() for v in torch.cuda.get_rng_state_all()]),protocol=4)
 before=tensor_sha();before_rng=rng()
 with (assets/'assets/mosi.pkl').open('rb') as f:data=pickle.load(f)
 results={};digests={}
 for role,key,count in [('val','dev',229),('test','test',685)]:
  records=data[key];assert len(records)==count
  ds=author.get_appropriate_dataset([(r[0],np.zeros_like(r[1]),r[2]) for r in records])
  def forward(dummy):
   values=[]
   with torch.no_grad():
    for start in range(0,count,128):
     batch=[v[start:start+128].to(author.DEVICE) for v in ds.tensors];batch[3]=torch.full_like(batch[3],dummy)
     values.append(forward_batch(model,tuple(batch))[0].view(-1).detach().cpu().numpy())
   return np.concatenate(values)
  pred=forward(0);dummy=forward(7);assert np.array_equal(pred,dummy) and np.isfinite(pred).all()
  pred_path=a.root/(role+'_prediction_only.npz');np.savez(pred_path,prediction=pred,row_ids=np.asarray([str(r[2]) for r in records]))
  digests[role]=sha(pred_path);write(a.root/(role+'_freeze.json'),dict(actual_utc=utc(),prediction_sha256=digests[role]))
  if role=='val':
   with np.load(bundle/(a.model+'_original_DEV.npz'),allow_pickle=False) as z:
    replay=float(np.max(np.abs(pred-z['valid_pred'])));assert replay<2e-5
  y=np.asarray([r[1] for r in records]).reshape(-1);np.save(a.root/(role+'_targets.npy'),y,allow_pickle=False);results[role]=metrics(pred,y)
 assert before==tensor_sha() and before_rng==rng()
 peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved());assert max(peak.values())<=6*1024**3
 write(a.root/'actual_stage_receipt.json',dict(status='OLD_FIXED_MODEL_VAL_TEST_GPU_COMPLETE_CPU_PENDING',actual_utc=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,plan_sha256=a.plan_sha,model=a.model,checkpoint_sha256=item['sha256'],scores=results,prediction_sha256=digests,original_DEV_replay_max_error=replay,dummy_replay_exact=True,all_parameters_RNG_unchanged=True,peak=peak,no_updates=True,no_selection=True,official_TEST_not_used_in_original_training=True))
def cpu(a):
 import numpy as np
 p=json.loads(a.plan.read_text());assert sha(a.plan)==a.plan_sha
 sys.path.insert(0,p['CPU_metric_bundle']);from sentiment_metrics_careflow_v1 import metrics
 raw=json.loads((a.input/'actual_stage_receipt.json').read_text());assert raw['plan_sha256']==a.plan_sha
 for role in ('val','test'):
  path=a.input/(role+'_prediction_only.npz');assert sha(path)==raw['prediction_sha256'][role]
  with np.load(path,allow_pickle=False) as z:result=metrics(z['prediction'],np.load(a.input/(role+'_targets.npy'),allow_pickle=False))
  assert result==raw['scores'][role]
 assert not a.root.exists();a.root.mkdir();write(a.root/'actual_stage_receipt.json',dict(status='OLD_VAL_TEST_ORIGINAL_CPU_ARITHMETIC_COMPLETE',actual_utc=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,original_receipt_sha256=sha(a.input/'actual_stage_receipt.json'),plan_sha256=a.plan_sha,model=raw['model'],CPU_model_forward=False,scores=raw['scores']))
def run(a):
 p=json.loads(a.plan.read_text());assert sha(a.plan)==a.plan_sha;assert not a.root.exists();a.root.mkdir()
 cmd=[sys.executable,str(Path(__file__).resolve()),a.worker,'--plan',str(a.plan),'--plan-sha',a.plan_sha,'--root',str(a.output),'--model',a.model]
 if a.input:cmd+=['--input',str(a.input)]
 with (a.root/'stdout.log').open('wb') as out,(a.root/'stderr.log').open('wb') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,cwd=a.plan.parent);write(a.root/'actual_child.json',dict(pid=child.pid,fullargv=cmd,actual_start_utc=utc()))
  try:code=child.wait(timeout=p['gpu_seconds'] if a.worker=='gpu' else 300)
  except subprocess.TimeoutExpired:child.terminate();code=child.wait();write(a.root/'timeout.json',dict(actual_utc=utc()))
 write(a.root/'natural_exit.json',dict(pid=child.pid,fullargv=cmd,natural_exit=code,actual_exit_utc=utc()));sys.exit(code)
def capture(a):
 assert not a.output.exists();a.output.mkdir();n=json.loads((a.root/'natural_exit.json').read_text());r=json.loads((a.input/'actual_stage_receipt.json').read_text());c=json.loads((a.root/'actual_child.json').read_text())
 assert n['natural_exit']==0 and n['pid']==r['pid']==c['pid'] and n['fullargv']==r['fullargv']==c['fullargv']
 payload=a.output/'payload';payload.mkdir();shutil.copytree(a.root,payload/'execution');shutil.copytree(a.input,payload/'original');shutil.copytree(a.plan.parent,payload/'source')
 write(payload/'manifest.json',dict(actual_utc=utc(),member_sha256={f.relative_to(payload).as_posix():sha(f) for f in payload.rglob('*') if f.is_file()}))
 with zipfile.ZipFile(a.output/'complete_capture.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in payload.rglob('*'):
   if f.is_file():z.write(f,f.relative_to(payload).as_posix())
 with zipfile.ZipFile(a.output/'complete_capture.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 write(a.output/'capture_receipt.json',dict(actual_utc=utc(),sha256=sha(a.output/'complete_capture.zip')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['gpu','cpu','run','capture']);p.add_argument('--model',choices=['A','B','C2'],required=True)
 for name in ('plan','root','input','output'):p.add_argument('--'+name,type=Path,required=name in ('plan','root'))
 p.add_argument('--plan-sha',required=True);p.add_argument('--worker',choices=['gpu','cpu']);a=p.parse_args();globals()[a.action](a)
