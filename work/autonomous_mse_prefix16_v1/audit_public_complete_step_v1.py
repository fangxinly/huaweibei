"""Download a newly preserved complete checkpoint; independently audit optimizer and history."""
import argparse,datetime,hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--publication',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
publication=json.loads(a.publication.read_text());expected=json.loads(a.expected.read_text());assert publication['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED'
UUID=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert UUID=='GPU-609b23d6-282d-8a2b-23f5-9433b824d212'
a.out.mkdir(exist_ok=False);cp=a.out/'complete_original.pt';rows=sorted([r for r in publication['assets'] if r['source_sha256']==expected['checkpoint_SHA']],key=lambda r:r['source_offset']);assert rows
with cp.open('xb') as target:
 for row in rows:
  assert target.tell()==row['source_offset'];h=hashlib.sha256();n=0
  with urllib.request.urlopen(row['url'],timeout=180) as f:
   for block in iter(lambda:f.read(8*1024**2),b''):target.write(block);h.update(block);n+=len(block)
  assert n==row['bytes'] and h.hexdigest()==row['sha256']
assert cp.stat().st_size==expected['checkpoint_bytes'] and sha(cp)==expected['checkpoint_SHA']
sys.path.insert(0,str(a.bundle))
import numpy as np
import torch
from fixed_flow_components_candidate import tensor_sha
from common_budget_selection_candidate import train_batches
plan=json.loads((a.bundle/'qualified_resume_plan.json').read_text());order=np.load(a.bundle/plan['orders_file'],allow_pickle=False);assert sha(a.bundle/plan['orders_file'])==plan['orders_SHA'];torch.set_num_threads(2);full=torch.load(cp,map_location='cpu');m=full['metadata'];steps=expected['steps'];assert m['steps']==steps and m['orders_SHA']==plan['orders_SHA'] and m['plan_SHA']==sha(a.bundle/'qualified_resume_plan.json')
assert tensor_sha(full['model'])==m['final_state_SHA'];info=m['parameter_info'];mapping=m['optimizer_index_to_name'];opt=full['optimizer'];assert len(opt['state'])==len(info)==len(mapping)
for i,q in opt['state'].items():
 assert float(q['step'])==steps
 for k in ('exp_avg','exp_avg_sq'):assert list(q[k].shape)==info[mapping[str(i)]]['shape'] and torch.isfinite(q[k]).all()
 assert (q['exp_avg_sq']>=0).all()
assert full['scheduler']['last_epoch']==steps and full['rng']['torch'].dtype==torch.uint8 and len(full['rng']['cuda'])==1
hist=full['history'];assert len(hist)==steps//40
for i,row in enumerate(hist):assert row['steps']==(i+1)*40 and row['epoch']==i+1 and row['dropped_rows']==[int(order[i,-1])]
best=int(np.argmin([r['DEV_batch_MSE'] for r in hist]))+1;assert best==m['best_epoch'] and hist[best-1]['DEV_batch_MSE']==m['best_MSE']
assert len(full['prefix_records'])==16
for i,row in enumerate(full['prefix_records']):assert row['step']==i+1 and row['rows']==list(train_batches(list(map(int,order[0])))[0][i])
result=dict(status='B_PUBLIC_COMPLETE_STEP_CHECKPOINT_MODEL_ADAM_RNG_ORDERS_HISTORY_PASSED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,UUID=UUID,source_SHA=sha(__file__),checkpoint_SHA=sha(cp),checkpoint_bytes=cp.stat().st_size,steps=steps,all_Adam_steps=steps,Adam_states=len(info),model_state_SHA=m['final_state_SHA'],selected_state_SHA=tensor_sha(full['selected_model']),best_epoch=best,scheduler_step=steps,RNG_orders_history_passed=True,CPU_encoder_forward=False,original_retained=str(cp))
(a.out/'audit_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
