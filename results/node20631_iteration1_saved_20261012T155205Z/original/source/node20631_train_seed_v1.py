"""One fresh official-split seed. No pooled data, resume, or TEST-based selection."""
import argparse,datetime as dt,hashlib,importlib.metadata,json,os,pathlib,pickle,random,shutil,subprocess,sys,time
P=pathlib.Path
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,o):
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n');q.replace(p)
def run(a):
 p=json.loads(a.plan.read_bytes());assert sha(a.plan)==a.plan_sha and p['execution_enabled']
 assert p['protocol']=='OFFICIAL_FIXED_SPLIT_FIVE_NEW_SEEDS' and a.seed in p['seeds'] and a.method in p['methods']
 assert not a.out.exists() and a.seed not in [128,0,1,2,3]
 for n,h in p['source_SHA'].items():assert sha(a.bundle/n)==h,n
 for n,h in p['asset_SHA'].items():assert sha(a.assets/n)==h,n
 assert sys.executable==p['python']
 assert {n:importlib.metadata.version(n) for n in p['runtime_versions']}==p['runtime_versions']
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==p['GPU_UUID']
 assert (dt.datetime.fromisoformat(p['lease_end_UTC'])-dt.datetime.now(dt.timezone.utc)).total_seconds()>p['per_job_budget_seconds']+7200
 assert shutil.disk_usage(a.out.parent).free>p['new_job_minimum_free_bytes']
 memmax=int(P('/sys/fs/cgroup/memory.max').read_text());memcur=int(P('/sys/fs/cgroup/memory.current').read_text())
 assert memmax-memcur>8*1024**3
 a.out.mkdir()
 with (a.out/'train.once').open('x') as f:json.dump(dict(actual_UTC=utc(),seed=a.seed,method=a.method,plan_SHA=a.plan_sha),f)
 os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 import numpy as np
 import torch
 from types import MethodType
 from paired_fulltrain_session_candidate import load_author_components,verify_public_encoder,REMOVED_AUTHOR
 from official_fulltrain_dev_guard_candidate import FullTrainDevGuard
 from common_budget_selection_candidate import train_batches,author_dev_batch_mse
 from encoder_adapter import fit_statistics,install
 from fixed_flow_components_candidate import forward_fixed,forward_batch,tensor_sha
 from candidate_adapter import make_candidate,optimizer_groups,candidate_objective
 torch.set_num_threads(2)
 start=time.monotonic()
 with (a.assets/'assets/mosi.pkl').open('rb') as f:data=pickle.load(f)
 guard=FullTrainDevGuard(data['train'],data['dev']);del data
 assert {k:list(v) for k,v in guard.ids.items()}==p['official_train_dev_IDs']
 assert not {i.split('[')[0] for i in guard.ids['train']} & {i.split('[')[0] for i in guard.ids['dev']}
 author,argv=load_author_components(a.bundle,a.assets);author.args.seed=a.seed;author.set_random_seed(a.seed)
 train_inputs=author.get_appropriate_dataset(guard.inputs_only('train'));dev_inputs=author.get_appropriate_dataset(guard.inputs_only('dev'))
 model,discard_opt,discard_sched=author.prep_for_training(4000);del discard_opt,discard_sched
 matched=verify_public_encoder(model,a.assets/'assets/deberta-v3-base')
 for encoder in (model.dberta.transa,model.dberta.transv):encoder.embed_positions._float_tensor.zero_()
 stats=None
 if a.method!='careflow':
  stats=fit_statistics(train_inputs)
  for name in REMOVED_AUTHOR:delattr(model.dberta,name)
  model.dberta.own_flow=make_candidate(a.method);install(model.dberta,stats);model.dberta.forward=MethodType(forward_fixed,model.dberta)
 model.cuda();model.dberta.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
 for v in model.parameters():v.requires_grad_(True)
 if a.method=='careflow':
  nd=('bias','LayerNorm.bias','LayerNorm.weight');named=list(model.named_parameters())
  groups=[dict(params=[v for n,v in named if not any(k in n for k in nd)],weight_decay=.01),dict(params=[v for n,v in named if any(k in n for k in nd)],weight_decay=0.)]
 else:groups,named=optimizer_groups(model)
 opt=torch.optim.AdamW(groups,lr=1e-5);sched=author.get_linear_schedule_with_warmup(opt,num_warmup_steps=400,num_training_steps=4000)
 ids=[id(v) for g in opt.param_groups for v in g['params']];assert len(set(ids))==len(ids) and set(ids)=={id(v) for n,v in named} and not opt.state
 names={id(v):n for n,v in named};indices={str(i):names[id(v)] for g,sg in zip(opt.param_groups,opt.state_dict()['param_groups']) for v,i in zip(g['params'],sg['params'])}
 train=author.get_appropriate_dataset(guard.train_supervision())
 gen=np.random.RandomState(a.seed);orders=np.stack([gen.permutation(1281) for _ in range(100)]);np.save(a.out/'orders.npy',orders,allow_pickle=False)
 for order in orders:train_batches([int(i) for i in order])
 def cpu(o):
  if torch.is_tensor(o):return o.detach().cpu().clone()
  if isinstance(o,dict):return {k:cpu(v) for k,v in o.items()}
  if isinstance(o,(tuple,list)):return type(o)(cpu(v) for v in o)
  return o
 def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),cuda=[v.clone() for v in torch.cuda.get_rng_state_all()])
 initial=tensor_sha(model.state_dict());write(a.out/'construction.json',dict(actual_UTC=utc(),seed=a.seed,method=a.method,public_encoder_tensors_matched=matched,initial_state_SHA=initial,orders_SHA=sha(a.out/'orders.npy'),model_parameters=sum(v.numel() for _,v in named),official_IDs=p['official_train_dev_IDs'],normalization='original per-batch minmax' if a.method=='careflow' else 'official TRAIN-only masked channel statistics',no_old_task_checkpoint_loaded=True,gradient_clipping=False))
 def predict(inputs,dummy=0):
  model.eval();result=[]
  with torch.no_grad():
   for i in range(0,len(inputs.tensors[0]),128):
    batch=tuple(v[i:i+128].cuda() for v in inputs.tensors);batch[3].fill_(dummy)
    pred=author._forward_eval(model,batch)[0].view(-1) if a.method=='careflow' else forward_batch(model,batch).view(-1)
    result.append(pred.detach().cpu().numpy())
  result=np.concatenate(result).astype(np.float32);assert np.isfinite(result).all();return result
 best=float('inf');best_epoch=0;best_state=None;best_pred=None;history=[];updates=0
 def checkpoint():
  meta=dict(method=a.method,seed=a.seed,plan_SHA=a.plan_sha,epochs_completed=len(history),steps=updates,best_epoch=best_epoch,best_VAL_batch_MSE=best,initial_state_SHA=initial,orders_SHA=sha(a.out/'orders.npy'),optimizer_index_to_name=indices,official_IDs=p['official_train_dev_IDs'],source_SHA=p['source_SHA'],asset_SHA=p['asset_SHA'],normalization='author_per_batch' if a.method=='careflow' else 'TRAIN_only',checkpoint_selection='VAL229 two equal-weight batches128/101 strict earliest',TEST_used_for_selection=False,resume_execution_qualified=False)
  obj=dict(metadata=meta,model=cpu(model.state_dict()),selected_model=best_state,optimizer=cpu(opt.state_dict()),scheduler=cpu(sched.state_dict()),rng=cpu(rng()),history=history,orders=orders,statistics=cpu(stats),guard_journal=guard.journal)
  target=a.out/'complete_state.pt';tmp=target.with_name(target.name+'.tmp');torch.save(obj,tmp);tmp.replace(target);del obj
  write(a.out/'checkpoint_receipt.json',dict(actual_UTC=utc(),SHA=sha(target),bytes=target.stat().st_size,steps=updates,epochs_completed=len(history)))
 for epoch in range(100):
  tick=time.monotonic();rowslist,omitted=train_batches([int(i) for i in orders[epoch]]);losses=[]
  for rows in rowslist:
   model.train();opt.zero_grad(set_to_none=True);batch=tuple(v[list(rows)].cuda() for v in train.tensors)
   if a.method=='careflow':
    pred,lf,lb=author._forward_eval(model,batch);loss=(pred.view(-1)-batch[3].view(-1)).square().mean()+.2*lf+.1*lb
   else:
    pred=forward_batch(model,batch);loss=candidate_objective(model,pred,batch[3])
   assert torch.isfinite(loss);loss.backward();missing=[]
   for n,v in named:
    if v.grad is None:missing.append(n)
    else:assert torch.isfinite(v.grad).all(),n
   allowed={'dberta.pooler.dense.weight','dberta.pooler.dense.bias'} if a.method=='careflow' else set()
   assert set(missing)==allowed,missing
   opt.step();sched.step();updates+=1;losses.append(float(loss.detach()))
   if updates==1 or updates%40==0:
    torch.cuda.synchronize();physical=dict(actual_UTC=utc(),seed=a.seed,method=a.method,step=updates,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),missing_gradients=missing)
    assert max(physical['peak_allocated'],physical['peak_reserved'])<=p['per_process_GPU_ceiling_bytes']
    write(a.out/'health.json',physical)
   assert time.monotonic()-start<p['per_job_budget_seconds']
  assert updates==(epoch+1)*40 and sched.last_epoch==updates
  state_sha=tensor_sha(model.state_dict());values=predict(dev_inputs)
  frozen=a.out/('VAL_epoch%03d.npz'%(epoch+1));np.savez(frozen,row_ids=np.asarray(guard.ids['dev']),prediction=values,model_state_sha256=np.asarray(state_sha))
  frozen_sha=sha(frozen);y=guard.dev_labels_after_frozen_prediction(frozen,frozen_sha,state_sha)
  score=author_dev_batch_mse([np.mean((values[i:i+128].astype(np.float64)-y[i:i+128])**2) for i in (0,128)],(128,101))
  if score<best:best,best_epoch,best_state,best_pred=score,epoch+1,cpu(model.state_dict()),values.copy()
  history.append(dict(actual_UTC=utc(),epoch=epoch+1,steps=updates,TRAIN_loss=float(np.mean(losses)),VAL_batch_MSE=score,best_epoch=best_epoch,best_VAL_batch_MSE=best,seconds=time.monotonic()-tick,omitted_rows=list(omitted),prediction_SHA=frozen_sha,state_SHA=state_sha))
  write(a.out/'progress.json',history[-1]);write(a.out/'history.json',history);print(json.dumps(history[-1]),flush=True)
  if (epoch+1)%10==0 or epoch==0:checkpoint()
  assert shutil.disk_usage(a.out).free>4*1024**3
 assert updates==4000 and len(history)==100
 model.load_state_dict(best_state,strict=True);selected_sha=tensor_sha(model.state_dict());before=rng();v0=predict(dev_inputs);v7=predict(dev_inputs,7)
 assert np.array_equal(v0,best_pred) and np.array_equal(v0,v7) and tensor_sha(model.state_dict())==selected_sha
 assert torch.equal(before['torch'],torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(before['cuda'],torch.cuda.get_rng_state_all()))
 with (a.assets/'assets/mosi.pkl').open('rb') as f:data=pickle.load(f)
 test=data['test'];assert len(test)==685;test_ids=[r[2].decode() if isinstance(r[2],bytes) else r[2] for r in test]
 assert test_ids==p['official_TEST_IDs'] and not set(test_ids)&(set(guard.ids['train'])|set(guard.ids['dev']))
 test_inputs=author.get_appropriate_dataset([(r[0],np.zeros((1,1),dtype=np.float32),r[2]) for r in test])
 t0=predict(test_inputs);t7=predict(test_inputs,7);assert np.array_equal(t0,t7) and tensor_sha(model.state_dict())==selected_sha
 target=a.out/'TEST_frozen_prediction.npz';np.savez(target,row_ids=np.asarray(test_ids),prediction=t0,model_state_sha256=np.asarray(selected_sha))
 write(a.out/'TEST_prediction_freeze.json',dict(actual_UTC=utc(),SHA=sha(target),selected_state_SHA=selected_sha,best_epoch=best_epoch,before_TEST_numeric_labels=True))
 from sentiment_metrics_careflow_v1 import metrics,SEMANTICS
 with (a.out/'official_score.once').open('x') as f:f.write(utc())
 truth=np.asarray([np.asarray(r[1]).reshape(-1)[0] for r in test],dtype=np.float32).astype(np.float64)
 result=dict(status='OFFICIAL_SEED100_COMPLETE_CPU_AUDIT_AND_FULL_PUBLICATION_PENDING',actual_UTC=utc(),method=a.method,seed=a.seed,epochs=100,updates=updates,best_epoch=best_epoch,selected_state_SHA=selected_sha,VAL=metrics(best_pred,y),TEST=metrics(t0,truth),metric_semantics=SEMANTICS,TEST_historically_seen=True,TEST_not_used_for_selection=True,no_pooled_training=True,checkpoint_SHA=sha(a.out/'complete_state.pt'),wall_seconds=time.monotonic()-start)
 write(a.out/'result.json',result);write(a.out/'guard_journal.json',guard.journal)
def main():
 q=argparse.ArgumentParser()
 for n in ['plan','bundle','assets','out']:q.add_argument('--'+n,type=P,required=True)
 q.add_argument('--plan-sha',required=True);q.add_argument('--method',required=True);q.add_argument('--seed',type=int,required=True);a=q.parse_args()
 try:run(a)
 except BaseException as exc:
  if a.out.exists():write(a.out/'failed_original.json',dict(actual_UTC=utc(),error_type=type(exc).__name__,message=str(exc)))
  raise
if __name__=='__main__':main()
