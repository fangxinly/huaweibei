"""Synthetic CPU Adam/scheduler/RNG trajectory audit, never task-data fitting."""
import copy,datetime as dt,json,os,pathlib,random
from types import SimpleNamespace

def main():
 r=pathlib.Path(__file__).parent
 with (r/'synthetic_CPU_resume.once').open('x') as f:f.write(dt.datetime.now(dt.timezone.utc).isoformat())
 mem=pathlib.Path('/proc/meminfo').read_text();available=int(next(x.split()[1] for x in mem.splitlines() if x.startswith('MemAvailable:')))*1024
 if available<6*1024**3:raise PermissionError('Synthetic CPU available RAM below6GiB')
 import numpy as np
 import torch
 from fixed_flow_components_candidate import tensor_sha
 from group5_epoch_resume_v1 import restore
 torch.set_num_threads(2)
 if torch.cuda.is_available():raise PermissionError('Synthetic CPU process unexpectedly has CUDA')
 def fresh():
  random.seed(128);np.random.seed(128);torch.manual_seed(128)
  model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.Dropout(.3),torch.nn.Linear(5,1))
  opt=torch.optim.AdamW(model.parameters(),lr=.01)
  scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda n:1-n/10)
  return SimpleNamespace(model=model,optimizer=opt,scheduler=scheduler,method='synthetic',stats=None,
   guard=SimpleNamespace(journal=[]),fold=dict(fold=0,rows=dict(fit=4,inner=2),row_ids=dict(fit=['f0','f1','f2','f3'],inner=['i0','i1'])))
 def step(s):
  s.optimizer.zero_grad();x=torch.randn(4,3)+random.random()+float(np.random.uniform());target=torch.randn(4,1)
  loss=(s.model(x)-target).square().mean();loss.backward();s.optimizer.step();s.scheduler.step()
 def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),cuda=[])
 def opt_names(s):
  names={id(v):n for n,v in s.model.named_parameters()}
  return {str(i):names[id(v)] for g,sg in zip(s.optimizer.param_groups,s.optimizer.state_dict()['param_groups']) for v,i in zip(g['params'],sg['params'])}
 s=fresh();orders=np.tile(np.arange(4),(100,1));hist=[]
 for epoch in (1,2):
  step(s);hist.append(dict(epoch=epoch,updates=epoch,inner_MSE=1/epoch,best_epoch=epoch,best_MSE=1/epoch,state_SHA=tensor_sha(s.model.state_dict())))
 saved=dict(metadata=dict(method='synthetic',fold=0,exact_plan_SHA='fixture-origin',source_SHA={'fixture':'synthetic'},historical_task_weights_used=False,outer_labels_decoded=False,fit_ids=s.fold['row_ids']['fit'],inner_ids=s.fold['row_ids']['inner'],history=hist,updates=2),
  model=copy.deepcopy(s.model.state_dict()),selected_model=copy.deepcopy(s.model.state_dict()),optimizer=copy.deepcopy(s.optimizer.state_dict()),scheduler=copy.deepcopy(s.scheduler.state_dict()),
  statistics=None,orders=orders,optimizer_index_to_name=opt_names(s),selected_inner_prediction=np.asarray([0.,1.],dtype=np.float32),guard_journal=[],rng=rng())
 path=r/'synthetic_full_state.pt';torch.save(saved,path);loaded=torch.load(path,map_location='cpu')
 for _ in range(3):step(s)
 expected_model=tensor_sha(s.model.state_dict());expected_rng=rng();expected_opt=s.optimizer.state_dict()
 t=fresh();state=restore(t,loaded,{'fixture':'synthetic'},'fixture-origin',orders,opt_names(t))
 assert state['updates']==2 and state['best_epoch']==2
 for _ in range(3):step(t)
 assert tensor_sha(t.model.state_dict())==expected_model and t.scheduler.last_epoch==s.scheduler.last_epoch==5
 actual_rng=rng();assert actual_rng['python']==expected_rng['python'] and np.array_equal(actual_rng['numpy'][1],expected_rng['numpy'][1]) and torch.equal(actual_rng['torch'],expected_rng['torch'])
 assert t.optimizer.state_dict()['param_groups']==expected_opt['param_groups']
 for i,e in expected_opt['state'].items():
  for k,v in e.items():assert torch.equal(t.optimizer.state_dict()['state'][i][k],v)
 rejected=0
 for source,origin in [({'fixture':'wrong'},'fixture-origin'),({'fixture':'synthetic'},'wrong-origin')]:
  try:restore(fresh(),loaded,source,origin,orders,opt_names(t))
  except PermissionError:rejected+=1
  else:raise AssertionError('Wrong provenance accepted')
 report=dict(status='SYNTHETIC_CPU_FULL_ADAM_SCHEDULER_RNG_RESUME_PASS',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),CPU_available_RAM_bytes=available,
  continuous_vs_resumed_model_moments_scheduler_rng_exact=True,provenance_negatives_rejected=rejected,
  actual_task_arrays_or_labels=False,native_full_model_retraining=False,CUDA_RNG_restore_exercised=False)
 (r/'synthetic_CPU_resume_report.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
