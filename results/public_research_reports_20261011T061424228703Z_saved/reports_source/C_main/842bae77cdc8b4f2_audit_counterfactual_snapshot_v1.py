from pathlib import Path
import argparse,datetime,hashlib,io,json,shlex,sys,zipfile
import numpy as np
a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--stage',choices=['live','complete'],required=True);a.add_argument('--previous',type=Path);a.add_argument('--nodes',nargs='+',default=['a','b','c']);a.add_argument('--weights',type=Path);a.add_argument('--leaf',default='');a.add_argument('--capture-source',type=Path);c=a.parse_args()
w=Path(__file__).resolve().parent;dep='inflow_counterfactual_v5_deployment_20261005T0520Z';src=w/dep
assign={'a':('none',7185,'GPU-5902bbd4-2328-0822-777c-1311d53ee5a3'),'b':('fixed',7510,'GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d'),'c':('predicted',11630,'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa')}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
capture=sha(c.capture_source or w/'capture_counterfactual_followup_v3.py');plan=json.loads((src/'frozen_plan.json').read_text());previous={x['node']:x for x in json.loads(c.previous.read_text(encoding='utf-8'))['rows']} if c.previous else {}
rows=[];shared=None;phases=[]
for node in c.nodes:
 mode,pid,uuid=assign[node];d=c.directory/node/c.leaf;proof=json.loads((d/'proof.json').read_text());raw=(d/'snapshot.zip').read_bytes()
 assert len(raw)==proof['archive_bytes'] and hashlib.sha256(raw).hexdigest()==proof['archive_sha256'] and proof['capture_source_sha256']==capture
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  members=json.loads(z.read('member_manifest.json'));assert members==proof['members'] and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(members)|{'member_manifest.json'}
  for name,e in members.items():
   b=z.read(name);assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],name
  def read(name):return json.loads(z.read(name))
  pre=dep+'/run_'+mode+'/';p=read(pre+'protocol.json');h=read(pre+'history.json');launch=read(dep+'/training_launch.json');r=read('actual_resources.json');log=z.read(dep+'/training.log').decode()
  assert 'Traceback' not in log and all(r[k]['returncode']==0 for k in ['gpu','compute','processes']) and uuid in r['gpu']['output']
  assert launch['pid']==pid and launch['gpu_uuid']==uuid and launch['performance_training_started'] and launch['args'][launch['args'].index('--mode')+1]==mode and launch['args'][launch['args'].index('--expected-initial-sha')+1]==plan['expected_initial_sha']
  assert launch['source_manifest_sha256']==sha(src/'source_manifest.json') and launch['plan_sha256']==sha(src/'frozen_plan.json') and launch['launcher_sha256']==sha(src/'launch_counterfactual_v5.py')
  assert read(dep+'/frozen_plan.json')==plan
  assert p['name']=='inflow_counterfactual_v5' and p['mode']==mode and p['gpu_uuid']==uuid and p['seed']==91814 and p['epochs']==100 and p['total_updates']==4000 and p['train_samples']==1281 and p['valid_samples']==229 and p['pretrained_tensors_verified']==198
  assert p['initial_model_sha256']==plan['expected_initial_sha'] and p['batch_order_sha256']==plan['expected_orders_sha']
  for name,digest in p['own_source_sha256'].items():assert hashlib.sha256(z.read(dep+'/'+name)).hexdigest()==digest==sha(src/name)
  orders=np.load(io.BytesIO(z.read(pre+'batch_orders.npy')),allow_pickle=False);assert orders.shape==(100,1280) and hashlib.sha256(orders.tobytes()).hexdigest()==p['batch_order_sha256']
  for order,digest in zip(orders,p['batch_order_epoch_sha256']):assert hashlib.sha256(order.tobytes()).hexdigest()==digest and len(set(order))==1280 and order.min()>=0 and order.max()<1281
  common={k:v for k,v in p.items() if k not in ['mode','gpu_uuid']}
  if shared is None:shared=common
  else:assert shared==common
  assert [e['epoch'] for e in h]==list(range(1,len(h)+1)) and all(np.isfinite(e['valid_mse']) for e in h)
  for j,e in enumerate(h):
   best=min(range(j+1),key=lambda i:h[i]['valid_mse']);assert e['best_epoch']==best+1 and e['best_valid_mse']==h[best]['valid_mse']
   if j<10:assert e['last_dev_batch_utility_weights']==[.5]*6
   elif mode=='none':assert e['last_dev_batch_utility_weights']==[0.]*6
   elif mode=='fixed':assert e['last_dev_batch_utility_weights']==[.5]*6
  delta=None
  if node in previous:
   old=previous[node];assert h[:old['epochs']]==old['history_prefix'];assert proof['captured_at']>old['captured_at'];delta=len(h)-old['epochs'];assert delta>=0
  phase=None
  if len(h)>=10:
   phase=read(pre+'shared_phase.json');assert phase['epoch']==10 and phase['batch_order_sha256']==plan['expected_orders_sha'];phases.append(phase['model_sha256'])
  matching=[line for line in r['processes']['output'].splitlines() if line.split() and line.split()[0]==str(pid)]
  complete_fields={}
  if c.stage=='live':
   assert 1<=len(h)<100 and len(matching)==1 and not ('COUNTERFACTUAL_V5_RUN_COMPLETE' in log)
   args=shlex.split(matching[0])[2:];assert args==launch['args'],(args,launch['args'])
   assert r['compute']['output'].strip(),'No GPU compute for live run'
  else:
   assert len(h)==100 and not matching and 'COUNTERFACTUAL_V5_RUN_COMPLETE' in log
   selection=read(pre+'selection.json');result=read(pre+'results.json');best=min(range(100),key=lambda i:h[i]['valid_mse'])
   assert selection['best_epoch']==best+1 and selection['valid_mse']==h[best]['valid_mse'] and selection['protocol_sha256']==members[pre+'protocol.json']['sha256'] and selection['selected_effective_mode']==('fixed' if best<10 else mode)
   assert result['selection']==selection and result['model_state_unchanged'] and not result['test_accessed']
   cp=c.weights/node/'run/best.pt';assert sha(cp)==selection['checkpoint_sha256'];assert cp.stat().st_size>700000000
   arrays=np.load(io.BytesIO(z.read(pre+'predictions.npz')),allow_pickle=False);assert arrays['valid_y'].shape==(229,)
   official_y=np.load(w/'utility_completed_20261005/a/run/predictions.npz',allow_pickle=False)['valid_y'];assert np.array_equal(arrays['valid_y'],official_y)
   for key in arrays.files:assert arrays[key].shape[0]==229 and np.isfinite(arrays[key]).all()
   assert all(arrays[key].shape==(229,6) for key in ['pair','utility','predicted_weights','weights']) and arrays['own'].shape==(229,3)
   assert np.allclose(arrays['predicted_weights'],1/(1+np.exp(-4*arrays['utility'])),atol=1e-7)
   effective=selection['selected_effective_mode'];assert np.allclose(arrays['weights'],0 if effective=='none' else .5 if effective=='fixed' else arrays['predicted_weights'])
   sys.path.insert(0,'C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen/outputs/monitoring_tools');from verify_careflow_five_seeds_v1 import metrics
   for key,predkey in [('valid','valid_pred'),('frozen_condition_off','condition_off_pred')]:
    measured=metrics(arrays[predkey],arrays['valid_y'])
    assert set(measured)==set(result[key])
    for name,value in measured.items():assert np.isclose(value,result[key][name],atol=1e-6),name
   assert np.isclose(result['valid']['author_batch_mse'],selection['valid_mse'],atol=1e-6)
   if effective=='none':assert np.allclose(arrays['valid_pred'],arrays['condition_off_pred'],atol=1e-6)
   complete_fields={'full_checkpoint_verified':True,'checkpoint':str(cp.resolve()),'checkpoint_bytes':cp.stat().st_size,'checkpoint_sha256':selection['checkpoint_sha256'],'selection':selection,'results':result,'files':{name:members[pre+name] for name in ['protocol.json','batch_orders.npy','history.json','selection.json','results.json','predictions.npz']},'extra_files':{'shared_phase.json':members[pre+'shared_phase.json']}}
  rows.append({'node':node,'mode':mode,'epochs':len(h),'increment':delta,'captured_at':proof['captured_at'],'archive_sha256':proof['archive_sha256'],'history_prefix':h,'shared_phase':phase,'actual_resources':r,'stage':c.stage,'source_initialization_orders_protocol_verified':True,**complete_fields})
assert len(set(phases))<=1
out={'audited_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':c.stage,'status':'COUNTERFACTUAL_SNAPSHOT_AUDITED','rows':rows,'shared_phase_matching_if_available':True,'limits':'Complete requires new full checkpoint. DEV exploratory; no TEST or semantic truth.'}
assert not c.out.exists();c.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'status':out['status'],'stage':c.stage,'epochs':{x['node']:x['epochs'] for x in rows}}))
