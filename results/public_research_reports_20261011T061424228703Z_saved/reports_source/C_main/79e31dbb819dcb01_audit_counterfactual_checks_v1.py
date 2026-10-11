from pathlib import Path
import argparse,datetime,hashlib,io,json,zipfile
import numpy as np
a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True);a.add_argument('--out',type=Path,required=True);c=a.parse_args()
work=Path(__file__).resolve().parent;dep='inflow_counterfactual_v5_checks_20261005T0516Z';src=work/dep
capture=hashlib.sha256((work/'capture_counterfactual_followup_v2.py').read_bytes()).hexdigest()
assign={'a':('none',6973,'GPU-5902bbd4-2328-0822-777c-1311d53ee5a3'),'b':('fixed',7298,'GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d'),'c':('predicted',11408,'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa')}
shared=None;rows=[]
for node,(mode,pid,uuid) in assign.items():
 p=c.directory/node;proof=json.loads((p/'proof.json').read_text());raw=(p/'snapshot.zip').read_bytes()
 assert len(raw)==proof['archive_bytes'] and hashlib.sha256(raw).hexdigest()==proof['archive_sha256'] and proof['capture_source_sha256']==capture
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  manifest=json.loads(z.read('member_manifest.json'));assert manifest==proof['members']
  assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(manifest)|{'member_manifest.json'}
  for name,e in manifest.items():
   b=z.read(name);assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'],name
  resources=json.loads(z.read('actual_resources.json'));assert all(resources[k]['returncode']==0 for k in ('gpu','compute','processes'))
  assert uuid in resources['gpu']['output'] and not resources['compute']['output'].strip()
  assert not any(line.split()[0]==str(pid) for line in resources['processes']['output'].splitlines() if line.split())
  launch=json.loads(z.read(dep+'/check_launch.json'));assert launch['pid']==pid and launch['gpu_uuid']==uuid and not launch['performance_training_started']
  protocol=json.loads(z.read(dep+'/check_'+mode+'/protocol.json'));checks=json.loads(z.read(dep+'/check_'+mode+'/checks.json'))
  log=z.read(dep+'/check.log').decode();assert 'Traceback' not in log and 'COUNTERFACTUAL_MECHANISM_CHECK_COMPLETE' in log
  steps=[json.loads(s) for s in log.splitlines() if s.startswith('{"check_step":')];assert [s['check_step'] for s in steps]==list(range(1,21))
  assert any(s['target_nonzero_fraction']>0 for s in steps) and any(s['utility_gradient_nonzero'] for s in steps)
  assert checks['status']=='COUNTERFACTUAL_MECHANISM_CHECK_COMPLETE' and checks['mode']==mode and checks['optimizer_updates']==20
  assert not checks['test_accessed'] and not checks['performance_experiment_started']
  for key in ['label_isolation_verified','initial_zero_target_verified','reference_and_utility_gradient_detach_verified','module_training_flags_restored','nonzero_reference_targets_after_updates_verified','nonzero_utility_gradient_after_updates_verified','shared_fixed_phase_verified','post_phase_mode_gradient_verified','batch_invariance_verified','mask_invariance_verified']:assert checks[key]
  assert protocol['train_samples']==1281 and protocol['valid_samples']==229 and protocol['seed']==91814 and protocol['pretrained_tensors_verified']==198
  for name,h in protocol['own_source_sha256'].items():assert hashlib.sha256(z.read(dep+'/'+name)).hexdigest()==h==hashlib.sha256((src/name).read_bytes()).hexdigest()
  orders=np.load(io.BytesIO(z.read(dep+'/check_'+mode+'/batch_orders.npy')),allow_pickle=False);assert orders.shape==(100,1280) and orders.dtype==np.dtype('<i8')
  assert hashlib.sha256(orders.tobytes()).hexdigest()==checks['batch_order_sha256']==protocol['batch_order_sha256']
  for row,h in zip(orders,protocol['batch_order_epoch_sha256']):assert hashlib.sha256(row.tobytes()).hexdigest()==h and len(set(row))==1280 and row.min()>=0 and row.max()<1281
  comparable={k:v for k,v in protocol.items() if k not in ('mode','gpu_uuid')}
  agreement={'protocol':comparable,'initial':checks['initial_model_sha256'],'flow':checks['initial_flow_sha256'],'after20':checks['after_shared_updates_model_sha256'],'orders':checks['batch_order_sha256']}
  if shared is None:shared=agreement
  else:assert shared==agreement,'Unmatched initialization/orders/shared updates'
  rows.append({'node':node,'checks':checks,'launch':launch,'captured_at':proof['captured_at'],'archive_sha256':proof['archive_sha256'],'members':len(manifest),'pid_retired':True,'actual_resources':resources})
result={'audited_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'THREE_MATCHED_GPU_MECHANISM_CHECKS_VERIFIED','rows':rows,'matched_full_initialization_orders_and_after20_state':True,'limits':'Mechanism only; no performance improvement or semantic truth established.'}
assert not c.out.exists();c.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'status':result['status'],'rows':len(rows)}))
