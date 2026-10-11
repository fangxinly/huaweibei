from pathlib import Path
import argparse,json,hashlib,zipfile,datetime,io
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest();rows={};initial=[]
uuids=['GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'];prefix='group_teacher/'
for fold,node in enumerate(['a','b','c']):
 d=a.directory/node;receipt=json.loads((d/'receipt.json').read_text());raw=(d/'snapshot.zip').read_bytes();assert len(raw)==receipt['bytes'] and sha(raw)==receipt['sha256'];assert (d/'exit_code.txt').read_text().strip()=='0'
 with zipfile.ZipFile(d/'snapshot.zip') as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  members=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(members)|{'member_manifest.json'}
  for n,m in members.items():
   b=z.read(n);assert sha(b)==m['sha256'] and len(b)==m['bytes']
  get=lambda n:json.loads(z.read(prefix+n));load=lambda n:np.load(io.BytesIO(z.read(prefix+n)),allow_pickle=False)
  plan=get('group_teacher_plan_v3.json');formal=get('group_teacher_formal_plan_v4.json');audit=get('three_fold_gpu_mechanism_audit.json');pre=get('precheck_v2/receipt.json');info=plan['folds'][fold]
  assert sha(z.read(prefix+'group_teacher_plan_v3.json'))==formal['parent_plan_sha256']==audit['plan_sha256']
  assert sha(z.read(prefix+'train_group_teacher_v1.py'))==formal['trainer_sha256']==sha((w/'train_group_teacher_v1.py').read_bytes())
  assert sha(z.read(prefix+'three_fold_gpu_mechanism_audit.json'))==formal['mechanism_audit_sha256']
  assert sha(z.read(prefix+'formal_space_authorization.json'))==formal['space_authorization_sha256']
  for n,h in plan['source_sha256'].items():assert sha(z.read(prefix+n))==h
  for n,m in info['files'].items():assert members[prefix+n]==m
  fit=load(f'fit_{fold}.npy');inner=load(f'inner_{fold}.npy');outer=load(f'outer_{fold}.npy');orders=load(f'orders_{fold}.npy')
  assert orders.shape==(100,len(fit)) and all(np.array_equal(np.sort(row),fit) for row in orders)
  assert not(set(fit)&set(inner) or set(fit)&set(outer) or set(inner)&set(outer))
  inv=json.loads(z.read('group_teacher_inventory.json'));assert inv['gpu'].split(',')[0].strip()==uuids[fold]
  launch=get('formal_launch.json');argv=launch['argv'];assert argv[1]=='/data/coding/group_teacher_v1_20261005T1650Z/train_group_teacher_v1.py' and argv[-4:]==['--fold',str(fold),'--uuid',uuids[fold]]
  history=get('run_v1/history.json') if prefix+'run_v1/history.json' in members else []
  assert [r['epoch'] for r in history]==list(range(1,len(history)+1)) and len(history)<=100
  for i,h in enumerate(history):assert h['best_epoch']==min(range(1,i+2),key=lambda e:history[e-1]['inner_sample_mse'])
  protocol=get('run_v1/protocol.json') if prefix+'run_v1/protocol.json' in members else None
  if protocol:
   assert protocol['fold']==fold and protocol['seed']==91818 and protocol['epochs']==100 and protocol['retained_parameter_tensors']==291 and protocol['trainable_parameters']==184749003
   assert not protocol['outer_labels_read'] and not protocol['dev_requested'] and not protocol['test_requested']
   assert protocol['initial_full_tensor_sha256']==pre['initial_full_tensor_sha256'] and protocol['initial_non_normalization_tensor_sha256']==audit['common_non_normalization_initial_tensor_sha256']
   assert protocol['formal_plan_sha256']==sha(z.read(prefix+'group_teacher_formal_plan_v4.json')) and protocol['orders_sha256']==members[prefix+f'orders_{fold}.npy']['sha256']
   initial.append(protocol['initial_non_normalization_tensor_sha256'])
  completion=get('run_v1/completion.json') if prefix+'run_v1/completion.json' in members else None
  if completion:
   assert len(history)==100 and completion['completed_epochs']==100 and completion['fold']==fold
   assert not completion['outer_labels_read'] and not completion['dev_requested'] and not completion['test_requested']
   assert completion['best_epoch']==min(range(1,101),key=lambda e:history[e-1]['inner_sample_mse'])
   assert completion['inner_strict_disk_replay_max_error']<=1e-6
   with load('run_v1/outer_scalar_predictions.npz') as out:assert np.array_equal(out['row_ids'],outer) and out['mu'].shape==(len(outer),) and np.isfinite(out['mu']).all()
   large=json.loads(z.read('large_file_manifest.json'));m=large[prefix+'run_v1/selected_full_checkpoint.pt'];assert m['sha256']==completion['full_checkpoint_sha256'] and m['bytes']==completion['full_checkpoint_bytes']
   state='TRAIN100_COMPLETE_FULL_LOCAL_AND_INDEPENDENT_CPU_PRESERVATION_PENDING'
  else:
   assert len(history)<100 and len(inv['matching_processes'])==1
   assert inv['matching_processes'][0]['argv']==argv
   assert prefix+'formal_exit.json' not in members
   state='LIVE_TRAINING' if history else 'LIVE_INITIALIZATION_NOT_YET_ONE_EPOCH'
  rows[node]={'state':state,'epochs':len(history),'capture_utc':receipt['utc'],'launch':launch,'actual_matching_processes':inv['matching_processes'],'gpu':inv['gpu'],'compute':inv['compute'],'last_epoch':history[-1] if history else None,'checkpoint_fresh_sha_is_not_local_download':True}
assert len(set(initial))<=1
result=dict(status='FORMAL_THREE_FOLD_TEACHER_SOURCE_ORDERS_INITIALIZATION_AND_ACTUAL_SNAPSHOT_STATES_AUDITED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),nodes=rows,whole_pipeline_crossfit=False,local_full_weights_verified_by_this_audit=False,independent_CPU_completed_weight_save_verified_by_this_audit=False)
assert not a.out.exists();a.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({n:{'state':r['state'],'epochs':r['epochs']} for n,r in rows.items()}))
