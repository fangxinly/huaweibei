"""Independent CPU audit of actual future GPU receipts; never substitutes for them."""
from pathlib import Path
import argparse,json,hashlib,datetime,math
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--plan-dir',type=Path,required=True)
p.add_argument('--receipts-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--estimated-lease-end',default='2026-10-06T12:08:17+00:00')
a=p.parse_args()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
plan_path=a.plan_dir/'group_teacher_plan_v3.json'
plan=json.loads(plan_path.read_text(encoding='utf-8'))
expected_uuids=['GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa']
all_receipts=[];estimates=[]
for fold in range(3):
 directory=a.receipts_dir/['a','b','c'][fold]/'precheck_v2'
 receipt_path=directory/'receipt.json'
 if not receipt_path.is_file():raise FileNotFoundError('Actual original GPU receipt required: '+str(receipt_path))
 r=json.loads(receipt_path.read_text(encoding='utf-8'));info=plan['folds'][fold]
 assert r['status']=='GROUP_TEACHER_FIT_ONLY_NORMALIZATION_OUTER_GUARD_PUBLIC_INIT_FULL_DISK_RELOAD_AND_FOUR_REAL_UPDATES_PASSED'
 assert r['fold']==fold and r['seed']==91818
 assert r['gpu'].split(',')[0].strip()==expected_uuids[fold] and not r['initial_compute']
 assert r['plan_sha256']==sha(plan_path)
 assert r['source_sha256']==plan['source_sha256']['check_group_teacher_v2.py']
 assert r['runtime_sha256']==plan['source_sha256']['group_teacher_runtime_v2.py']
 assert r['all_retained_parameter_gradients_finite'] and r['retained_parameter_tensors']==291 and r['all_other_initial_tensors_equal_v1'] and r['original_v1_initial_prediction_error']==0
 assert r['outer_access_denied'] is True and r['real_parameter_update'] is True
 assert all(r[key]==0 for key in ['fit_label_replacement_error','inner_label_replacement_error','strict_disk_reload_error'])
 assert all(r[key] is False for key in ['outer_predictions_generated','dev_requested','test_requested'])
 assert r['pretrained_matched_tensors']>=190 and r['trainable_parameters']>100000000
 assert r['initial_checkpoint_bytes']>400000000 and r['initial_full_state_tensor_count']>190
 assert len(r['initial_checkpoint_sha256'])==64
 for role in ['fit','inner','outer']:assert r[role+'_rows']==info[role+'_rows']
 journal=r['data_access_journal'];assert len(journal)==2
 for entry,role in zip(journal,['fit','inner']):
  assert entry['role']==role and entry['label_access'] is True
  assert np.array_equal(entry['rows'],np.load(a.plan_dir/f'{role}_{fold}.npy'))
 assert len(r['gradient_l1'])==4 and len(r['update_seconds'])==4
 for gradients in r['gradient_l1']:
  assert set(gradients)=={'text','audio','visual','audio_transformer','visual_transformer','fusion','decoder'}
  assert all(np.isfinite(x) and x>0 for x in gradients.values())
 assert all(np.isfinite(x) and x>0 for x in r['update_seconds'])
 assert np.isfinite(r['inner128_two_forwards_seconds']) and r['inner128_two_forwards_seconds']>0
 total_mib=float(r['gpu'].split(',')[2].strip().split()[0])
 assert 0<r['peak_allocated_bytes']<total_mib*1024**2
 witness_path=directory/'normalization_witness.npz'
 assert sha(witness_path)==r['normalization_witness_sha256']
 with np.load(witness_path,allow_pickle=False) as z:
  assert set(z.files)=={name+'_'+field for name in ['audio','visual'] for field in ['raw_fit_content_tokens','mean','std','active']}
  for name in ['audio','visual']:
   values=z[name+'_raw_fit_content_tokens'].astype(np.float64)
   assert values.ndim==2 and values.shape[0]>=info['fit_rows'] and np.isfinite(values).all()
   mean=values.mean(axis=0);std=values.std(axis=0,ddof=0)
   np.testing.assert_allclose(z[name+'_mean'],mean.astype(np.float32),rtol=2e-6,atol=2e-6)
   np.testing.assert_allclose(z[name+'_std'],np.maximum(std,1e-6).astype(np.float32),rtol=2e-6,atol=2e-6)
   assert np.array_equal(z[name+'_active'],std>=1e-6)
 with np.load(directory/'initial_replay_predictions.npz',allow_pickle=False) as z:
  assert set(z.files)=={'prediction','fit_row_ids'} and z['prediction'].shape==(32,) and np.isfinite(z['prediction']).all()
  assert np.array_equal(z['fit_row_ids'],np.load(a.plan_dir/f'orders_{fold}.npy')[0,:32])
 # Four updates are a pilot rather than a guarantee; double the measured worst
 # step/evaluation cost and explicitly budget setup + strict reload + outer inference.
 updates=info['total_updates'];inner_batches=math.ceil(info['inner_rows']/128)
 estimated=2*(updates*max(r['update_seconds'])+100*inner_batches*r['inner128_two_forwards_seconds']+r['elapsed_seconds'])
 estimates.append({'fold':fold,'seconds_conservative_pilot_estimate':estimated,'peak_allocated_bytes':r['peak_allocated_bytes'],'original_receipt_sha256':sha(receipt_path)})
 all_receipts.append(r)
assert len({r['initial_non_normalization_tensor_sha256'] for r in all_receipts})==1
assert len({r['initial_full_state_tensor_count'] for r in all_receipts})==1
assert len({r['trainable_parameters'] for r in all_receipts})==1
now=datetime.datetime.now(datetime.timezone.utc)
lease_end=datetime.datetime.fromisoformat(a.estimated_lease_end)
assert lease_end.tzinfo is not None
remaining=(lease_end-now).total_seconds()
longest=max(e['seconds_conservative_pilot_estimate'] for e in estimates)
if remaining<longest+7200:raise RuntimeError('Measured pilot budget does not leave two-hour preservation margin against estimated lease end')
result={'status':'THREE_ORIGINAL_GPU_MECHANISM_RECEIPTS_INDEPENDENT_CPU_AUDIT_PASSED','utc':now.isoformat(),'plan_sha256':sha(plan_path),'common_non_normalization_initial_tensor_sha256':all_receipts[0]['initial_non_normalization_tensor_sha256'],'folds':estimates,'estimated_lease_end':lease_end.isoformat(),'platform_deadline_verified':False,'remaining_seconds_at_audit':remaining,'preservation_margin_seconds':7200,'budget_is_pilot_estimate_not_guarantee':True,'full_initial_weights_downloaded_or_cpu_reloaded_by_this_audit':False,'formal_training_started_by_this_audit':False,'fresh_remote_and_local_disk_check_still_required_before_launch':True}
a.output.parent.mkdir(parents=True,exist_ok=True)
assert not a.output.exists()
a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result['status'])
