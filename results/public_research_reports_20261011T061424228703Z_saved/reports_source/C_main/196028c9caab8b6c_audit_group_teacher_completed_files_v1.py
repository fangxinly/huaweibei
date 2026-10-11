"""Independent completion/arrays audit. Does not import the training runtime."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,zipfile
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

REQUIRED=['run_v1/selected_full_checkpoint.pt','run_v1/protocol.json','run_v1/history.json','run_v1/selected_inner_predictions.npz','run_v1/outer_scalar_predictions.npz','run_v1/completion.json','formal_exit.json']
SOURCE_PINS={'train_group_teacher_v1.py':'7c94f90ff648909e0abe1a6ccdb7a88543650142c6633ebadb8a0062068b8791','group_teacher_runtime_v2.py':'6f9074453ed15c32537efb7803210801f6123cc1647c6f4fe4d7126e2616dc9e','group_teacher_plan_v3.json':'f0878166da09433c27119a08fdfd8682f9b9e29f2bfe0a758f29eee45cf131a4','group_teacher_formal_plan_v4.json':'9d06176ababb06bc0c97daf4797b20a535d30f40cd8eb9e9e20fdabb81367bbe'}

def audit(directory,fold,verify_full=True):
 d=Path(directory);get=lambda name:json.loads((d/name).read_text(encoding='utf-8'))
 h=get('run_v1/history.json')
 assert len(h)==100 and [r['epoch'] for r in h]==list(range(1,101)),'Not an actual completed 100-epoch teacher'
 assert all(math.isfinite(r['inner_sample_mse']) and math.isfinite(r['fit_sample_mse']) for r in h)
 assert all(r['best_epoch']==min(range(1,i+2),key=lambda e:h[e-1]['inner_sample_mse']) for i,r in enumerate(h))
 c=get('run_v1/completion.json');p=get('run_v1/protocol.json');plan=get('group_teacher_plan_v3.json');formal=get('group_teacher_formal_plan_v4.json')
 assert get('formal_exit.json')['exit_code']==0
 assert c['status']=='GROUP_TEACHER_100_INNER_SELECTED_FULL_RELOADED_OUTER_SCALAR_PREDICTIONS_COMPLETE'
 assert c['fold']==p['fold']==fold and c['completed_epochs']==p['epochs']==100 and p['seed']==91818
 assert c['best_epoch']==min(range(1,101),key=lambda e:h[e-1]['inner_sample_mse'])
 assert c['best_inner_sample_mse']==h[c['best_epoch']-1]['inner_sample_mse']
 assert c['inner_strict_disk_replay_max_error']<=1e-6
 assert c['protocol_sha256']==sha(d/'run_v1/protocol.json')
 assert p['trainer_sha256']==formal['trainer_sha256']==sha(d/'train_group_teacher_v1.py')
 assert p['plan_sha256']==formal['parent_plan_sha256']==sha(d/'group_teacher_plan_v3.json')
 assert p['formal_plan_sha256']==sha(d/'group_teacher_formal_plan_v4.json')
 assert p['mechanism_audit_sha256']==formal['mechanism_audit_sha256']==sha(d/'three_fold_gpu_mechanism_audit.json')
 assert p['space_authorization_sha256']==formal['space_authorization_sha256']==sha(d/'formal_space_authorization.json')
 assert p['retained_parameter_tensors']==291 and p['trainable_parameters']==184749003
 assert p['unused_pooler_removed'] and p['all_retained_gradients_required_every_update']
 assert p['initial_non_normalization_tensor_sha256']=='3813506519a54e1a7baa54ee37446b00e6b2f71f3a234621da1328db270a9567'
 for obj in [c,p]:assert not obj['outer_labels_read'] and not obj['dev_requested'] and not obj['test_requested']
 assert [entry['role'] for entry in c['data_access_journal']]==['fit','inner','outer_prediction_only_after_selected100']
 assert not c['data_access_journal'][-1]['label_access']
 for n,v in SOURCE_PINS.items():assert sha(d/n)==v
 for n,v in plan['source_sha256'].items():assert sha(d/n)==v
 info=plan['folds'][fold]
 mapping=get('train_row_video_mapping.json');assert sha(d/'train_row_video_mapping.json')==plan['mapping_receipt']['mapping_sha256']
 assert len(mapping)==1281 and [r['row'] for r in mapping]==list(range(1281)) and len({r['video_id'] for r in mapping})==52
 for n,v in info['files'].items():assert (d/n).stat().st_size==v['bytes'] and sha(d/n)==v['sha256']
 fit=np.load(d/f'fit_{fold}.npy',allow_pickle=False);inner=np.load(d/f'inner_{fold}.npy',allow_pickle=False);outer=np.load(d/f'outer_{fold}.npy',allow_pickle=False);orders=np.load(d/f'orders_{fold}.npy',allow_pickle=False)
 assert orders.shape==(100,len(fit)) and all(np.array_equal(np.sort(row),fit) for row in orders)
 assert not(set(fit)&set(inner) or set(fit)&set(outer) or set(inner)&set(outer)) and set(fit)|set(inner)|set(outer)==set(range(1281))
 for role,ids in [('fit',fit),('inner',inner),('outer',outer)]:
  assert {mapping[int(i)]['video_id'] for i in ids}==set(info[role+'_videos'])
  assert c['data_access_journal'][{'fit':0,'inner':1,'outer':2}[role]]['rows']==ids.tolist()
 assert not(set(info['fit_videos'])&set(info['inner_videos']) or set(info['fit_videos'])&set(info['outer_videos']) or set(info['inner_videos'])&set(info['outer_videos']))
 assert p['fit_rows']==len(fit)==info['fit_rows'] and p['inner_rows']==len(inner)==info['inner_rows'] and p['outer_rows']==c['outer_rows']==len(outer)==info['outer_rows']
 assert p['orders_sha256']==sha(d/f'orders_{fold}.npy')
 with np.load(d/'run_v1/selected_inner_predictions.npz',allow_pickle=False) as z:
  assert set(z.files)=={'prediction','label','row_ids'} and np.array_equal(z['row_ids'],inner)
  assert z['prediction'].shape==z['label'].shape==(len(inner),) and np.isfinite(z['prediction']).all() and np.isfinite(z['label']).all()
  mse=float(np.mean((z['prediction'].astype(np.float64)-z['label'].astype(np.float64))**2));assert abs(mse-c['best_inner_sample_mse'])<=1e-12
 with np.load(d/'run_v1/outer_scalar_predictions.npz',allow_pickle=False) as z:
  assert set(z.files)=={'row_ids','mu'} and np.array_equal(z['row_ids'],outer) and z['mu'].shape==(len(outer),) and np.isfinite(z['mu']).all()
 assert sha(d/'run_v1/outer_scalar_predictions.npz')==c['outer_predictions_sha256']
 if verify_full:
  checkpoint=d/'run_v1/selected_full_checkpoint.pt'
  assert checkpoint.stat().st_size==c['full_checkpoint_bytes'] and sha(checkpoint)==c['full_checkpoint_sha256']
  with zipfile.ZipFile(checkpoint) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 return {'status':'COMPLETED100_INNER_SELECTION_AND_ORIGINAL_ARRAYS_SOURCE_ORDERS_AUDITED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fold':fold,'best_epoch':c['best_epoch'],'best_inner_mse':mse,'outer_rows':len(outer),'full_file_sha_crc_checked':verify_full,'full_model_tensor_sha_checked_here':False,'CPU_preservation_claimed_here':False,'whole_pipeline_crossfit':False}

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--fold',type=int,choices=[0,1,2],required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 r=audit(args.directory,args.fold);assert not args.out.exists();args.out.write_text(json.dumps(r,indent=2),encoding='utf-8');print(r['status'])
