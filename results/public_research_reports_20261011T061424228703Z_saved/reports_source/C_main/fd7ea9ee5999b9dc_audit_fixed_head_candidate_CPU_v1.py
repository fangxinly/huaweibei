"""Independent NumPy reconstruction, no task labels, Torch or model forward."""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.run);out=root/'out';bundle=root/'source'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
r=read(out/'actual_candidate_collection_receipt.json');ex=read(root/'natural_exit.json');plan=read(bundle/'candidate_collection_plan.json')
assert ex['exit_code']==0 and ex['natural_wait_verified'] and ex['child_pid']==r['pid'] and ex['child_full_argv'][1:]==r['argv']
assert ex['original_receipt_sha256']==sha(out/'actual_candidate_collection_receipt.json')
assert sha(out/'original_headFIT232_label_free.npz')==r['original_array_sha256'] and r['plan_sha256']==sha(bundle/'candidate_collection_plan.json')
assert r['checkpoint_state_before_after_sha256']==plan['checkpoint_state_sha256'] and r['FIT_statistics_before_after_sha256']==plan['FIT_statistics_sha256']
assert r['rng_unchanged'] and not r['task_labels_read'] and not r['head_fit_or_optimizer_updates'] and not r['headEVAL201_inputs_or_labels_used'] and not r['actual_performance_or_gain_measured']
assert r['forward_calls']==34 and r['batch_sizes']==[32]*7+[8] and all(not x['labels_read'] for x in r['input_access_journal'])
for n,h in plan['source_and_role_sha256'].items():assert sha(bundle/n)==h
rows=np.load(bundle/'head_development_fit_0.npy',allow_pickle=False);mp=read(bundle/'train_row_video_mapping.json')
with np.load(out/'original_headFIT232_label_free.npz',allow_pickle=False) as z:
 expected={'row_ids','video_ids','segment_ids','pF','pC','pC_repeat','pF_restored','delta','W_raw','F_repeat_errors','C_repeat_errors','F_acc2_class','C_acc2_class','F_acc7_class','C_acc7_class'}
 assert set(z.files)==expected and np.array_equal(z['row_ids'],rows)
 assert z['video_ids'].tolist()==[mp[int(i)]['video_id'] for i in rows] and z['segment_ids'].tolist()==[mp[int(i)]['segment_id'] for i in rows]
 arrays={n:z[n].copy() for n in z.files}
for n in ['pF','pC','pC_repeat','pF_restored']:assert arrays[n].dtype==np.float32 and arrays[n].shape==(232,) and np.isfinite(arrays[n]).all()
f=arrays['pF'].astype(np.float64);c=arrays['pC'].astype(np.float64);delta=c-f;v=arrays['video_ids']
assert np.array_equal(delta,arrays['delta']) and arrays['delta'].dtype==np.float64 and np.array_equal(4*delta**2,arrays['W_raw'])
fe=np.abs(f-arrays['pF_restored']);ce=np.abs(c-arrays['pC_repeat']);assert np.array_equal(fe,arrays['F_repeat_errors']) and np.array_equal(ce,arrays['C_repeat_errors']) and max(fe.max(),ce.max())<=1e-6
weights=np.asarray([1/(9*np.sum(v==x)) for x in v]);wmass=weights*delta**2;total=float(wmass.sum());vmass=np.asarray([wmass[v==x].sum() for x in np.unique(v)])
ess=None if total==0 else float(1/np.sum((vmass/total)**2));diag=r['diagnostics']
assert abs(total-diag['video_equal_delta_squared_mass'])<1e-15
if ess is None:assert diag['W_effective_video_mass'] is None
else:assert abs(ess-diag['W_effective_video_mass'])<1e-12
assert diag['W_empirical_feasibility_pass']==(total>0) and diag['exact_nonzero_delta_rows']==int(np.sum(delta!=0))
for name,aa,bb in [('acc2',f>=0,c>=0),('acc7',np.round(np.clip(f,-3,3)),np.round(np.clip(c,-3,3)))]:
 assert np.array_equal(aa,arrays['F_'+name+'_class']) and np.array_equal(bb,arrays['C_'+name+'_class'])
 assert diag['endpoint_'+name+'_class_diff_rows']==int(np.sum(aa!=bb))
assert max(x['dummy0_vs7_error'] for x in r['dummy_label_checks'])==0
assert all(max(x['first_euler_errors'])==0 and x['sequence']==['F','C','C','F'] for x in r['semantic_batch_checks'])
original_B=Path('/data/coding/staged_reference100_cpu_audit_20261006T184140Z')
original=read(original_B/'cpu_original_receipt.json');original_exit=read(original_B/'cpu_original_receipt.actual_exit.json')
assert original_exit['exit_code']==0 and original_exit['natural_wait_verified'] and original['best_epoch']==41
assert sha(original_B/'run/out/selected_best_full.pt')==plan['checkpoint_file_sha256'] and original['states']['selected_best_full.pt']['state_sha256']==plan['checkpoint_state_sha256']
record={'status':'ACTUAL_HEADFIT232_ORIGINAL_ARRAY_SCOPE_SOURCE_REFERENCE_CPU_AUDIT_PASSED','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),
 'original_collection_receipt_sha256':sha(out/'actual_candidate_collection_receipt.json'),'original_GPU_natural_exit_sha256':sha(root/'natural_exit.json'),'original_array_sha256':sha(out/'original_headFIT232_label_free.npz'),
 'original_B_reference_full_checkpoint_fresh_sha256':plan['checkpoint_file_sha256'],'original_B_complete_reference_CPU_receipt_sha256':sha(original_B/'cpu_original_receipt.json'),
 'rows':232,'videos':9,'max_repeat_error':float(max(fe.max(),ce.max())),'delta_squared_mass':total,'W_effective_video_mass':ess,'GPU_used':False,'CPU_model_forward':False,'task_labels_read':False,'new_performance_or_head_fit':False}
Path(a.out).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
