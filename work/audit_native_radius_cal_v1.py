from pathlib import Path
import argparse,hashlib,json,numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
plan=read(r/'plan.json');receipt=read(r/'mechanism/receipt.json');ex=read(r/'mechanism_exit.json');la=read(r/'mechanism_launch.json')
assert ex['exit_code']==0 and ex['child_pid']==la['child_pid']==receipt['pid'] and la['argv']==la['actual_proc_argv']
assert receipt['status']=='CAL_ONLY_NATIVE_RADIUS_ZERO_LABEL_GPU_MECHANISM_COMPLETED' and receipt['source_sha256']==sha(r/'diagnose_native_radius_cal_v1.py')==plan['source_sha256']
assert receipt['plan_sha256']==sha(r/'plan.json') and receipt['rows']==418 and receipt['videos']==18
for k in ['real_labels_read','CAL_fit','EVAL_metrics','DEV_read','TEST_read','path_solver_rerun','selected_predictions_generated']:assert receipt[k] is False
assert receipt['EVAL_forward_rows']==0 and receipt['model_state_before']==receipt['model_state_after']==plan['expected_model_state']
assert receipt['parameter_gradients_all_None'] and receipt['optimizer_steps']==0 and not receipt['optimizer_created']
assert receipt['poisoned_label_error']==0 and receipt['single_token_single_batch_max_error']<=1e-6 and receipt['seconds']<plan['maximum_seconds'] and receipt['peak_allocated_bytes']<=plan['maximum_peak_allocated_bytes']
file=r/'mechanism/cal_candidate_arrays.npz';assert sha(file)==receipt['prediction_sha256'];z=np.load(file,allow_pickle=False)
assert np.array_equal(z['row'],np.load(r/'authorized_cal_rows.npy',allow_pickle=False)) and len(np.unique(z['video']))==18 and len(z['row'])==418
msg=z['candidate_messages'].astype(np.float64);f=z['original_messages'].astype(np.float64);sc=z['message_scale'].astype(np.float64)
assert np.array_equal(msg[:,0],f) and np.array_equal(msg[:,4],z['native_messages'])
shift=z['native_messages'].astype(np.float64)-f;rad=np.linalg.norm((shift/sc).reshape(418,-1),axis=1)
assert np.allclose(rad,z['normalized_native_radius'],rtol=1e-6,atol=1e-8)
for i,amp in enumerate([.125,.25,.5,1.]):
    for idx,expected in [(1+i,f+amp*shift),(9+i,f-amp*shift),(5+i,f+sc*rad[:,None,None]*amp*z['teacher_unit'].astype(np.float64))]:
        bound=12*np.finfo(np.float32).eps*(np.abs(f)+np.abs(expected)+1e-6)
        assert np.all(np.abs(msg[:,idx]-expected)<=bound),(idx,np.max(np.abs(msg[:,idx]-expected)))
assert np.max(np.abs(z['candidate_prediction'][:,0]-z['saved_pf']))<=1e-6 and np.max(np.abs(z['candidate_prediction'][:,4]-z['saved_native_prediction']))<=1e-6
pred=z['candidate_prediction'].astype(np.float64);delta=pred-pred[:,0,None];valid=np.abs(delta)<=z['cap'][:,None];valid[:,0]=True
assert np.array_equal(valid,z['candidate_valid']) and np.isfinite(pred).all() and np.isfinite(msg).all()
rel=np.linalg.norm(((msg-f[:,None])/sc[:,None]).reshape(418,13,-1),axis=2)/np.linalg.norm((f/sc).reshape(418,-1),axis=1)[:,None]
assert rel.max()<=.25001 and np.allclose(rel,z['relative_message_shift'],rtol=2e-4,atol=2e-7)
unique=np.ones((418,13),bool)
for i in range(1,13):
    for j in range(i):unique[:,i]&=~(msg[:,i]==msg[:,j]).all((1,2))
assert np.array_equal(unique,z['candidate_unique'])
for i,target in enumerate([z['target_old'],z['target_cal']]):assert np.array_equal(2*(pred[:,0]-target)[:,None]*delta+delta**2,z['proxy_risk_change'][:,i])
rows={int(row):i for i,row in enumerate(z['row'])};assert z['valid_lengths'][rows[454]]==z['valid_lengths'][rows[620]]==1
pools=[[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,9,10,11,12]];cover=[]
for ids in pools:
    eligible=(valid[:,ids]&unique[:,ids]);q=z['proxy_risk_change'][:,1][:,ids]
    neg=eligible&(q<0);neg[z['lambda_value']==0]=False
    cover.append({'mean_unique':float(unique[:,ids].sum(1).mean()),'mean_cap_valid_unique':float(eligible.sum(1).mean()),'prior_grid_mean_valid':float(z['prior_grid_candidate_valid'][:,ids].sum(1).mean()),'rows_negative_CAL_proxy':int(neg.any(1).sum()),'native_amp1_cap_valid_rows':int(valid[:,4].sum())})
proof={'status':'CAL_NATIVE_RADIUS_ORIGINAL_GPU_MECHANISM_NUMPY_VERIFIED','array_sha256':sha(file),'source_sha256':sha(r/'diagnose_native_radius_cal_v1.py'),'rows':418,'videos':18,
    'model_state_unchanged':True,'native_replay_max':receipt['saved_native_terminal_replay_max_error'],'F_replay_max':receipt['saved_F_terminal_replay_max_error'],'coverage_OT_and_Opm':cover,'new_risk_metrics':False,'labels_read':False,'CPU_model_forward':False}
a.out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
