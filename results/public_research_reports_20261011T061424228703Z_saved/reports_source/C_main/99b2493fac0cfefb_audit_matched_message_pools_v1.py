"""Independent NumPy proof of original phase arrays and real process receipts."""
from pathlib import Path
import argparse,hashlib,json,numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['precheck','execute'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(a.root/'plan.json');r=read(a.root/a.phase/'receipt.json');ex=read(a.root/(a.phase+'_exit.json'));launch=read(a.root/(a.phase+'_launch.json'))
assert ex['exit_code']==0 and ex['child_pid']==launch['child_pid']==r['pid']
assert launch['argv']==launch['actual_proc_argv'] and launch['argv'][2:]==r['argv']
assert r['source_sha256']==sha(a.root/'diagnose_matched_message_pools_v1.py')==plan['source_sha256']
assert r['plan_sha256']==sha(a.root/'plan.json') and r['prediction_sha256']==sha(a.root/a.phase/'predictions_frozen.npz')
assert r['model_state_before']==r['model_state_after'] and r['no_parameter_gradients'] and r['optimizer_steps']==0 and not r['optimizer_created']
assert r['real_labels_read'] is False and r['seconds']<plan['maximum_seconds'] and r['actual_peak_allocated_bytes']<=plan['maximum_peak_allocated_bytes']
for e in r['evidence']:
    assert e['label_replacement_max_error']==0 and max(e['replay_errors'].values())<=1e-6
assert all(e['relative_error']<1e-4 for e in r['double_finite_difference_hvp'])
assert r['single_row_batch_replay_max_error']<=1e-6
z=np.load(a.root/a.phase/'predictions_frozen.npz',allow_pickle=False);v={k:z[k] for k in z.files};z.close()
assert not set(v)&{'y','label','labels','oracle'}
rows=v['row'];assert len(rows)==r['rows']
expected=np.arange(1281) if a.phase=='execute' else np.r_[np.arange(32),620,621,1280]
assert np.array_equal(rows,expected)
with np.load(a.root/'roles.npz',allow_pickle=False) as roles:
    for k in ['fold','video','role']:assert np.array_equal(v[k],roles[k][rows])
with np.load(a.root/'mu_input.npz',allow_pickle=False) as mu:assert np.array_equal(v['mu'],mu['mu'][rows])
params=read(a.root/'cal_parameters.json');assert np.array_equal(v['lambda_value'],np.asarray(params['lambda_by_fold'])[v['fold']])
assert np.array_equal(v['cap'],np.asarray(params['cap_by_fold'])[v['fold']])
pf=v['pf'].astype(np.float64);pr=v['candidate_prediction'].astype(np.float64);delta=pr-pf[:,None]
valid=np.isfinite(pr)&np.isfinite(v['candidate_messages']).all((2,3))&(np.abs(delta)<=v['cap'][:,None]);valid[:,0]=True
assert np.array_equal(valid,v['candidate_valid']) and np.array_equal(pr[:,0],pf)
assert np.array_equal(v['candidate_messages'][:,0],v['original_messages'])
assert np.max(np.abs(v['target_cal']-(pf+v['lambda_value']*(v['mu'].astype(np.float64)-pf))))<1e-12
assert np.max(np.abs(v['target_old']-(v['p0']-v['rho_old']).astype(np.float64)))==0
pools=[[0,1,2,3,4],[0,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,9,10,11,12]]
for arm in range(8):
    ids=np.asarray(pools[arm%4]);t=v['target_old'] if arm<4 else v['target_cal']
    q=(2*(pf-t)[:,None]*delta+delta**2)[:,ids]
    score=np.where(valid[:,ids],q,np.inf);score[:,0]=0
    idx=ids[np.argmin(score,axis=1)];idx[np.min(score,axis=1)>=0]=0
    if arm>=4:idx[v['lambda_value']==0]=0
    assert np.array_equal(idx,v['selected_index'][:,arm]),arm
    chosen=v['candidate_prediction'][np.arange(len(rows)),idx]
    assert np.array_equal(chosen,v['selected_prediction'][:,arm])
    assert np.max(np.abs(chosen-v['selected_replay'][:,arm]))<=1e-6
    assert (np.abs(chosen.astype(np.float64)-pf)<=v['cap']).all()
for i in range(13):
    duplicate=np.zeros(len(rows),dtype=bool)
    for j in range(i):duplicate|=(v['candidate_messages'][:,i]==v['candidate_messages'][:,j]).all((1,2))
    assert np.array_equal(duplicate,v['candidate_duplicate'][:,i])
for start,direction in [(1,v['old_unit']),(5,v['teacher_unit']),(9,-v['old_unit'])]:
    norm=np.linalg.norm(direction.reshape(len(rows),-1),axis=1)
    assert np.all((norm<1e-8)|(np.abs(norm-1)<1e-5))
    for i,amp in enumerate([.125,.25,.5,1.]):
        expected=v['original_messages']+v['message_scale']*v['normalized_radius'][:,None,None]*amp*direction
        assert np.max(np.abs(expected-v['candidate_messages'][:,start+i]))<1e-6
relative=np.linalg.norm(((v['candidate_messages']-v['original_messages'][:,None])/v['message_scale'][:,None]).reshape(len(rows),13,-1),axis=2)/np.maximum(np.linalg.norm((v['original_messages']/v['message_scale']).reshape(len(rows),-1),axis=1),1e-12)[:,None]
assert np.max(np.abs(relative-v['relative_message_shift']))<1e-5 and relative.max()<.25001
if a.phase=='execute':assert max(r['maximum_replay_errors'].values())<=1e-6 and len(np.unique(v['video']))==52
proof=dict(status='MATCHED_MESSAGE_POOLS_ORIGINAL_ARRAYS_INDEPENDENT_NUMPY_AUDIT_PASSED',phase=a.phase,rows=len(rows),arrays_sha256=r['prediction_sha256'],original_pid=r['pid'],original_argv=launch['actual_proc_argv'],original_exit0=True,model_forward_performed_by_auditor=False,
    all_eight_selectors_reconstructed=True,nominal_candidates=13,main_pools_equal_nominal_budget=9,maximum_relative_message_shift=float(relative.max()),maximum_selected_terminal_replay_error=float(np.max(np.abs(v['selected_prediction']-v['selected_replay']))),all_original_numerical_parameters_frozen=True,real_y_absent_from_predictor_arrays=True)
a.out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
