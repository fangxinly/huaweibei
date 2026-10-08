"""One fold0 fixed-budget mechanism trial. INNER input-only, no INNER labels."""
import argparse,datetime,os,pickle,sys,time
from pathlib import Path
import numpy as np
import torch
from contract import read,write,sha,split_calibration,video
from pipeline import Mechanism,subset,state_sha

def cpu(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return v
def run(a):
    assert sha(a.plan)==a.plan_sha;p=read(a.plan);assert p['real_train_enabled'] and p['scope']=='single_fold0_fixed_mechanism_trial'
    assert sha(a.cache)==p['cache_reference']['SHA'];split=read(a.bundle/'split.json');f=split['folds'][0];assert sha(a.bundle/'split.json')==p['split_SHA']
    with np.load(a.cache,allow_pickle=False) as z:
        ids=z['row_ids'].astype(str);features=[z[k].copy() for k in ('text','audio','vision')]
    assert ids.tolist()==split['canonical_row_ids'];pos={s:i for i,s in enumerate(ids)};fit=np.array([pos[s] for s in f['row_ids']['fit']]);inner=np.array([pos[s] for s in f['row_ids']['inner']])
    with (a.assets/'assets/mosi.pkl').open('rb') as file:c=pickle.load(file)
    records=list(c['train'])+list(c['dev'])+list(c['test']);del c
    original_ids=[r[2].decode() if isinstance(r[2],bytes) else r[2] for r in records];assert original_ids==ids.tolist()
    # Only the already frozen fold FIT may expose scalar labels. INNER/OUTER
    # labels are neither indexed nor returned by this training entry point.
    y=np.array([np.asarray(records[i][1]).reshape(-1)[0] for i in fit],dtype=np.float64);assert np.isfinite(y).all() and (abs(y)<=3).all();del records
    ti,ci=split_calibration(ids[fit],p['CAL_fraction'],128);tr=fit[ti];cal=fit[ci]
    assert not {video(ids[i]) for i in tr}&{video(ids[i]) for i in cal};assert not {video(ids[i]) for i in fit}&{video(ids[i]) for i in inner}
    assert not a.out.exists();a.out.mkdir();torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    write(a.out/'actual_role_identity.json',dict(fit_ids=ids[fit].tolist(),task_train_ids=ids[tr].tolist(),calibration_ids=ids[cal].tolist(),inner_ids=ids[inner].tolist(),whole_pickle_materialized=True,label_scalars_only_frozen_FIT=True,inner_labels_read=False,outer_labels_read=False))
    start=time.monotonic();model=Mechanism(subset(features,tr),y[ti],ids[tr],subset(features,cal),y[ci],ids[cal],p['config'],'cuda')
    complete=cpu(model.complete());torch.save(complete,a.out/'complete_small_training_state.pt');checkpoint_sha=sha(a.out/'complete_small_training_state.pt');pred,extra=model.predict(subset(features,inner))
    np.savez(a.out/'frozen_INNER_predictions.npz',row_ids=ids[inner],checkpoint_SHA=np.array(checkpoint_sha),**pred)
    np.savez(a.out/'frozen_INNER_gate_details.npz',**{name+'_'+k:v[k] for name,v in extra.items() if name!='conditional_donor_MSE' for k in ('rhat','delta','gate')})
    np.save(a.out/'original_FIT_supervision.npy',y,allow_pickle=False)
    # Fresh native replay from the actually written entire checkpoint.
    restored=Mechanism.restore(torch.load(a.out/'complete_small_training_state.pt',map_location='cpu'),'cuda');back,_=restored.predict(subset(features,inner));replay=max(float(np.max(abs(pred[k]-back[k]))) for k in pred);assert replay<1e-6
    parameter_counts={name:sum(x.numel() for x in module.parameters()) for name,module in [('baseline',model.stack.baseline),('flow',model.stack.decompositions['flow']),('regression',model.stack.decompositions['regression']),('proposal',model.stack.proposals['flow']),('gate',model.gates['flow'])]}
    write(a.out/'actual_train_receipt.json',dict(status='FIXED_FROZEN_FEATURE_INNOVATION_PILOT_PREDICTIONS_COMPLETE_UNSCORED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,
        plan_SHA=a.plan_sha,cache_SHA=sha(a.cache),checkpoint_SHA=checkpoint_sha,checkpoint_bytes=(a.out/'complete_small_training_state.pt').stat().st_size,prediction_SHA=sha(a.out/'frozen_INNER_predictions.npz'),prediction_replay_max_error=replay,
        rows=dict(task_train=len(tr),calibration=len(cal),inner=len(inner)),head_parameter_counts=parameter_counts,complete_training_records=len(complete['records']),calibration=model.calibration,
        no_INNER_OUTER_labels_indexed=True,original_TEST_not_a_holdout=True,elapsed_seconds=time.monotonic()-start,conditional_donor_MSE_on_INNER_inputs=extra['conditional_donor_MSE'],max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),max_memory_reserved_bytes=torch.cuda.max_memory_reserved(),no_peak_reset=True))
    print(read(a.out/'actual_train_receipt.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','bundle','assets','out','cache'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())
