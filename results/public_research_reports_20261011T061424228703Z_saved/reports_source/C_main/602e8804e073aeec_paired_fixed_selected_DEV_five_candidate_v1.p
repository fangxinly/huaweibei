"""Report all five metrics of the two already selected models. No selection or forward.

Requires both complete original CPU and fresh whole-model replay joints before
decoding the original selected DEV arrays. Existing DEV is descriptive only.
"""
import argparse, datetime, hashlib, json, sys
from pathlib import Path
import numpy as np
from sentiment_metrics_careflow_v1 import metrics, SEMANTICS

METHODS=('minimal_fixed_F','careflow')
def require(value,message):
    if not value: raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();require(sha(a.plan)==a.plan_sha,'Exact frozen descriptive DEV plan required')
    plan=read(a.plan)
    require(plan['status']=='PAIRED_ALREADY_SELECTED_DEV_FIVE_DESCRIPTIVE_PROTOCOL_FROZEN' and
            plan['methods']==list(METHODS) and plan['new_model_or_forward_or_final_TEST_enabled'] is False,
            'Wrong scope or unfrozen plan')
    require(a.out.resolve().is_relative_to(Path(plan['D_root']).resolve()) and not a.out.exists(),'Fresh once-only D result required')
    for n,h in plan['source_sha256'].items():require(sha(a.plan.parent/n)==h,'Frozen score source mismatch')
    parents={}
    # All gates for both methods complete before loading any target or prediction array.
    for method in METHODS:
        spec=plan['parents'][method];base=Path(spec['method_D_root'])
        for key in ('training_joint','fresh_joint'):
            require(sha(spec[key]['path'])==spec[key]['sha256'],'Exact complete parent joint SHA required')
        t=read(spec['training_joint']['path']);f=read(spec['fresh_joint']['path'])
        require(t['status']=='ACTUAL_PAIRED_METHOD_FULLTRAIN_GPU_D_OTHER_CPU_CAPTURE_JOINT_PASSED_NO_FINAL_TEST' and
                f['status']=='ACTUAL_PAIRED_FULLTRAIN_D_ORIGINAL_OTHER_CPU_AND_FRESH_PUBLIC_SELECTED_REPLAY_JOINT_PASSED_NO_FINAL_TEST',
                'Both original whole CPU and fresh replay qualified before metrics')
        require(t['method']==f['method']==method and t['plan_sha256']==f['plan_sha256']==plan['training_plan_sha256'] and
                f['whole_training_joint_sha256']==spec['training_joint']['sha256'] and
                t['whole_checkpoint_D_SHA_CRC_passed'] is True and f['all_original_model_and_array_gates_passed'] is True and
                t['final_TEST_executed'] is False and f['final_TEST_executed'] is False,'Parent scope mismatch')
        original=base/'a/run';r=read(original/'out/actual_stage_receipt.json')
        require(sha(original/'out/actual_stage_receipt.json')==t['original_stage_receipt_sha256']==spec['GPU_receipt_sha256'],
                'Original selected checkpoint receipt mismatch')
        require(r['metadata']['best_epoch']==spec['best_epoch'] and r['metadata']['best_state_SHA']==spec['best_state_SHA'] and
                r['metadata']['epochs']==100 and r['metadata']['optimizer_steps']==4000,'Fixed whole selected identity mismatch')
        for name,h in spec['arrays_sha256'].items():require(sha(original/'out'/name)==h,'Exact original DEV physical array SHA required')
        parents[method]=(original,r)
    rows={};labels=None;ids=None
    for method in METHODS:
        original,r=parents[method];identity=read(original/'out/actual_official_row_identity.json')
        y=np.load(original/'out/original_DEV_selection_targets.npy',allow_pickle=False)
        require(y.shape==(229,) and y.dtype==np.float64 and np.isfinite(y).all() and (np.abs(y)<=3).all(),
                'Original author FP32 targets cast FP64, without transformation')
        with np.load(original/'out/selected_best_DEV_replay.npz',allow_pickle=False) as z:
            require(set(z.files)=={'row_ids','prediction','model_state_sha256'} and z['prediction'].dtype==np.float32 and
                    z['prediction'].shape==(229,) and z['row_ids'].tolist()==identity['ids']['dev'] and
                    str(z['model_state_sha256'].item())==r['metadata']['best_state_SHA'],'Original selected replay ID/state/schema')
            prediction=z['prediction'].copy();this_ids=z['row_ids'].tolist()
        if labels is None:labels=y.copy();ids=this_ids
        else:require(np.array_equal(labels,y) and ids==this_ids,'Both models exact same DEV target/ID/order')
        values=metrics(prediction,y)
        # Different numerical form for independent Pearson and explicit per-class F1.
        matrix=np.asarray(values['nonzero_confusion_matrix']);support=matrix.sum(1)
        explicit=sum((2*matrix[i,i]/(matrix[i,:].sum()+matrix[:,i].sum()) if
                     matrix[i,:].sum()+matrix[:,i].sum() else 0)*support[i] for i in (0,1))/support.sum()
        corr=float(np.corrcoef(prediction.astype(np.float64),y)[0,1])
        require(abs(explicit-values['F1'])<=1e-14 and abs(corr-values['Corr'])<=1e-14,'Independent F1/Pearson aggregation mismatch')
        rows[method]={'fixed_best_epoch':r['metadata']['best_epoch'],'best_state_SHA':r['metadata']['best_state_SHA'],
                      'all_five_same_fixed_prediction':values}
    f=rows['minimal_fixed_F']['all_five_same_fixed_prediction'];c=rows['careflow']['all_five_same_fixed_prediction']
    keys=('Acc7','Acc2','F1','MAE','Corr')
    strict={k:(f[k]<c[k] if k=='MAE' else f[k]>c[k]) for k in keys}
    result={'status':'ACTUAL_TWO_ALREADY_SELECTED_DEV_FIVE_DESCRIPTIVE_ARRAY_AGGREGATION_COMPLETE_NO_FINAL_TEST',
            'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'argv':sys.argv,'frozen_plan_sha256':a.plan_sha,'rows':rows,'semantics':SEMANTICS,
            'F_minus_C_same_DEV':{k:f[k]-c[k] for k in keys},'strict_improvement_each':strict,
            'all_five_strict_on_already_explored_DEV':all(strict.values()),
            'checkpoint_or_readout_selected_using_these_scores':False,'new_GPU_or_CPU_model_forward':False,
            'new_TEST_or_201_labels_accessed':False,'independent_confirmation_or_paper_TEST_superiority':False,
            'TRAIN_DEV_history_already_explored':True,'overall_five_metric_goal_complete':False}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
