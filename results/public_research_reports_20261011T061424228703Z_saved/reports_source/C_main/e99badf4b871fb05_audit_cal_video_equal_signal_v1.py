"""Independent weighted-dot-product reconstruction from CAL-only numeric arrays."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
plan=read(a.root/'plan.json');res=read(a.root/'execute/calibration_result.json');receipt=read(a.root/'execute/receipt.json')
assert plan['source_sha256']==sha(a.root/'calibrate_video_equal_residual_v1.py')==res['source_sha256']==receipt['source_sha256']
assert receipt['plan_sha256']==sha(a.root/'plan.json') and receipt['result_sha256']==sha(a.root/'execute/calibration_result.json')
assert receipt['arrays_sha256']==sha(a.root/'execute/calibration_inputs.npz')==res['inputs_sha256']
assert receipt['real_eval_values_decoded']==0 and not receipt['model_forward_or_training']
for phase in ['precheck','execute']:
    ex=read(a.root/(phase+'_exit.json'));runtime=read(a.root/phase/'runtime.json')
    assert ex['exit_code']==0 and ex['original_runtime_argv_and_pid_match'] and ex['child_pid']==runtime['pid']
    assert ex['argv'][1:]==runtime['actual_python_argv']
with np.load(a.root/'row_fold_calibration_roles.npz',allow_pickle=False) as roles:
    allowed=roles['row_ids'][roles['role']=='calibration'];folds=roles['fold'];videos=roles['video']
with np.load(a.root/'execute/calibration_inputs.npz',allow_pickle=False) as z:v={k:z[k] for k in z.files}
assert set(v)=={'row','fold','video','pf','mu','y','old_prediction'}
assert np.array_equal(np.sort(v['row']),allowed) and len(v['row'])==418 and len(np.unique(v['video']))==18
assert np.array_equal(v['fold'],folds[v['row']]) and np.array_equal(v['video'],videos[v['row']])
assert sorted(res['outcome_decoded_row_ids'])==allowed.tolist() and not res['eval_metrics_computed']
def weights(video):
    names,counts=np.unique(video,return_counts=True);lookup=dict(zip(names,counts))
    return np.asarray([1/(len(names)*lookup[name]) for name in video],dtype=np.float64)
verified=[]
for k in range(3):
    take=v['fold']==k;vid=v['video'][take];r=v['y'][take]-v['pf'][take];t=v['mu'][take]-v['pf'][take]
    assert len(np.unique(vid))==6 and take.sum()==plan['cal_counts'][k]
    w=weights(vid);C=float(w@(r*t));B=float(w@(t*t));lam=0. if B==0 else float(np.clip(C/B,0,1));item=res['folds'][k]
    for actual,reported in [(C,item['C_video_equal']),(B,item['B_video_equal']),(lam,item['lambda_video_equal'])]:assert abs(actual-reported)<1e-12
    cap=float(np.quantile(abs(v['old_prediction'][take]-v['pf'][take]),.95,method='linear'));assert abs(cap-item['output_cap_old_native_abs_shift_quantile95'])<1e-12
    for test in item['leave_one_video_out']:
        keep=vid!=test['removed_video'];wk=weights(vid[keep]);c=float(wk@(r[keep]*t[keep]));b=float(wk@(t[keep]*t[keep]));l=0. if b==0 else float(np.clip(c/b,0,1))
        assert abs(c-test['C'])<1e-12 and abs(b-test['B'])<1e-12 and abs(l-test['lambda_video_equal'])<1e-12
    q=float(w@((lam*t-r)**2-r**2));assert abs(q-item['cal_fitted_surrogate_output_relative_risk'])<1e-12 and q<=1e-12
    verified.append(dict(fold=k,C_video_equal=C,B_video_equal=B,lambda_video_equal=lam,output_cap=cap,loo_lambda_range=item['loo_lambda_range'],loo_C_sign_flip=item['loo_C_sign_flip']))
proof=dict(status='CAL_ONLY_VIDEO_EQUAL_INDEPENDENT_CPU_AUDIT_PASSED',actual_gpu_forward=False,rows=418,videos=18,eval_label_values_or_metrics_received=False,source_sha256=sha(a.root/'calibrate_video_equal_residual_v1.py'),plan_sha256=sha(a.root/'plan.json'),original_execute_receipt_sha256=sha(a.root/'execute/receipt.json'),original_natural_exit_sha256=sha(a.root/'execute_exit.json'),calibration_array_sha256=sha(a.root/'execute/calibration_inputs.npz'),result_sha256=sha(a.root/'execute/calibration_result.json'),folds=verified,scope='CAL fitting and sensitivity; no EVAL performance or actual message control. Global positive target shrinkage only.')
a.out.write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))
