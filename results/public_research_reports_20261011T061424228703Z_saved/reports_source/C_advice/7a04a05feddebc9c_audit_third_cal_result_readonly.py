"""Inspect preserved CAL result and frozen plan only; no NPZ outcome access or new fit."""
from pathlib import Path
import hashlib
import json
import math

root=Path(__file__).resolve().parents[1]
cal=Path('D:/CodexBackups/selective_flow_20261003_1105/cal_video_equal_signal_actual_20261006T063535Z')
pool=Path('D:/CodexBackups/selective_flow_20261003_1105/matched_message_pools_actual_20261006T064950Z')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
result_path=cal/'execute/calibration_result.json'
assert sha(result_path)=='9030a3d4745149ae172bfe85b7312fd4e485557d865db914ec84aadf642a8042'
result=json.loads(result_path.read_text(encoding='utf-8'))
params=json.loads((pool/'cal_parameters.json').read_text(encoding='utf-8'))
plan=json.loads((pool/'plan.json').read_text(encoding='utf-8'))
assert sha(pool/'diagnose_matched_message_pools_v1.py')==plan['source_sha256']
assert sha(pool/'cal_parameters.json')=='81784aa7e7a0cd94eccab8270d4831a03c3ede19b1f1ff8696c2d2e78b2d3e91'
assert result['eval_values_decoded']==0 and not result['eval_metrics_computed']
assert params['original_result_sha256']==sha(result_path)
out=[]
for fold in result['folds']:
    k=fold['fold']; c=fold['C_video_equal']; b=fold['B_video_equal']; lam=fold['lambda_video_equal']
    assert lam==params['lambda_by_fold'][k]
    assert fold['output_cap_old_native_abs_shift_quantile95']==params['cap_by_fold'][k]
    lo=fold['leave_one_video_out']
    out.append(dict(fold=k,C=c,B=b,lambda_frozen=lam,
        positive_C_videos=sum(v['C']>0 for v in fold['per_video']),
        positive_leave_one_out=sum(v['C']>0 for v in lo),
        lambda_leave_one_out_range=fold['loo_lambda_range'],
        output_interpolation_CAL_risk_change=fold['cal_fitted_surrogate_output_relative_risk'],
        video_equal_target_shift_rms=lam*math.sqrt(b),
        cap_frozen=fold['output_cap_old_native_abs_shift_quantile95']))
audit=dict(scope='Original CAL JSON and frozen proposal consistency, no raw CAL/EVAL labels or new fit',
           cal_result_sha256=sha(result_path),pool_plan_sha256=sha(pool/'plan.json'),
           pool_source_sha256=sha(pool/'diagnose_matched_message_pools_v1.py'),
           raw_outcomes_read=False,eval_labels_read=False,new_fitting=False,new_gpu=False,
           folds=out)
(root/'work/third_cal_result_readonly_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
