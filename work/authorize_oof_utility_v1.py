from pathlib import Path
import datetime,json,hashlib
p=Path('D:/CodexBackups/selective_flow_20261003_1105/oof_utility_preparation_20261006T033054Z')
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
r=json.loads((p/'precheck_receipt.json').read_text());x=json.loads((p/'precheck_exit.json').read_text())
plan=json.loads((p/'plan.json').read_text())
assert r['passed'] and x['exit_code']==0 and r['source_sha256']==plan['source_sha256'] and r['plan_sha256']==sha(p/'plan.json')
assert r['model_state_before']==r['model_state_after'] and r['no_parameter_grads'] and r['optimizer_steps']==0
assert r['actual_peak_allocated_bytes']<plan['maximum_peak_allocated_bytes']
assert r['pessimistic_41_batch_seconds']<plan['maximum_seconds']
assert all(v['label_replacement_max_error']==0 and v['original_F_replay_max_error']<=1e-6 and v['original_learned_replay_max_error']<=1e-6 and v['oof_mu_input_gradient_hvp']['input_hvp_relative_error']<1e-5 for v in r['evidence'])
assert 620 in r['evidence'][1]['rows'] and r['evidence'][1]['valid_lengths'][0]==1
old=Path('outputs/有限任务风险C2完整权重本地独立核验.json')
prior=json.loads(old.read_text(encoding='utf-8'))
assert prior['receipt']['best_epoch']==37 and prior['receipt']['addon_sha256']==plan['pinned_files'][plan['formal_root']+'/run/best_addon.pt']
assert prior['receipt']['full_raw_vs_cached_prediction_max_error']<=1e-6
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execute_authorized_after_actual_precheck=True,
    precheck_receipt_sha256=sha(p/'precheck_receipt.json'),precheck_actual_exit=x,
    plan_sha256=sha(p/'plan.json'),source_sha256=plan['source_sha256'],
    prior_original_input_C2_strict_replay_proof_sha256=sha(old),prior_original_input_replay_max_error=prior['receipt']['full_raw_vs_cached_prediction_max_error'],
    limits='New GPU precheck checks coordinates and immutable controls; prior original-input disk replay is retained proof, not rerun this turn.')
(p/'execution_authorization.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
Path('outputs/OOF教师效用真实GPU预检独立门控.json').write_text(json.dumps(dict(authorization=out,original_receipt=r),indent=2),encoding='utf-8')
print(json.dumps(out))
