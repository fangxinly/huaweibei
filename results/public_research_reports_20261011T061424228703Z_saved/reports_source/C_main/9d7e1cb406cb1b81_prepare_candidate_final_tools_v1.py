"""New posttrain source preparation; no real labels or training execution."""
import argparse,ast,hashlib,json,shutil,zipfile
from pathlib import Path

base=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args()
root=base/'work'/('candidate_posttrain_preparation_'+a.tag);root.mkdir()
old=base/'work/A_candidate_recovery400_qualified_20261009T002805Z'
tools=base/'work/official_mse_posttrain_tools_v1'
new=base/'work/candidate_final_tools_v1'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def replace(s,old,new):
    assert old in s,old
    return s.replace(old,new)
for f in old.iterdir():
    if f.suffix in ('.py','.npy'):shutil.copy2(f,root/f.name)
for f in tools.glob('*.py'):
    if f.name!='freeze_posttrain_stage_v1.py':shutil.copy2(f,root/f.name)
for f in new.glob('*.py'):shutil.copy2(f,root/f.name)
s=(base/'work/polarity_intensity_official_pipeline_preparation/infer_official.py').read_text(encoding='utf8')
s=replace(s,"assert p['training_D_B_CPU_gate'] and p['candidate_training_D_B_CPU_gate']","assert p['training_Release_B_CPU_gate']")
s=replace(s,"    np.savez(a.out/'fixed_official_VAL_TEST_prediction.npz',**result)","    frozen=a.input_root/'out/selected_DEV_frozen_prediction.npz';assert sha(frozen)==p['training_original_reference']['member_SHA']['out/selected_DEV_frozen_prediction.npz']\n    with np.load(frozen,allow_pickle=False) as z:selection_replay_error=float(np.max(np.abs(result['VAL_prediction']-z['prediction'])))\n    assert selection_replay_error<=1e-4\n    peak=dict(allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved());assert max(peak.values())<=p['GPU_peak_ceiling_bytes']\n    np.savez(a.out/'fixed_official_VAL_TEST_prediction.npz',**result)")
s=replace(s,"checks=checks,TRAIN_VAL_TEST_no_training_overlap", "checks=checks,candidate_mode=p['candidate_mode'],VAL_inference16_vs_selection128_maxerror=selection_replay_error,peak=peak,TRAIN_VAL_TEST_no_training_overlap")
(root/'infer_official.py').write_text(s,encoding='utf8')
s=(root/'score_official.py').read_text(encoding='utf8')
s=replace(s,"roles=roles,prediction_SHA", "roles=roles,candidate_mode=p['candidate_mode'],recovery_compute_disclosure=p['recovery_compute_disclosure'],prediction_SHA")
(root/'score_official.py').write_text(s,encoding='utf8')
s=(root/'audit_public_small_stage_v1.py').read_text(encoding='utf8')
s=replace(s,"assert uuid=='GPU-609b23d6-282d-8a2b-23f5-9433b824d212'", "assert uuid==a.uuid")
s=replace(s,"p.add_argument('--prediction-original',type=Path)", "p.add_argument('--prediction-original',type=Path);p.add_argument('--uuid',required=True)")
(root/'audit_public_small_stage_v1.py').write_text(s,encoding='utf8')
s=(root/'run_posttrain_capture_v1.py').read_text(encoding='utf8')
s=replace(s,"'ACTUAL_OFFICIAL_MSE_'", "'ACTUAL_OFFICIAL_CANDIDATE_'")
s=replace(s,"assert (a.stage == 'audit' and a.node == 'B') or (a.stage in ('infer', 'score') and a.node == 'A')", "assert a.node==p['execution_node']; assert (a.stage=='audit' and a.node!=p['training_node']) or (a.stage in ('infer','score') and a.node==p['training_node'])")
s=replace(s,"B_posttrain_native_result.json", "candidate_posttrain_native_result.json")
s=replace(s,"B_posttrain_native_exit.json", "candidate_posttrain_native_exit.json")
s=replace(s,"qualify_posttrain_contract_v1.py", "qualify_candidate_posttrain_v1.py")
s=replace(s,"choices=['A', 'B']", "choices=['A', 'B', 'C']")
s=replace(s,"if a.received_archive:\n            cmd += ['--received-archive', str(a.received_archive)]", "assert a.received_archive is None, 'Only complete public source transport is qualified for this new audit'")
s=replace(s,"'B_original_archive': str(a.root / 'out/A_complete_original.zip')", "'peer_original_archive': str(a.root / 'out/complete_training_original.zip')")
(root/'run_posttrain_capture_v1.py').write_text(s,encoding='utf8')
sources={f.name:sha(f) for f in root.iterdir() if f.suffix in ('.py','.npy')}
for f in root.glob('*.py'):ast.parse(f.read_text(encoding='utf8'))
plan=json.loads((old/'qualified_resume_plan.json').read_text(encoding='utf8'))
plan.update(status='CANDIDATE_POSTTRAIN_CPU_SYNTHETIC_QUALIFICATION_ONLY',actualclock_before_qualification_UTC=a.stamp,asset_sha256={},source_sha256=sources,native_only=True,real_final100_not_completed=True,actual_stage_protocol_not_yet_frozen=True)
(root/'native_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
with zipfile.ZipFile(root/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
    for f in root.iterdir():
        if f.suffix!='.zip':z.write(f,f.name)
print(json.dumps(dict(root=str(root),plan_SHA=sha(root/'native_plan.json'),source_zip_SHA=sha(root/'source.zip'),source_files=len(sources),AST_only_local=True,real_VAL_TEST_not_executed=True)))
