"""Freeze native-only candidate source, with original bytes retained."""
import argparse,ast,hashlib,json,shutil,zipfile
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--actualclock',required=True);a=p.parse_args()
ws=Path(__file__).resolve().parent.parent;pointer=json.loads((ws/'work/official_upgrade_pointer.json').read_text(encoding='utf8'))
old=Path(pointer['source']);oldplan=json.loads(Path(pointer['active_training_plan']).read_text(encoding='utf8'))
stamp=a.actualclock.replace('-','').replace(':','').replace(' UTC','Z').replace(' ','T')
root=ws/'work'/('residual_direction_v2_frozen_'+stamp);root.mkdir();bundle=root/'bundle';bundle.mkdir()
for name in ('anchored_flow.py','incremental_message.py','official_upgrade.py','legacy_flow_model.py','finite_single_token_reader_v1.py','common.py'):
    assert sha(old/name)==oldplan['source_sha256'][name];shutil.copy2(old/name,bundle/name)
wrapper=(old/'run_stage.py').read_text(encoding='utf8');assert "native='official_upgrade.py'" in wrapper
(bundle/'run_stage.py').write_text(wrapper.replace("native='official_upgrade.py'","native='native_contract.py'"),encoding='utf8')
for name in ('controlled_flow.py','oof_selector.py','native_contract.py'):shutil.copy2(ws/'work/residual_direction_v2_candidate'/name,bundle/name)
for file in bundle.glob('*.py'):ast.parse(file.read_text(encoding='utf8'))
plan=dict(status='RESIDUAL_DIRECTION_V2_NATIVE_ONLY_FROZEN_NO_REAL_TRAIN_ENABLED',actualclock_UTC=a.actualclock,
    allowed_stages=['native'],source_sha256={f.name:sha(f) for f in bundle.glob('*.py')},
    asset_sha256=oldplan['asset_sha256'],runtime_versions=oldplan['runtime_versions'],GPU_UUID=oldplan['GPU_UUID'],
    conservative_lease_end_UTC='2026-10-08T14:00:00+00:00',stage_budget_seconds={'native':120},
    remote_free_floor_bytes=100_000_000,complete_archive_bytes_ceiling=5_000_000,archive_part_bytes=1_000_000_000,
    trusted_asset_reference=str(old),native_only_no_task_labels_weights_forward_or_performance=True,
    official_training_only_for_future_controller=True,
    fixed_features='12 fixed random slot projections + same-flow p0 + 7 actual mask deltas = 20',
    ridge_normalized_population_lambda=1.0,estimated_utility='2*s*rho_hat*actual_mask_delta-actual_mask_delta^2',
    masks=['OFF']+[list(pair) for pair in ((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))]+['ALL'],
    decoder_paths='Each mask uses the same original two-step flow and shared decoder; no summing delta contributions.',
    calibration='Separate predeclared TRAIN videos excluded from every fitted predecessor and controller; evaluate actual deployment stack there.',
    complete_crossfit_training_enabled=False,CaReFlow_or_new_VAL_TEST_execution_enabled=False,
    limitation='Source/native contracts only. The prior real OOF pilot failed generalization. No new performance improvement or risk guarantee claimed.')
planpath=bundle/'native_protocol.json';planpath.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
zipath=root/'frozen_source.zip'
with zipfile.ZipFile(zipath,'x',zipfile.ZIP_DEFLATED) as z:
    for f in bundle.iterdir():z.write(f,f.name)
with zipfile.ZipFile(zipath) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
d=Path(pointer['D'])/('residual_direction_v2_native_'+stamp);d.mkdir();shutil.copy2(zipath,d/zipath.name);shutil.copy2(planpath,d/planpath.name)
value=dict(actualclock_UTC=a.actualclock,root=str(root),bundle=str(bundle),D=str(d),ZIP=str(zipath),ZIP_SHA=sha(zipath),plan_SHA=sha(planpath),
    source_sha256=plan['source_sha256'],local_AST_pass=True,real_OOF_training_complete=False,real_native_CPU_complete=False)
(root/'source_freeze_receipt.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copy2(root/'source_freeze_receipt.json',d/'source_freeze_receipt.json')
(ws/'work/residual_direction_v2_current.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(value,ensure_ascii=False))
