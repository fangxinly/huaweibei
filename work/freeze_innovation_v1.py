"""Freeze one new source bundle and actual local saving capacity, no execution."""
import argparse,ast,datetime,shutil,zipfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path('work/innovation_v1').resolve()))
from contract import read,write,sha,synthetic_contract

ap=argparse.ArgumentParser();ap.add_argument('--clock',required=True);a=ap.parse_args();stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
root=Path('work')/('innovation_v1_frozen_'+stamp);root.mkdir();bundle=root/'bundle';bundle.mkdir()
for file in Path('work/innovation_v1').glob('*.py'):
    ast.parse(file.read_text(encoding='utf8'));shutil.copy2(file,bundle/file.name)
old=Path('work/weak_retention_train40_20261007T145201Z/bundle');oldplan=read(old/'training_execution_protocol.json');third=read('outputs/第三租期恢复与弱情绪损失原生检查实际接续.json')
for name in ('split.json','sentiment_metrics_careflow_v1.py'):
    expected=oldplan['split_sha256'] if name=='split.json' else oldplan['source_sha256'][name];assert sha(old/name)==expected;shutil.copy2(old/name,bundle/name)
local=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),D_free_bytes=shutil.disk_usage('D:/').free,C_free_bytes=shutil.disk_usage('C:/').free,
    single_trial_new_originals_upper_bytes=300000000,additional_local_margin_bytes=1000000000,no_additional_deletion=True)
assert local['D_free_bytes']>local['single_trial_new_originals_upper_bytes']+local['additional_local_margin_bytes'];write(root/'actual_local_capacity.json',local)
tests=synthetic_contract();write(root/'actual_numpy_contract_check.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tests=tests,AST_all_sources_passed=True,no_Torch_runtime_check_yet=True))
plan=dict(status='INNOVATION_V1_SOURCE_NATIVE_CACHE_PROTOCOL_FROZEN',actualclock_freeze_UTC=a.clock,scope='single_fold0_fixed_mechanism_trial',real_train_enabled=False,allowed_stages=['native','cache'],
    config=dict(latent_dim=32,inner_crossfit_k=5,outer_crossfit_k=5,baseline_steps=160,flow_steps=160,proposal_steps=160,gate_steps=200),CAL_fraction=.2,task_seed=128,
    optimizer=dict(name='AdamW',lr=.001,weight_decay=.01,batch=64,clip_norm=1),all_fixed_heads_use_final_step_no_INNER_selection=True,
    source_sha256={p.name:sha(p) for p in bundle.iterdir()},split_SHA=sha(bundle/'split.json'),asset_sha256=oldplan['asset_sha256'],
    runtime_versions={n:third['public_restoration']['A']['runtime']['versions'][n] for n in ('numpy','torch','transformers')},
    GPU_UUID={n:third['public_restoration'][n]['physical_preflight']['uuid'] for n in ('A','B','C')},
    public_assets_root='/data/coding/multimodal_flow_public_20261007T141430Z',allowed_endpoints=['REDACTED_SERVER_HOST.invalid:53423','REDACTED_SERVER_HOST.invalid:53428','REDACTED_SERVER_HOST.invalid:53495'],
    conservative_lease_end_UTC=third['conservative_lease_end_UTC'],platform_expiry_verified=False,human_lease_reference=dict(path='outputs/第三租期恢复与弱情绪损失原生检查实际接续.json',SHA=sha('outputs/第三租期恢复与弱情绪损失原生检查实际接续.json'),human_confirmed_hours=24),
    local_storage_reservation=local,remote_space_floor_bytes=2000000000,stage_budget_seconds=dict(native=300,cache=900,train=1800),
    native_D_and_other_CPU_reference=None,cache_D_reference=None,cache_reference=None,
    methods=['baseline_calibrated','full_message','flow_ordinary_gate','flow_residual_gate','regression_residual_gate'],
    gate_calibration='CAL videos excluded from every trainable predecessor, normalization and gate fitting; one nonnegative slope per residual gate and ordinary gate. Clip final gate to [0,1]. No no-harm guarantee.',
    baseline_calibration='Whole-video internal OOF within each declared task-training set. Complete-pipeline OOF for outer gate features excludes held videos from every predecessor.',
    frozen_features='Public DeBERTa masked aligned subtoken mean; aligned raw A/V means. No task checkpoint. Train-only raw scaling plus deterministic fixed orthogonal projection, common orientation across OOF models.',
    flow='Conditional noise-to-donor CFM MSE; eight Euler steps/eight fixed antithetic noise draws estimate conditional mean. Finite-sample estimate, not exact expectation/PID.',
    correction='For each direction h(receiver,message)-h(receiver,zero); all message-off corrections exactly zero. Sum divided by six is one total delta; gates use joint utility, no additive per-direction utility claim.',
    evaluation='Fold0 INNER inputs only until all predictions and complete small state D+other-node CPU saved; then all five arms scored together once. No original TEST score/selection. No OUTER scoring or full fivefold claim.')
write(bundle/'protocol.json',plan);zp=root/'frozen_source_bundle.zip'
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
    for file in bundle.iterdir():z.write(file,'bundle/'+file.name)
with zipfile.ZipFile(zp) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/('innovation_v1_source_preparation_'+stamp);D.mkdir();shutil.copy2(zp,D/zp.name)
for name in ('actual_local_capacity.json','actual_numpy_contract_check.json'):shutil.copy2(root/name,D/name)
write(D/'actual_source_seal.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_ZIP_SHA=sha(zp),copied_ZIP_SHA=sha(D/zp.name),protocol_SHA=sha(bundle/'protocol.json'),CRC=True,unique=True,complete_source_copied=True,remote_execution=False))
pointer=dict(status='SOURCE_FROZEN_NUMPY_CONTRACT_PASSED_NATIVE_PENDING',root=str(root.resolve()),bundle=str(bundle.resolve()),D=str(D),archive_SHA=sha(zp),protocol_SHA=sha(bundle/'protocol.json'))
write('work/innovation_v1_current.json',pointer);print(pointer)
