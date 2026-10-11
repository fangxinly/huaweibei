"""Preserve new small evidence/source snapshots; never modifies old archives."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

base = Path(__file__).resolve().parents[1]
stamp = '20261008T161607Z'
clock = '2026-10-08 16:16:07 UTC'
out = base/'outputs/code_review_response_20261008T154000Z'
dest = Path('D:/CodexBackups/selective_flow_20261003_1105')/('code_review_and_pretraining_'+stamp)
dest.mkdir(exist_ok=False)
files = list(out.glob('*'))
files += list((base/'work/official_flow_health_candidate_v1').glob('*'))
for name in ('training_health_v1.py', 'native_training_health_v1.py',
             'external_pretraining_guard_v1.py', 'check_research_contracts_v1.py',
             'github_access_check_guarded_v2.py', 'negative_gain_review_v2.py',
             'message_output_backtrack_v2.py', 'prepare_review_reporting_fixes_v2.py',
             'prepare_official_flow_health_candidate.py', 'publish_review_sources_v1.py',
             'save_review_evidence_v1.py'):
    files.append(base/'work'/name)
files = [p for p in files if p.is_file()]
sources = {p.relative_to(base).as_posix(): p for p in files}
assert len(sources) == len(files)
archive = dest/'small_evidence_and_sources.zip'
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
    for name, path in sorted(sources.items()): z.writestr(name, path.read_bytes())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(set(z.namelist())) == len(sources)
    for name, path in sources.items(): assert z.read(name) == path.read_bytes()
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
receipt = dict(status='LOCAL_REVIEW_EVIDENCE_SOURCE_SNAPSHOT_D_SHA_CRC_UNIQUE_VERIFIED',
    actualclock_before_snapshot_UTC=clock, ZIP_SHA256=h(archive), ZIP_bytes=archive.stat().st_size,
    members={name: dict(bytes=p.stat().st_size, SHA256=h(p)) for name,p in sorted(sources.items())},
    remote_capture_is_only_original_native_health=False, new_remote_capture=False,
    synthetic_contracts_pass=True, actual_external_data_overlap_audit=False,
    MSE_candidate_real_TRAIN_executed=False, external_pretraining_executed=False,
    new_VAL_TEST_scores=False, immutable_historical_evidence_unchanged=True,
    full_training_storage_dependency='At least 20GB permanent path still missing',
    original_health_D_archive='D:/CodexBackups/selective_flow_20261003_1105/training_health_20261008T154602Z',
    closed_sessions=[dict(id=62342, command='exit', actual_exit_code=0), dict(id=14680, command='bye', actual_exit_code=0)])
(dest/'snapshot_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
(out/'D_snapshot_receipt.json').write_bytes((dest/'snapshot_receipt.json').read_bytes())
state = dict(status='REAL_FIRST_GRADIENT_AND_NATIVE_SYNTHETIC_STARTUP_AUDITED_MSE_SOURCE_ONLY_EXTERNAL_DESIGN_ONLY',
    actualclock_before_snapshot_UTC=clock, report=str(out/'REPORT.md'), D_directory=str(dest),
    D_snapshot_SHA256=receipt['ZIP_SHA256'],
    original_first_gradient_SHA256='4aeb3b600935346cff38530bc2e5f4fd7b39cdc493dfd8a3f89d83b1854a5a27',
    native_health_joint_SHA256=h(out/'native_health_joint.json'),
    first_gradient_only_not_long_term_inertness=True, warmup_first_actual_LR_zero=True,
    synthetic_later_nonzero_gradient_steps=dict(fields_role_message=3, reader=4),
    no_real_new_training_or_VAL_TEST_scores=True, external_pretraining_full_protocol_frozen=False,
    data_encoder_asset_budget_provenance_pending=True, permanent_backup_20GB_pending=True,
    current_official_MAE=dict(ours_VAL=.5972565370388166, ours_TEST=.650616015,
                            CaReFlow_VAL=.6045238262758688, CaReFlow_TEST=.619535294),
    original10min_automation_unchanged=True, goal_complete=False)
path = base/'outputs/代码审阅梯度与外部预训练研究实际接续.json'
path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
shutil.copyfile(path, dest/path.name)
statuspath = base/'outputs/研究接续状态.md'
old = statuspath.read_text(encoding='utf-8-sig')
prefix = ('最新实际 '+clock+'：先读《代码审阅梯度与外部预训练研究实际接续.json》及code_review_response_20261008T154000Z/REPORT.md。'
          '原D完整训练成员首步梯度全流分支0/gain1.391e-8真实核过，但仅首步不冒长期惰性。A原TorchCPU合成原AdamW+400warmup双损失各10步child495自然0：fields/role/message第3步、reader第4步非零，首步LR0；MSE不自动修慢启动，非真实性能。'
          '原capture/D/source/UUID/argv/PID/SHA/ZIPCRC唯一过，SSH62342/SFTP14680明确exit/bye自然0禁复用。新MSE单变量官方候选+持续TRAIN梯度与抽样实际Adam日志仅源码准备，无完整编码器资格/新TRAIN/VALTEST。'
          '新外部条件预训练四臂及标注曲线仅草案：条件方差非任务冗余/epsilon非互补真值，MOSI-MOSEI实际资产重叠与近重复待核，metadata守卫5类合成泄漏拒绝过。'
          '安全默认凭据辅助与v2报告元数据已修，原冻结历史不改/不重评分。小件D '+str(dest)+' 完整SHA/CRC唯一保存。'
          '当前C1.53GB/D.327GB/E.523GB，长期20GB路径仍待，无外部数据下载或预训练；现官方VAL/TEST五项不变，整体/正式超过/五折未完成，原10min不改。以下历史。\n\n')
assert not old.startswith(prefix)
statuspath.write_text(prefix+old, encoding='utf-8')
print(json.dumps(dict(status=receipt['status'], members=len(sources), ZIP_bytes=receipt['ZIP_bytes'], D=str(dest))))
