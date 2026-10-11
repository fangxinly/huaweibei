"""Local D preservation of new research; no remote snapshot or model download."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import zipfile
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[1]
research = root / 'work/existing_direction_simplex_oracle_20261006T051944Z'
assert json.loads((research / 'independent_cpu_audit.json').read_text(encoding='utf-8'))['status'] == 'LOCAL_INDEPENDENT_NUMPY_CONVEX_CERTIFICATE_AUDIT_PASSED'
plan = json.loads((research / 'three_stage_candidate_design_v1.json').read_text(encoding='utf-8'))
assert plan['actual_GPU_or_CAL_fit_or_new_100'] is False
original_advice = Path('C:/Users/21234/.codex/attachments/0f4287ca-03cb-4619-8f2f-4e2d74eba28b/已粘贴的文本.txt')
# Independent algebra and an example showing precision alone is insufficient.
p = np.array([-.7, 0., .9, 1.8])
mu = np.array([-.3, .1, 1.2, -.2])
delta = np.array([.2, -.1, .7, -.8])
residual_form = -2 * (mu - p) * delta + delta**2
square_form = (p + delta - mu)**2 - (p - mu)**2
identity_error = float(np.max(np.abs(residual_form - square_form)))
assert identity_error < 1e-14
toy_risk = np.array([-.001]*8 + [.01]*2)
assert np.mean(toy_risk < 0) == .8 and float(np.mean(toy_risk)) > 0
math_path = research / 'residual_identity_and_precision_counterexample.json'
assert not math_path.exists()
math_path.write_text(json.dumps({'status': 'LOCAL_SYNTHETIC_MATH_CHECK_PASSED',
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'residual_square_objective_identity_max_error': identity_error,
    'toy_beneficial_precision': float(np.mean(toy_risk < 0)),
    'toy_mean_risk_increase': float(np.mean(toy_risk)),
    'actual_CAL_fit_or_EVAL_metrics_or_GPU': False}, indent=2), encoding='utf-8')
backup_root = Path('D:/CodexBackups/selective_flow_20261003_1105')
fresh_free = {drive: shutil.disk_usage(drive).free for drive in ('C:/', 'D:/')}
assert fresh_free['D:/'] >= 1024**3
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
destination = backup_root / ('three_stage_direction_research_' + stamp)
destination.mkdir(exist_ok=False)
files = [(root / 'work/analyze_existing_direction_simplex_oracle_v1.py', 'source/analyze_existing_direction_simplex_oracle_v1.py'),
         (root / 'work/audit_existing_direction_simplex_oracle_v1.py', 'source/audit_existing_direction_simplex_oracle_v1.py'),
         (Path(__file__), 'source/preserve_three_stage_research_v1.py'),
         (original_advice, 'user_supplied_advice.txt'),
         (root / 'outputs/三级消息控制建议评估与短实验修订设计.md', 'research_report.md'),
         (root / 'outputs/研究接续状态.md', 'continuation_before_new_result.md')]
files += [(path, 'original_research/' + path.name) for path in sorted(research.iterdir()) if path.is_file()]
records = {}
for source, relative in files:
    target = destination / 'files' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    source_digest = sha(source)
    shutil.copyfile(source, target)
    assert sha(target) == source_digest
    records[relative] = {'sha256': source_digest, 'bytes': target.stat().st_size}
archive = destination / 'research.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for relative in records:
        z.write(destination / 'files' / relative, relative)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(set(z.namelist())) == len(records)
    assert set(z.namelist()) == set(records)
    for relative, record in records.items():
        assert hashlib.sha256(z.read(relative)).hexdigest() == record['sha256']
proof = {'status': 'LOCAL_NEW_RESEARCH_PERMANENT_D_SHA_ZIP_CRC_MEMBERS_PASSED',
         'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'backup_directory': str(destination), 'fresh_free_bytes': fresh_free,
         'files': records, 'zip_sha256': sha(archive), 'zip_crc_and_unique_members_passed': True,
         'user_advice_sha256': sha(original_advice), 'remote_capture_or_independent_node_CPU': False,
         'large_weight_download': False, 'actual_GPU_or_calibration': False,
         'posthoc_label_known_simplex_only': True}
prefix = ('最新本地研究UTC' + proof['utc'] + '：新建议评估见三级消息控制建议评估与短实验修订设计.md。'
          '原冻结TRAIN输出的标签已知凸组合Oracle：旧方向5.33%，旧+新第1步受限最优6.61%；'
          '独立NumPy凸证书通过，只是事后输出组合，不是消息控制/CAL拟合/泛化，约5.91%系数不得部署。'
          '新建议采纳有限幅度+独立接受，残差改写数学等价、大分歧非置信度、precision须联报净风险。'
          '新材料已永久D全SHA/ZIPCRC/member核验，见三级消息控制新研究本地永久保存.json；远端未重试。\n\n')
state = root / 'outputs/研究接续状态.md'
state.write_text(prefix + state.read_text(encoding='utf-8'), encoding='utf-8')
state_backup = destination / 'continuation_after_new_result.md'
shutil.copyfile(state, state_backup)
assert sha(state) == sha(state_backup)
proof['updated_continuation_outside_original_zip'] = {'path': str(state_backup), 'sha256': sha(state_backup)}
proof_path = root / 'outputs/三级消息控制新研究本地永久保存.json'
assert not proof_path.exists()
proof_path.write_text(json.dumps(proof, indent=2), encoding='utf-8')
shutil.copyfile(proof_path, destination / 'preservation_proof.json')
assert sha(proof_path) == sha(destination / 'preservation_proof.json')
print(json.dumps({'status': proof['status'], 'backup_directory': str(destination),
                  'files_in_zip': len(records), 'zip_sha256': proof['zip_sha256'],
                  'synthetic_identity_error': identity_error, 'toy_precision': .8,
                  'toy_mean_risk_increase': float(np.mean(toy_risk)),
                  'updated_continuation_sha256': sha(state), 'preservation_proof_sha256': sha(proof_path)}))
