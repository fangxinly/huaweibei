"""Append-only backup using filesystem paths, not shell-transcoded filenames."""
from pathlib import Path
import datetime, hashlib, json

root = Path(__file__).resolve().parents[1]
out = root / 'outputs'
work = root / 'work'
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
dest = Path('D:/CodexBackups/selective_flow_20261003_1105') / ('jacobian_reports_' + stamp)
files = sorted(out.glob('*Jacobian*'))
files += [out / n for n in [
    '有限任务风险反馈下一步设计.md', '研究接续状态.md',
    '连续向量三臂七文件轮转CPU保存核验.json']]
files += [work / n for n in [
    'train_soft_vector_v2.py', 'soft_vector_runtime_v2.py',
    'soft_vector_jacobian_candidate_v1.py', 'collect_label_free_jacobian_v1.py',
    'audit_label_free_jacobian_v1.py', 'assemble_soft_full_checkpoint_v2.py',
    'diagnose_soft_vector_v2.py', 'analyze_soft_vector_v2.py',
    'capture_soft_vector_v3.py', 'audit_soft_snapshot_v3.py',
    'verify_soft_preservation_cpu_v1.py', 'formal_plan_v2.json',
    'check_soft_full_inference_v2.py']]
files += sorted((work / 'jacobian_checks').glob('*preflight.json'))
files += [work / 'new_p4_checks/session_closures_20261005T135054Z.json']
assert files and all(p.is_file() for p in files)
assert len({p.name for p in files}) == len(files)
dest.mkdir(exist_ok=False)
rows = []
for p in files:
    target = dest / p.name
    content = p.read_bytes()
    target.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
    rows.append(dict(source=str(p), destination=str(target), bytes=len(content), sha256=digest))
record = dict(status='ADDITIVE_PERMANENT_RECORDS_ALL_SHA_VERIFIED',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows=rows,
    limits='Append-only records. Previous partial backup retained. Mechanism evidence is the actual frozen plan and three original preflight reports, not an invented missing filename.')
receipt = out / ('Jacobian完整研究与接续资料永久保存核验_' + stamp + '.json')
encoded = json.dumps(record, ensure_ascii=False, indent=2)
for p in [dest / 'late_manifest.json', receipt]:
    with p.open('x', encoding='utf-8') as f:
        f.write(encoded)
print(json.dumps(dict(status=record['status'], files=len(rows), receipt=str(receipt), directory=str(dest)), ensure_ascii=False))
