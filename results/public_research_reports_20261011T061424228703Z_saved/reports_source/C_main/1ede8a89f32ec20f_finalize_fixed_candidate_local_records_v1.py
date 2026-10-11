import hashlib, json, shutil, zipfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
DIR = Path('D:/CodexBackups/selective_flow_20261003_1105/minimal_fixed_v2_local_candidate_20261006T122148Z')
now = datetime.now(timezone.utc).isoformat()
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
space = {drive: shutil.disk_usage(drive).free for drive in ('C:/', 'D:/')}
assert space['D:/'] > 100 * 1024 * 1024
proof = json.loads((DIR / 'preservation_receipt.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(DIR / 'source_candidate_package.zip') as z:
    names = z.namelist()
    assert len(names) == len(set(names)) and z.testzip() is None
    assert set(names) == set(proof['members'])
    for name, meta in proof['members'].items():
        assert sha(DIR / name) == meta['sha256']
        assert hashlib.sha256(z.read(name)).hexdigest() == meta['sha256']
assert sha(DIR / 'source_candidate_package.zip') == proof['package_sha256']
ledger_path = ROOT / 'outputs/研究建议交流接续.json'
ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
ledger['local_fixed_flow_candidate']['official_automation_update_receipt'] = str(DIR / 'actual_local_preservation_and_automation_tool_receipts.json')
ledger['local_fixed_flow_candidate']['official_automation_update_status'] = 'ACTIVE'
ledger['updated_utc'] = now
ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
out = DIR / 'final_local_records'
out.mkdir(exist_ok=False)
paths = [ROOT / 'outputs/研究接续状态.md', ledger_path,
         ROOT / 'outputs/完整流fixed_v2本地候选源码接续.md',
         DIR / 'actual_local_preservation_and_automation_tool_receipts.json', Path(__file__)]
members = {}
for p in paths:
    q = out / p.name
    shutil.copyfile(p, q)
    assert sha(p) == sha(q)
    members[q.name] = {'bytes': q.stat().st_size, 'sha256': sha(q)}
with zipfile.ZipFile(DIR / 'final_local_records.zip', 'x', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.iterdir()):
        z.write(p, p.name)
with zipfile.ZipFile(DIR / 'final_local_records.zip') as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(set(z.namelist()))
    assert set(z.namelist()) == set(members)
    for name, meta in members.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == meta['sha256']
receipt = {'status': 'LOCAL_PREPARATION_RECORDS_D_SOURCE_PACKAGE_AND_COPY_SHA_ZIP_CRC_PASSED',
           'actual_utc': now, 'free_bytes_before': space, 'members': members,
           'records_zip_sha256': sha(DIR / 'final_local_records.zip'),
           'source_package_sha256': proof['package_sha256'],
           'original_remote_capture_times_unchanged': True,
           'new_Torch_or_GPU_verified': False, 'new_scores': False,
           'new_remote_capture_or_other_node_CPU': False,
           'whole_research_or_lease_complete': False}
(DIR / 'final_local_records_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': receipt['status'], 'actual_utc': now, 'new_GPU': False, 'new_scores': False}))
