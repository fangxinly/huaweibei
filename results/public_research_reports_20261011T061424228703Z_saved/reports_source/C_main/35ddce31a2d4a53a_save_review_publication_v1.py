"""Save verified public Git snapshot and update mutable continuation pointers."""
import hashlib
import json
import shutil
from pathlib import Path

base = Path(__file__).resolve().parents[1]
out = base/'outputs/github_review_update_20261008T161418Z'
verified = base/'outputs/github_polarity_update_20261008T153225Z_v2/blob_verification_4e1f8997683c.json'
report = json.loads(verified.read_text(encoding='utf-8'))
assert report['remote_matches'] and not report['index_blob_SHA_failures'] and report['checkout_clean']
dest = Path('D:/CodexBackups/selective_flow_20261003_1105/code_review_and_pretraining_20261008T161607Z')
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
files = [verified, out/'published_source.bundle', out/'precommit_verification.json']
copies = {}
for p in files:
    q = dest/('publication_'+p.name)
    assert not q.exists()
    shutil.copyfile(p, q)
    assert h(q) == h(p)
    copies[q.name] = dict(SHA256=h(q), bytes=q.stat().st_size)
receipt = dict(status='GITHUB_SOURCE_PUBLICATION_REMOTE_BLOBS_AND_D_GIT_BUNDLE_VERIFIED',
    actualclock_before_final_receipt_UTC='2026-10-08 16:17:33 UTC', **report,
    D_files=copies, no_credentials_data_weights_uploaded=True,
    new_real_training_or_VAL_TEST=False)
(out/'publication_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
shutil.copyfile(out/'publication_receipt.json', dest/'publication_receipt.json')
path = base/'outputs/代码审阅梯度与外部预训练研究实际接续.json'
state = json.loads(path.read_text(encoding='utf-8'))
state['GitHub_publication'] = receipt
path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
shutil.copyfile(path, dest/'代码审阅梯度与外部预训练研究实际接续.json')
ledger = base/'outputs/GitHub上传实际接续.json'
original = ledger.read_bytes()
(out/'prior_GitHub_continuation.json').write_bytes(original)
old = json.loads(original.decode('utf-8-sig'))
old['latest_publication'] = receipt
ledger.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding='utf-8')
shutil.copyfile(ledger, dest/'GitHub上传实际接续.json')
statuspath = base/'outputs/研究接续状态.md'
s = statuspath.read_text(encoding='utf-8-sig')
statuspath.write_text('GitHub后续实际核验 2026-10-08 16:17:33 UTC：审阅报告/原优化器启动诊断/MSE源码/外部数据metadata守卫/安全默认凭据辅助/v2报告修正已同步main '+report['commit']+'；1782文件、1778manifest Gitblob SHA全部过，remote main一致、clean、完整Gitbundle验证并D原SHA保存。非新训练/VALTEST/外部预训练，旧真实结果保持。\n\n'+s, encoding='utf-8')
print(json.dumps(dict(status=receipt['status'], commit=report['commit'], D_bundle_bytes=copies['publication_published_source.bundle']['bytes'])))
