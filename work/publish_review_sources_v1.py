"""Prepare publication copies only; frozen originals and evidence are read-only."""
import ast
import collections
import hashlib
import json
from pathlib import Path
from prepare_github_source_upload import sanitize, ASSIGN, TOKENS, PRIVATE, RANDOM_QUOTED

base = Path(__file__).resolve().parents[1]
dest = base/'github_upload_20261008T133346Z'
out = base/'outputs/github_review_update_20261008T161418Z'
out.mkdir(exist_ok=False)
manifest = json.loads((dest/'source_manifest.json').read_text(encoding='utf-8'))
(out/'previous_source_manifest.json').write_bytes((dest/'source_manifest.json').read_bytes())
records = {r['path']: r for r in manifest['records']}
files = [(p, 'work/official_flow_health_candidate_v1/'+p.name)
         for p in sorted((base/'work/official_flow_health_candidate_v1').iterdir())
         if p.suffix in {'.py', '.npy'}]
for name in ('training_health_v1.py', 'native_training_health_v1.py',
             'prepare_official_flow_health_candidate.py', 'external_pretraining_guard_v1.py',
             'check_research_contracts_v1.py', 'github_access_check_guarded_v2.py',
             'negative_gain_review_v2.py', 'message_output_backtrack_v2.py',
             'prepare_review_reporting_fixes_v2.py', 'publish_review_sources_v1.py'):
    files.append((base/'work'/name, 'work/'+name))
files.append((base/'work/github_access_check_guarded_v2.py', 'work/github_access_check.py'))
files.append((base/'outputs/code_review_response_20261008T154000Z/REPORT.md', 'docs/code_review_and_pretraining_direction.md'))
for source, relative in files:
    original = source.read_bytes()
    clean, changes = (original, {}) if source.suffix == '.npy' else sanitize(original)
    if source.suffix == '.py': ast.parse(clean.decode('utf-8-sig'))
    target = dest/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(clean)
    records[relative] = dict(path=relative, origin=source.relative_to(base).as_posix(),
        kind='generated_order' if source.suffix == '.npy' else ('research_note' if source.suffix == '.md' else 'source'),
        bytes=len(clean), original_sha256=hashlib.sha256(original).hexdigest(),
        published_sha256=hashlib.sha256(clean).hexdigest(), sanitizations=changes)
manifest['records'] = [records[k] for k in sorted(records)]
manifest['latest_source_update'] = 'Original gradient audit and synthetic optimizer startup completed; MSE health candidate source prepared; external pretraining design only, no real new TRAIN/VAL/TEST.'
(dest/'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
readme = dest/'README.md'
old = readme.read_text(encoding='utf-8-sig')
assert '## 代码审阅与外部条件预训练' not in old
readme.write_text(old.rstrip()+'''

## 代码审阅与外部条件预训练

[证据、修正与实验阶梯](docs/code_review_and_pretraining_direction.md)。原真实TRAIN首步梯度已核归档；原优化器/warmup下CPU合成检查显示流各通路随后可启动，因此首步零梯度不能证明100轮一直惰性。新 [MSE-only候选](work/official_flow_health_candidate_v1/)只改主任务损失并加入TRAIN梯度/实际抽样Adam更新日志；**完整编码器资格、真实训练及新VAL/TEST仍未执行**。

外部预训练目前是设计草案，尚未获得实际MOSI/MOSEI重叠与资产核验。条件方差不能称任务冗余，潜变量不能自动称任务互补。守卫只能检查声明的元数据，不能代替真实来源或近重复审计。

`work/github_access_check.py` 已改为安全默认：import或无参数执行不访问Git凭据/网络。只有显式 `--check-repo-access` 才进行可选访问核验。历史副本保留为来源，不建议直接运行历史批次工具；旧租期计划不能用于新机器。报告v2移除了负gain的预定结论及backtrack无标签的虚假布尔证书，历史结果未改写或重跑。

合成数学/泄漏/凭据默认行为检查：`python work/check_research_contracts_v1.py`。不读取实际数据、权重或凭据。
''', encoding='utf-8')
counts = collections.Counter(str(Path(r['path']).parent).replace('\\', '/') for r in records.values())
index = '# 源码目录索引\n\n逐文件来源和SHA见 `source_manifest.json`。\n\n|目录|文件数|\n|---|---:|\n'
for directory, count in sorted(counts.items()): index += f'|[{directory}]({directory}/)|{count}|\n'
(dest/'CODE_INDEX.md').write_text(index, encoding='utf-8')
expected = set(records) | {'README.md', 'source_manifest.json', 'CODE_INDEX.md', '.gitignore'}
findings = []
for p in dest.rglob('*'):
    if not p.is_file() or '.git' in p.relative_to(dest).parts: continue
    relative = p.relative_to(dest).as_posix()
    assert relative in expected, relative
    if p.suffix == '.npy': continue
    s = p.read_text(encoding='utf-8-sig')
    if p.suffix == '.py': ast.parse(s)
    for kind, rx in [('provider_token', TOKENS), ('private_key', PRIVATE), ('credential_assignment', ASSIGN), ('random_quoted', RANDOM_QUOTED)]:
        for hit in rx.finditer(s):
            if kind == 'credential_assignment':
                value = hit.group(3)
                if value.startswith(('REDACTED', 'YOUR_', '<', '${')) or value in {'password', 'passwd', 'PASSWORD', 'example', 'placeholder'}: continue
            if kind == 'random_quoted':
                value = hit.group(2)
                if not (any(c.isupper() for c in value) and any(c.islower() for c in value) and any(c.isdigit() for c in value)): continue
            findings.append(dict(path=relative, kind=kind, line=s.count('\n', 0, hit.start())+1))
for r in records.values(): assert hashlib.sha256((dest/r['path']).read_bytes()).hexdigest() == r['published_sha256']
assert not findings, findings
report = dict(status='SOURCE_PUBLICATION_READY', source_paths=[r for _, r in files],
              total_files=len(expected), all_Python_AST_pass=True, all_manifest_SHA_pass=True,
              secret_findings=0, real_data_labels_weights_or_raw_receipts_published=False,
              new_real_training_or_scores=False)
(out/'precommit_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(status=report['status'], files_updated=len(files), total_files=len(expected), secret_findings=0)))
