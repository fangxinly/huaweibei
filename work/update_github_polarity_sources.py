"""Publish reviewed source copies only; original experiment receipts remain local."""
import ast,collections,hashlib,json,sys
from pathlib import Path
from prepare_github_source_upload import sanitize,ASSIGN,TOKENS,PRIVATE,RANDOM_QUOTED
base=Path(__file__).resolve().parents[1]
dest=base/'github_upload_20261008T133346Z'
receiptroot=base/'outputs/github_polarity_update_20261008T153225Z_v2';receiptroot.mkdir(exist_ok=False)
manifest=json.loads((dest/'source_manifest.json').read_text(encoding='utf-8'))
(receiptroot/'previous_source_manifest.json').write_bytes((dest/'source_manifest.json').read_bytes())
records={r['path']:r for r in manifest['records']}
pipeline=base/'work/polarity_intensity_integration_frozen_20261008T153052Z/bundle'
files=[]
for p in sorted(pipeline.iterdir()):
    if p.suffix not in ('.py','.npy'):continue
    files.append((p,'work/polarity_intensity_official_v1/'+p.name))
for name in ('prepare_polarity_native_bundle.py','audit_polarity_native_bundle.py','prepare_polarity_training_integration.py','freeze_polarity_integration_bundle.py','audit_polarity_integration.py','update_github_polarity_sources.py'):
    files.append((base/'work'/name,'work/'+name))
added=[]
for source,relative in files:
    original=source.read_bytes()
    clean,changes=(original,{}) if source.suffix=='.npy' else sanitize(original)
    if source.suffix=='.py':ast.parse(clean.decode('utf-8-sig'))
    target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(clean)
    records[relative]=dict(path=relative,origin=source.relative_to(base).as_posix(),kind='generated_order' if source.suffix=='.npy' else 'source',bytes=len(clean),
        original_sha256=hashlib.sha256(original).hexdigest(),published_sha256=hashlib.sha256(clean).hexdigest(),sanitizations=changes)
    added.append(relative)
manifest['records']=[records[k] for k in sorted(records)]
manifest['latest_source_update']='Polarity/intensity CPU synthetic module and export integration passed on two nodes; full encoder/real TRAIN/VAL/TEST not executed.'
(dest/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
counts=collections.Counter(str(Path(r['path']).parent).replace('\\','/') for r in manifest['records'])
index='# 源码目录索引\n\n完整逐文件来源和SHA见 `source_manifest.json`。\n\n|目录|文件数|\n|---|---:|\n'
for directory,count in sorted(counts.items()):index+=f'|[{directory}]({directory}/)|{count}|\n'
(dest/'CODE_INDEX.md').write_text(index,encoding='utf-8')
note='''
## 原流极性／强度候选：实现和验证状态

实现位于 [polarity_intensity_official_v1](work/polarity_intensity_official_v1/)。保留原100维、两步Euler、六方向消息流，在同一读出上加入零初始化的符号与非负幅度头（共2002参数）；文本、音频和视觉均可影响符号和幅度。固定比较 `regression_aux` 与 `factorized_aux` 两种等容量模式，使用同样的辅助监督。全局原gain及一维重标定捷径仍可能存在，这一候选没有替代折外效用控制或校准消融。

两台 Linux PyTorch 2.1.0+cu121 的 CPU 合成检查实际通过：标签／padding／排列／状态／RNG守卫、真实消息关闭与开启、梯度、优化器覆盖、保存／严格加载和回放。首次集成曾因 deepcopy 前向计算图失败，已改为重建模块再严格加载参数，原失败证据保留于研究备份。未使用真实数据、预训练权重或GPU前向；**完整编码器预检、正式训练及新的VAL/TEST五项结果仍未执行**。官方五项表仍是上方既有结果。

可运行的合成检查：

```bash
mkdir -p scratch
python work/polarity_intensity_official_v1/native_contract.py scratch/polarity_native.json
python work/polarity_intensity_official_v1/integration_contract.py scratch/polarity_integration.json
```

训练／推理接入已写入该目录的 `train_official.py`、`infer_official.py`，但运行前仍需新公共资产、完整编码器预检、来源冻结、保存容量及真实预算门通过。旧计划不能直接复用。主目标固定为最终带符号预测的Huber损失，幅度与符号监督仅作辅助；符号头零真值权重为0，最终MAE/Corr不裁剪预测。零头初始化的乘积读出仅在原预测约[-3,3]内还原原回归，范围外的anchor有显式裁剪，不能声称无条件恒等。
'''
readme=dest/'README.md';old=readme.read_text(encoding='utf-8-sig');assert '## 原流极性／强度候选：实现和验证状态' not in old
readme.write_text(old.rstrip()+'\n'+note,encoding='utf-8')
expected=set(records)|{'README.md','source_manifest.json','CODE_INDEX.md','.gitignore'}
findings=[]
for p in dest.rglob('*'):
    if not p.is_file() or '.git' in p.relative_to(dest).parts:continue
    relative=p.relative_to(dest).as_posix();assert relative in expected,relative
    if p.suffix=='.npy':continue
    text=p.read_text(encoding='utf-8-sig')
    if p.suffix=='.py':ast.parse(text)
    for kind,rx in [('provider_token',TOKENS),('private_key',PRIVATE),('credential_assignment',ASSIGN),('random_quoted',RANDOM_QUOTED)]:
        for hit in rx.finditer(text):
            if kind=='credential_assignment':
                value=hit.group(3)
                if value.startswith(('REDACTED','YOUR_','<','${')) or value in {'password','passwd','PASSWORD','example','placeholder'}:continue
            if kind=='random_quoted':
                value=hit.group(2)
                if not (any(c.isupper() for c in value) and any(c.islower() for c in value) and any(c.isdigit() for c in value)):continue
            findings.append(dict(path=relative,kind=kind,line=text.count('\n',0,hit.start())+1))
for r in records.values():assert hashlib.sha256((dest/r['path']).read_bytes()).hexdigest()==r['published_sha256']
assert not findings,findings
report=dict(status='PUBLIC_SOURCE_UPDATE_READY',new_source_paths=added,total_files=len(expected),total_source_files=sum(r['kind']=='source' for r in records.values()),all_manifest_SHA_pass=True,all_Python_AST_pass=True,secret_findings=0,
    real_data_labels_weights_or_raw_receipts_published=False,full_training_or_new_scores=False)
(receiptroot/'precommit_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status=report['status'],new_files=len(added),total_files=len(expected),secret_findings=0)))
