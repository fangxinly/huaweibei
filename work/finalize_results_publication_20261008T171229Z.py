"""Finalize public documentation only after real release asset verification."""
import collections
import hashlib
import json
from pathlib import Path
from publish_release_assets_v1 import credential_headers, request


def main():
    base = Path(__file__).resolve().parents[1]
    repo = base / 'github_upload_20261008T133346Z'
    folder = base / 'outputs/github_results_publication_20261008T171229Z'
    receipt_path = folder / 'release_upload_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt['status'] != 'PUBLISHED_ALL_ASSETS_REMOTE_SHA_VERIFIED':
        raise ValueError('Release has not completed')
    headers = credential_headers()
    remote = request('https://api.github.com/repos/' + receipt['repository'] + '/releases/' + str(receipt['release_id']), headers)
    if remote['draft'] or remote['tag_name'] != receipt['tag']:
        raise ValueError('Release is not public under expected tag')
    remote_assets = {r['id']: r for r in request(remote['assets_url'] + '?per_page=100', headers)}
    for row in receipt['assets']:
        uploaded = remote_assets[row['id']]
        if uploaded['state'] != 'uploaded' or uploaded['size'] != row['bytes'] or uploaded.get('digest') != 'sha256:' + row['sha256']:
            raise ValueError('Fresh remote asset identity/SHA mismatch')
        row['url'] = uploaded['browser_download_url']
    receipt['release_url'] = remote['html_url']
    receipt['fresh_public_metadata_verified'] = True
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    public = {k: receipt[k] for k in ['repository', 'tag', 'status', 'assets', 'release_url', 'fresh_public_metadata_verified']}
    (repo / 'results/release_asset_manifest.json').write_text(json.dumps(public, ensure_ascii=False, indent=2), encoding='utf-8')
    note = '# 下载与复现\n\n'
    note += f"[已发布的模型与 MOSI 输入]({receipt['release_url']})，共 {len(receipt['assets'])} 个附件，{sum(r['bytes'] for r in receipt['assets']):,} 字节。每个附件的远端 SHA256 已与上传字节流核对；大小、下载 URL、分片位置和原完整文件 SHA 见 [附件清单](../results/release_asset_manifest.json)。\n\n"
    note += '|附件|字节数|SHA256|\n|---|---:|---|\n'
    for row in receipt['assets']:
        note += f"|[{row['name']}]({row['url']})|{row['bytes']}|`{row['sha256']}`|\n"
    note += '''
对应官方结果：原流改进的 VAL MAE 为 0.597256537、TEST MAE 为 0.650616015；CaReFlow 的 TEST MAE 为 0.619535294；历史固定 F 的 TEST MAE 为 0.643699787。五项完整结果在 `results/official_upgrade_VAL_TEST.json` 与 `results/fixed_models_VAL_TEST_five_metrics.json`。此次上传没有产生新训练成绩。

## 先复核已保存的分数

克隆仓库并安装 NumPy 后，在仓库根目录运行：

```bash
python work/reproduce_published_VAL_TEST_v1.py
```

脚本按原五项口径复核原流改进 ON/OFF 的 VAL/TEST，以及 CaReFlow 和历史 F 的 TEST，误差容许值为 1e-12。CSV 便于阅读；原始 float32 预测值从 NPZ 读取，避免 CSV 字符串少量舍入造成差异。这里是既有预测的算术复现，不是重新选模型或新的独立实验。CaReFlow 的 VAL 点值保留在 JSON 中，此脚本没有据此声称重跑了它的 VAL 模型前向。

## 还原原流完整状态

下载 `official-upgrade-complete-final-and-selected.pt.part00`、`.part01` 与 `release_asset_manifest.json`，放到同一目录。需要额外至少 3.46GB 空间保存合并文件；脚本拒绝覆盖已有文件。

```bash
python work/reassemble_release_checkpoint_v1.py downloads/release_asset_manifest.json downloads official-upgrade-complete-final-and-selected.pt downloads/complete_final_and_selected.pt
```

原文件大小为 2,960,771,819 字节，SHA256 为 `9f670b5a2835be0d5d4638c34427024eadb326fc6e52ebff73a291ae2f2d34cf`。完整状态含最后第100轮、VAL选中的第30轮、优化器、调度器和RNG；**报告成绩对应 `selected_model`，不要用 `model` 的第100轮替代**。

原环境为 PyTorch 2.1.0+cu121；正式推理加载方式见 `work/official_anchored_upgrade_20261008T053429Z/infer_official.py`。归档协议含历史路径和机器资格，复现时需要生成新环境对应的资产/ID/源/预算计划，不能直接把旧租期计划当新执行资格。

`careflow-official-best93.pt` 对应官方 CaReFlow 第93轮；`minimal-F-official-best89.pt` 对应旧 F 第89轮。它们是已经选中的整模型，不是新的方法变体。输入资产包括当前使用的 MOSI 预处理 pickle、DeBERTa v3 base 权重、配置及分词模型。MOSI 文件包含原官方 TRAIN/DEV/TEST；训练仍应只索引 TRAIN，VAL 用于原选模，TEST 不能据此反复选结构。数据与公共预训练资产沿用上游来源和使用条件，本项目不宣称其所有权。

下载后的公共资产目录应按原接口放置（以下路径相对于传给 `--assets` 的根目录）：

|下载文件|资产目录中的位置|
|---|---|
|mosi.pkl|assets/mosi.pkl|
|deberta-v3-base-pytorch_model.bin|assets/deberta-v3-base/pytorch_model.bin|
|deberta-v3-base-config.json|assets/deberta-v3-base/config.json|
|deberta-v3-base-spm.model|assets/deberta-v3-base/spm.model|
|deberta-v3-base-tokenizer_config.json|assets/deberta-v3-base/tokenizer_config.json|

服务器密码未发布。本地原始权重、数据和冻结证据未因这次 GitHub 上传而删除。此前压缩后 SHA 无法核验的旧中间状态不在这些附件中；新 MSE 候选尚未完成真实训练。
'''
    (repo / 'docs/download_and_reproduce.md').write_text(note, encoding='utf-8')
    score = (folder / 'saved_score_reproduction.stdout.txt').read_bytes()
    if json.loads(score)['status'] != 'SAVED_PREDICTIONS_REPRODUCED_NOT_NEW_EXPERIMENT':
        raise ValueError('Saved arithmetic replay did not pass')
    (repo / 'results/upload_score_reproduction.json').write_bytes(score)
    original = (base / 'work/finalize_results_publication_20261008T171229Z.py').read_bytes()
    (repo / 'work/finalize_results_publication_20261008T171229Z.py').write_bytes(original)
    manifest = json.loads((repo / 'source_manifest.json').read_text(encoding='utf-8'))
    records = {r['path']: r for r in manifest['records']}
    for relative in ['results/release_asset_manifest.json', 'results/upload_score_reproduction.json', 'docs/download_and_reproduce.md', 'work/finalize_results_publication_20261008T171229Z.py']:
        blob = (repo / relative).read_bytes()
        records[relative] = dict(path=relative, origin='release_publication_finalization', kind='publication_result', bytes=len(blob),
            original_sha256=hashlib.sha256(blob).hexdigest(), published_sha256=hashlib.sha256(blob).hexdigest(), sanitizations={})
    manifest['records'] = [records[k] for k in sorted(records)]
    manifest['latest_source_update'] = 'Existing official VAL TEST predictions replayed; models and MOSI input assets actually published with remote SHA verification; no new scientific training.'
    (repo / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = collections.Counter(str(Path(r['path']).parent).replace('\\', '/') for r in records.values())
    index = '# 源码与结果目录索引\n\n逐文件来源和 SHA 见 `source_manifest.json`。\n\n|目录|文件数|\n|---|---:|\n'
    for directory, count in sorted(counts.items()):
        index += f'|[{directory}]({directory}/)|{count}|\n'
    (repo / 'CODE_INDEX.md').write_text(index, encoding='utf-8')
    print(json.dumps(dict(status='PUBLIC_RELEASE_DOCUMENTATION_READY', assets=len(receipt['assets']), url=receipt['release_url'])))


if __name__ == '__main__':
    main()
