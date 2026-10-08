"""Build only publication copies; original checkpoints and evidence are read-only."""
import ast
import collections
import hashlib
import json
from pathlib import Path
from prepare_github_source_upload import sanitize


def main():
    base = Path(__file__).resolve().parents[1]
    repo = base / 'github_upload_20261008T133346Z'
    out = base / 'outputs/github_results_publication_20261008T171229Z'
    out.mkdir(exist_ok=False)
    manifest = json.loads((repo / 'source_manifest.json').read_text(encoding='utf-8'))
    (out / 'previous_source_manifest.json').write_bytes((repo / 'source_manifest.json').read_bytes())
    records = {r['path']: r for r in manifest['records']}
    files = [
        ('work/publish_release_assets_v1.py', 'work/publish_release_assets_v1.py'),
        ('work/reassemble_release_checkpoint_v1.py', 'work/reassemble_release_checkpoint_v1.py'),
        ('work/prepare_results_publication_20261008T171229Z.py', 'work/prepare_results_publication_20261008T171229Z.py'),
        ('outputs/TRAIN_DEV退化定位与下一步训练方案.md', 'docs/TRAIN_DEV_diagnosis_and_plan.md'),
        ('outputs/固定双方TRAIN_DEV退化诊断.json', 'results/TRAIN_DEV_diagnosis.json'),
        ('outputs/TEST与既有0.59结果对齐分析.md', 'docs/TEST_vs_previous_059.md'),
        ('outputs/所有固定模型VAL_TEST五项实际对齐结果.json', 'results/fixed_models_VAL_TEST_five_metrics.json'),
        ('outputs/official_aligned_upgrade_review_20261008T111850Z/原方案改进与CaReFlow官方VAL_TEST实际结果.json', 'results/official_upgrade_VAL_TEST.json'),
        ('outputs/official_upgrade_completion_20261008T112205Z/官方对齐改进完成总结.md', 'docs/official_upgrade_completion.md'),
    ]
    backup = Path('D:/CodexBackups/selective_flow_20261003_1105')
    files.append((str(backup / 'official_anchored_upgrade_actual_20261008T053429Z/A_infer/extracted/out/fixed_official_VAL_TEST_prediction.npz'), 'results/official_upgrade_VAL_TEST_predictions.npz'))
    for source, relative in files:
        path = Path(source) if Path(source).is_absolute() else base / source
        original = path.read_bytes()
        clean, edits = (original, {}) if path.suffix == '.npz' else sanitize(original)
        if path.suffix == '.py':
            ast.parse(clean.decode('utf-8-sig'))
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(clean)
        origin = path.relative_to(base).as_posix() if path.is_relative_to(base) else 'preserved_official_prediction_archive/' + path.name
        records[relative] = dict(path=relative, origin=origin, kind='prediction_data' if path.suffix == '.npz' else 'research_result',
            bytes=len(clean), original_sha256=hashlib.sha256(original).hexdigest(),
            published_sha256=hashlib.sha256(clean).hexdigest(), sanitizations=edits)
    manifest['records'] = [records[k] for k in sorted(records)]
    manifest['latest_source_update'] = 'Actual official VAL/TEST results and predictions published; streaming release uploader added; no new scientific training.'
    (repo / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = collections.Counter(str(Path(r['path']).parent).replace('\\', '/') for r in records.values())
    index = '# 源码与结果目录索引\n\n逐文件来源和 SHA 见 `source_manifest.json`。\n\n|目录|文件数|\n|---|---:|\n'
    for directory, count in sorted(counts.items()):
        index += f'|[{directory}]({directory}/)|{count}|\n'
    (repo / 'CODE_INDEX.md').write_text(index, encoding='utf-8')
    readme = repo / 'README.md'
    text = readme.read_text(encoding='utf-8-sig')
    text = text.replace('**数据集、逐样本标签/预测、原始日志、模型权重/Adam/RNG 检查点、离线 wheel 包、凭据不在 Git 中**',
        '**本轮已补入逐样本预测和实际结果；对应模型与 MOSI 数据通过 Release 附件发布，下载状态见下方说明。服务器密码仍脱敏，原始捕获包与离线 wheel 不随此次发布**')
    text += '\n\n## 本轮结果与复现文件\n\n[TRAIN/DEV 退化诊断](docs/TRAIN_DEV_diagnosis_and_plan.md)、[旧 0.59 与 TEST 对账](docs/TEST_vs_previous_059.md)、[各固定模型 VAL/TEST 五项](results/fixed_models_VAL_TEST_five_metrics.json)、[原流改进实际结果](results/official_upgrade_VAL_TEST.json)和 [229/685 行预测数组](results/official_upgrade_VAL_TEST_predictions.npz)已补入。历史诊断文件按其原数据角色解释，不能把其中 INNER 或合并折结果当官方 VAL/TEST。\n\n模型/输入发布状态见 [下载与复现](docs/download_and_reproduce.md)。新 MSE 候选仍未完成真实训练；本轮是上传现有产物。存储压缩曾失败且后置 SHA 未完成，该文件不作为此次发布模型。\n'
    readme.write_text(text, encoding='utf-8')
    (repo / 'docs/download_and_reproduce.md').write_text('# 下载与复现\n\n对应官方 VAL/TEST 的模型与 MOSI 输入正在上传 Release，完成后写入逐附件 URL、大小和 SHA256。当前不要把上传准备当已经完成。\n\n不包含压缩后 SHA 未能核验的旧中间状态。新训练尚未开始。\n', encoding='utf-8')
    records['docs/download_and_reproduce.md'] = dict(path='docs/download_and_reproduce.md', origin='generated_publication_note', kind='research_note',
        bytes=(repo / 'docs/download_and_reproduce.md').stat().st_size,
        original_sha256=hashlib.sha256((repo / 'docs/download_and_reproduce.md').read_bytes()).hexdigest(),
        published_sha256=hashlib.sha256((repo / 'docs/download_and_reproduce.md').read_bytes()).hexdigest(), sanitizations={})
    manifest['records'] = [records[k] for k in sorted(records)]
    (repo / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'precommit.json').write_text(json.dumps(dict(files=[r for _, r in files], original_files_modified=False, credentials_published=False,
        current_weights_release_pending=True, real_new_training=False), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='PUBLICATION_COPIES_PREPARED', updated_files=len(files))))


if __name__ == '__main__':
    main()
