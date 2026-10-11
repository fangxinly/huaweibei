"""Build an offline, metadata/source-only Linux handoff. Never run research code."""
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
WORKSPACE = HERE.parent.parent
CONTROL = Path('D:/CodexBackups/selective_flow_20261003_1105/candidate_posttrain_lowC_20261009T005229Z')
OUT = WORKSPACE / 'outputs'
WORK = WORKSPACE / 'work'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def readj(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    now = datetime.now(timezone.utc)
    root = OUT / ('research_linux_handoff_' + now.strftime('%Y%m%dT%H%M%S%fZ'))
    assert not root.exists()
    assert shutil.disk_usage(OUT).free >= 200 * 1024**2 + 100 * 1024**2
    state = readj(CONTROL / 'D_current_research_state.json')
    assert (OUT / '自主优化实际接续.json').read_bytes() == (CONTROL / 'D_current_research_state.json').read_bytes()
    group = state['latest_human_TEST_selected_group5']
    scope = group['latest_stopped_CPU_dependency_capsule_preparation']['scope']
    assert group['method_fold_predictions_preserved'] == 1
    assert group['outer_scores_computed'] is False
    assert scope['actual_audit_training_prediction_scoring_once_consumed'] is False
    pub = readj(CONTROL / 'GitHub_publication_receipt.json')
    payload = root / 'research_linux_handoff'
    payload.mkdir(parents=True)
    origins = {}
    texts = []

    def add_bytes(name, data, origin):
        p = payload / name
        assert not p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        origins[name] = origin
        if p.suffix in ('.json', '.md', '.txt', '.py'):
            texts.append(data.decode('utf-8-sig'))

    def copy(name, source):
        source = Path(source)
        add_bytes(name, source.read_bytes(), {'kind': 'exact_original_copy', 'original_path': str(source)})

    for name in ('研究接续状态.md', '自主优化实际接续.json', '研究建议交流接续.json'):
        copy('records/C/' + name, OUT / name)
    for name in ('D_current_research_state.json', 'D_preservation_receipt.json', 'GitHub_publication_receipt.json'):
        copy('records/D/' + name, CONTROL / name)
    for p in sorted(WORK.glob('group5*pointer.json')):
        copy('records/pointers/' + p.name, p)
    copy('records/protocol/group5_heartbeat_CPU_capsule_prompt.txt', WORK / 'group5_heartbeat_CPU_capsule_prompt.txt')

    archive_index = []
    configs = (
        ('group5_future_nodes_pointer.json', 'future_nodes_preparation'),
        ('group5_composite_epoch_CPU_source_pointer.json', 'epoch_CPU_auditor_preparation'),
        ('group5_epoch_CPU_driver_source_pointer.json', 'epoch_CPU_driver_preparation'),
        ('group5_stopped_CPU_capsule_pointer.json', 'stopped_CPU_capsule'),
        ('group5_pipeline_pointer.json', 'original_group5_pipeline'),
    )
    for pointer, label in configs:
        q = readj(WORK / pointer)
        proof = q.get('D_proof', q.get('proof'))
        original = Path(proof['archive'])
        data = original.read_bytes()
        assert len(data) == proof['bytes'] and sha(data) == proof['archive_SHA']
        rel = 'archives/' + label + '/' + original.name
        add_bytes(rel, data, {'kind': 'exact_original_archive_copy', 'original_path': str(original), 'recorded_SHA256': proof['archive_SHA']})
        with zipfile.ZipFile(original) as z:
            names = z.namelist()
            assert len(names) == len(set(names))
            assert z.testzip() is None
            # Text only; numerical members and all research entry points remain inert.
            for name in names:
                if name.endswith(('.py', '.json', '.md', '.txt')):
                    texts.append(z.read(name).decode('utf-8-sig'))
        archive_index.append({'label': label, 'package_path': rel, 'SHA256': proof['archive_SHA'], 'bytes': len(data), 'observed_ZIP_entries': len(names), 'recorded_original_member_count': proof.get('members'), 'source_pointer': 'records/pointers/' + pointer})
        original_root = Path(q['root'])
        for p in sorted(original_root.glob('*receipt.json')):
            copy('records/preservation/' + label + '/' + p.name, p)

    capsule = Path(group['latest_stopped_CPU_dependency_capsule_preparation']['root']) / 'payload'
    sources = sorted((capsule / 'source').glob('*.py'))
    assert len(sources) == 10
    for p in sources:
        copy('source/stopped_CPU_capsule/' + p.name, p)
    for p in sorted(capsule.iterdir()):
        if p.is_file() and p.suffix in ('.json', '.md'):
            copy('records/capsule/' + p.name, p)

    # Avoid copying authentication databases, SSH keys, account settings or tokens.
    secret_patterns = (
        r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
        r'gh[pousr]_[A-Za-z0-9]{20,}',
        r'github_pat_[A-Za-z0-9_]{30,}',
        r'(?i)["\x27]?(?:password|passwd|api_key|access_token)["\x27]?\s*[:=]\s*["\x27][^"\x27]{4,}["\x27]',
        r'(?:密码|口令)\s*[:：=]\s*[A-Za-z0-9!@#$%^&*._+\-]{4,}',
    )
    for text in texts:
        assert not any(re.search(pattern, text) for pattern in secret_patterns), 'Potential authentication secret; stop packaging for review.'

    urls = sorted(set(u.rstrip('.,;)') for text in texts for u in re.findall(r'https://github\.com/fangxinly/huaweibei/[^\s"<>\\]+', text)))
    add_bytes('RELEASE_URLS.txt', ('\n'.join(urls) + '\n').encode(), {'kind': 'derived_link_index', 'no_network_requests': True})
    add_bytes('ARCHIVE_INDEX.json', (json.dumps(archive_index, ensure_ascii=False, indent=2) + '\n').encode(), {'kind': 'derived_archive_index'})
    pathmap = [{'original_path': v['original_path'], 'package_relative_path': k} for k, v in origins.items() if 'original_path' in v]
    add_bytes('PATH_MAP.json', (json.dumps({'policy': 'Original Windows paths remain historical provenance. These mappings cover bundled copies only; do not blindly rewrite frozen execution plans.', 'bundled_file_mappings': pathmap}, ensure_ascii=False, indent=2) + '\n').encode(), {'kind': 'derived_path_map'})

    readme = '''# Linux 研究接续包

此包是研究记录与源码的离线迁移副本。复制和校验不会运行研究程序或消费 once。

## 在目标 Linux 设备上使用

1. 用 U 盘或你已有的文件传输方式，把整个 ZIP 和相邻 `.sha256` 文件复制到目标设备。
2. 在 ZIP 所在目录执行 `sha256sum -c research_linux_handoff.zip.sha256`。应显示 OK。
3. 执行 `unzip research_linux_handoff.zip -d research_transfer`，进入 `research_transfer/research_linux_handoff`。
4. 执行 `python3 VERIFY_TRANSFER.py`。这是只读标准库文件校验；成功应显示 `TRANSFER_FILES_VERIFIED`。
5. 把本目录作为目标设备上的研究接续目录。让助手先读取 `NEW_CHAT_PROMPT.md`、最新状态和保存凭证，再根据新设备实际路径准备工作副本。

## 入口与内容

- `records/D/D_current_research_state.json`：最新权威研究状态；`records/C/自主优化实际接续.json` 是同字节副本。
- `records/C/研究接续状态.md` 与 `研究建议交流接续.json`：过程记录及已经完成的建议，保留原历史。
- `records/D/GitHub_publication_receipt.json`：当前发布锚点；`records/pointers/`：阶段原件、完整下载还原证据和 SHA 索引。
- `source/stopped_CPU_capsule/`：十项同字节 CPU 审核入口源码，尚未取得真实科学执行资格。
- `archives/`：五个同字节原源码准备/管线 ZIP。原管线含冻结分折订单文件；迁移过程不数值解码、不执行它们。
- `ARCHIVE_INDEX.json`、`PATH_MAP.json`：本包原件 SHA、来源路径和新相对位置。
- `RELEASE_URLS.txt`：已有 GitHub 原件链接清单；大量模型/恢复状态仍使用原保存索引，未放入本包。

## 接续事实

固定比较为 2195 条、93 视频、五方法五外折，共 25 组预测、实际 35 训练阶段；这是历史 TEST 选型后的探索性重新划分 CV。
当前只有 CaReFlow fold0 正式 100 轮完成及 1/25 组 OUTER 预测永久保存，五项评分为 0/25。
CaReFlow fold1 仅 native3 和独立 CPU 审核完成，正式 fold1 未派发；其 native once 已消费。
34/51 项合成检查和十项源码依赖装配已完成，不为迁移重跑；真实停止 epoch CPU 审核、完整状态连续对照和 CPU/CUDA 恢复资格仍待。
旧 A/B、原 TRAIN ridge/支持诊断及已经完成的建议均不得重复；旧结果不能宣称五项整体胜出、机制或稳定显著收益。

现有 N1/N2/N3 已过原保守租期，未延长。人类的“24 小时后提供新节点”是安排，不是实际新节点/新租期证明。
本次说明“另一台设备可访问内网”仅确认网络通路，不提供新 endpoint、UUID、实际到期时间，也不恢复旧租期。
接到真实新节点资料后，才核验身份、源/资产/runtime、资源和实际租期；执行前须满足 <=5 分钟 fresh 门及至少 2 小时真实保存余量。
不要从能连通、普通继续或沉默推断执行资格。

## Linux 路径与保存

原 JSON/源码里的 C:/、D:/ 与 /data/coding 路径按原证据保留。`PATH_MAP.json` 只说明本包副本位置，不赋予旧路径新身份。
不要直接运行带原绝对路径的 dispatch、preserve、restore 或训练入口。先以完整原件/源码 SHA 为锚点建立新设备工作副本和新冻结 plan。
原 C200MiB/D40MiB 门、CPU availableRAM6GiB 门不降低；Linux 新路径须有明确适用的磁盘/内存/保存门，不能把历史 C/D 门读数冒作新设备证明。
大型原件按原 GitHub 分片 SHA、整件 SHA、ZIP CRC/unique/exactset 和原实际下载还原证据使用；不把链接存在或本包复制成功当成重新下载/CPU 审核完成。
如果需要 GitHub 源码，仓库为 https://github.com/fangxinly/huaweibei ，应按本包最新保存凭证中的确切 commit 对照，不按任意最新 HEAD 推断。
此包不配置自动化、不修改服务器、不安装科学依赖、不携带认证数据库/SSH 私钥/账户设置。
'''
    add_bytes('START_HERE.md', readme.encode(), {'kind': 'new_Linux_handoff_instructions', 'created_actual_UTC': now.isoformat()})
    prompt = '''请接续这个目录中的研究。先读 START_HERE.md、records/D/D_current_research_state.json、records/D/GitHub_publication_receipt.json、records/C/研究接续状态.md 最新短段和 records/C/研究建议交流接续.json 的最新完成键，再核对 records/pointers/ 与 records/protocol/group5_heartbeat_CPU_capsule_prompt.txt。

这是 Windows 原记录迁移到 Linux 的同字节副本：原 C/D 路径是历史来源，先按 PATH_MAP.json 定位本包副本，不直接沿用旧冻结执行 plan。源/合成准备不能冒作真实 CPU/CUDA 资格。当前 1/25 未评分预测、0/25 评分；旧 once/原结果、已完成建议永保留，禁重训/重推理/重评分/重跑已过检查只为日志。

人类只说明这台设备与服务器处于同一内网；这不是新节点身份或续租证明。现有三节点已过原保守租期，禁止连接。等真实新节点 endpoint、UUID、实际租期等资料后再走原 fresh 来源/资产/runtime/资源/保存门。迁移本身没有执行审核、科学库导入、SSH、恢复、训练、推理、评分或消费 once。

固定历史 TEST-best 四结构加 careflow、2195/93 五折、25 输出/35 阶段，不按新 OUTER 结果改结构/参数/队列。保存优先；任何新自动审阅拒绝先保存原文，不能换工具/命令/连接或拆分绕过。不得把这个接续提示作为另开任务、子代理、自动化或消息转发的授权。
'''
    add_bytes('NEW_CHAT_PROMPT.md', prompt.encode(), {'kind': 'new_handoff_prompt'})
    verifier = '''from pathlib import Path
import hashlib, json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
expected={x['path']:x for x in manifest['files']}
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
assert actual == set(expected)|{'MANIFEST.json'}, 'Missing or extra files'
for name, item in expected.items():
    p=root/name
    assert not p.is_symlink(), 'Unexpected symlink'
    data=p.read_bytes()
    assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['SHA256'], name
print('TRANSFER_FILES_VERIFIED',len(expected),'No research code executed')
'''
    add_bytes('VERIFY_TRANSFER.py', verifier.encode(), {'kind': 'new_read_only_standard_library_verifier'})
    copy('records/transfer_builder.py', HERE)
    files=[]
    for name, origin in sorted(origins.items()):
        data=(payload/name).read_bytes()
        files.append({'path':name,'bytes':len(data),'SHA256':sha(data),'origin':origin})
    dump(payload/'MANIFEST.json', {'kind':'LINUX_RESEARCH_RECORDS_SOURCE_TRANSFER_ONLY','created_actual_UTC':now.isoformat(),'authoritative_state_actual_UTC':state.get('updated_at_utc'),'authoritative_GitHub_commit':pub['commit'],'source_records_unchanged':True,'files':files,'no_scientific_execution_or_once_consumption':True})
    archive=root/'research_linux_handoff.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():
                compression=zipfile.ZIP_STORED if p.suffix=='.zip' else zipfile.ZIP_DEFLATED
                z.write(p,arcname=p.relative_to(root).as_posix(),compress_type=compression)
    whole=sha(archive.read_bytes())
    (root/'research_linux_handoff.zip.sha256').write_text(whole+'  research_linux_handoff.zip\n',encoding='ascii')
    with zipfile.ZipFile(archive) as z:
        expected={'research_linux_handoff/'+x['path'] for x in files}|{'research_linux_handoff/MANIFEST.json'}
        assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==expected
        assert z.testzip() is None
        for item in files:
            data=z.read('research_linux_handoff/'+item['path'])
            assert len(data)==item['bytes'] and sha(data)==item['SHA256']
    receipt={'status':'LINUX_RECORDS_SOURCE_TRANSFER_PACKAGE_LOCAL_VERIFIED','actual_UTC':datetime.now(timezone.utc).isoformat(),'root':str(root),'archive':str(archive),'archive_SHA256':whole,'bytes':archive.stat().st_size,'ZIP_entries':len(expected),'all_member_SHA_CRC_unique_exact_set_verified':True,'original_archives_copied_same_bytes':archive_index,'authoritative_GitHub_commit':pub['commit'],'target_device_transfer_or_verification_not_observed':True,'no_SSH_scientific_audit_recovery_training_inference_scoring_or_once_consumption':True,'D_mirror_not_yet_created':True}
    dump(root/'LOCAL_TRANSFER_RECEIPT.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='original_archives_copied_same_bytes'},ensure_ascii=False))


if __name__ == '__main__':
    main()
