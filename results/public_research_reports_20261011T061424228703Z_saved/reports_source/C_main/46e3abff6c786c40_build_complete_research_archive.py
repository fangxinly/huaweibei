"""Offline export of visible research history and existing files; stdlib only."""
import collections
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
SESSIONS = Path('C:/Users/21234/.codex/sessions')
ROOTS = {
    'C_main': WS,
    'C_original': Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen'),
    'C_advice': Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review'),
    'D_preserved': Path('D:/CodexBackups/selective_flow_20261003_1105'),
}
THREADS = {
    'original': ('01a0f6eb-6bf5-71e2-96f0-176b880d6a53', 'Summarize multimodal emotion studies'),
    'main': ('01a109db-b31a-78d3-82f7-282321c5bf58', '整理多模态情绪研究聊天'),
    'advice': ('01a10fcb-6663-70a2-9a76-60e5634d0c03', '多模态消息控制研究建议'),
}
TEXT_EXT = {'.py', '.md', '.json', '.jsonl', '.log', '.txt', '.csv', '.tsv', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.sh', '.ps1', '.bat', '.r', '.ipynb', '.html', '.tex', '.xml'}
ARRAY_EXT = {'.npy', '.npz'}
SKIP_DIR = {'.git', '.venv', 'node_modules', '__pycache__', 'publication.git', '.ssh', '.aws'}
MAX_ZIP = 64 * 1024**2
MIN_FREE = 200 * 1024**2
TOKEN = re.compile(r'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,})')
ASSIGN = re.compile(r'''(?:password|passwd|密码|api[_ -]?key|access[_ -]?token|secret[_ -]?key)["']?\s*[:=：]\s*["']?([A-Za-z0-9!@#$%^&*_.+/=~-]{5,128})''', re.I)
STDIN = re.compile(r'''["']?chars["']?\s*:\s*(["'])(.*?)\1''', re.S)


def digest(b):
    return hashlib.sha256(b).hexdigest()


def jsonbytes(q):
    return (json.dumps(q, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def records(p, limit):
    with p.open('rb') as f:
        pos = 0
        line = 0
        while pos < limit:
            raw = f.readline()
            start = pos
            pos += len(raw)
            line += 1
            if pos > limit or not raw.endswith(b'\n'):
                break
            yield line, start, pos, json.loads(raw)


def visible(r):
    if r.get('type') != 'response_item':
        return None
    p = r.get('payload', {})
    t = p.get('type')
    if t == 'message':
        role = p.get('role')
        if role not in {'user', 'assistant'}:
            return None
        if role == 'assistant' and p.get('phase') not in {'commentary', 'final_answer'}:
            return None
        parts = [c.get('text', '') for c in p.get('content', []) if c.get('type') in {'input_text', 'output_text'}]
        return {'kind': 'message', 'id': p.get('id'), 'role': role, 'phase': p.get('phase'), 'text': '\n'.join(parts)}
    if t in {'function_call', 'custom_tool_call'}:
        return {'kind': 'tool_call', 'id': p.get('id'), 'call_id': p.get('call_id'), 'name': p.get('name'), 'text': p.get('arguments', p.get('input', ''))}
    if t in {'function_call_output', 'custom_tool_call_output'}:
        val = p.get('output', '')
        return {'kind': 'tool_output', 'id': p.get('id'), 'call_id': p.get('call_id'), 'text': val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)}
    # Internal reasoning, system/developer instructions and agent internals are not chat exports.
    return None


def discover():
    result = {}
    for label, (tid, title) in THREADS.items():
        rows = []
        for p in SESSIONS.rglob('*' + tid + '*.jsonl'):
            with p.open('rb') as f:
                meta = json.loads(f.readline())['payload']
            rows.append({'path': p, 'meta': meta, 'snapshot_bytes': p.stat().st_size})
        rows.sort(key=lambda x: x['meta']['timestamp'])
        assert rows, label
        for i, row in enumerate(rows):
            limit = row['snapshot_bytes']
            if i + 1 < len(rows):
                hb = rows[i + 1]['meta'].get('history_base')
                assert hb is not None
                alias = re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', row['path'].name)[-1]
                assert hb['thread_id'] == alias, (label, alias, hb)
                limit = hb['end_byte_offset']
                assert 0 < limit <= row['snapshot_bytes']
            row['canonical_limit'] = limit
        result[label] = rows
    return result


def gather_secrets(chains):
    values = set()
    ignore = {'password', 'passwd', 'false', 'true', 'none', 'null', 'undefined', 'input', 'getpass', 'secret', 'token', 'placeholder', 'os.environ', 'REDACTED'}
    for rows in chains.values():
        for row in rows:
            for _, _, _, r in records(row['path'], row['snapshot_bytes']):
                v = visible(r)
                if not v:
                    continue
                text = v['text']
                values.update(TOKEN.findall(text))
                for x in ASSIGN.findall(text):
                    if x.lower() not in ignore and re.search(r'\d|[!@#$%^&*]', x) and not x.startswith(('SHA', 'sha', 'http')):
                        values.add(x)
                if v['kind'] == 'tool_call' and 'write_stdin' in (v.get('name', '') + text):
                    for _, x in STDIN.findall(text):
                        y = x.replace('\\n', '').replace('\\r', '').strip()
                        if 5 <= len(y) <= 128 and re.fullmatch(r'[A-Za-z0-9!@#$%^&*_.+/=~-]+', y) and re.search(r'\d|[!@#$%^&*]', y):
                            values.add(y)
    return sorted(values, key=len, reverse=True)


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = WS / 'outputs' / ('complete_research_archive_' + stamp)
    out.mkdir()
    chains = discover()
    secrets = gather_secrets(chains)
    redactions = collections.Counter()

    def redact(text, kind):
        original = text
        for s in secrets:
            if s in text:
                redactions[kind] += text.count(s)
                text = text.replace(s, '[REDACTED_CREDENTIAL]')
        text, n = TOKEN.subn('[REDACTED_TOKEN]', text)
        redactions[kind] += n
        text, n = re.subn(r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z ]+ )?PRIVATE KEY-----', '[REDACTED_PRIVATE_KEY]', text, flags=re.S)
        redactions[kind] += n
        return text, text != original

    generated = out / 'generated'
    generated.mkdir()
    coverage = {'snapshot_UTC': datetime.now(timezone.utc).isoformat(), 'threads': [], 'unavailable_cloud_chats': [{'title': '实验结果总结', 'id': '6ac44cad-3168-83e9-94a9-d39699d392de', 'observed_error': 'cloudflare_challenge', 'contents_not_exported': True}], 'excluded': 'Internal reasoning and system/developer messages are not visible chat history. Credentials are redacted. No raw session database is copied.'}
    for label, rows in chains.items():
        counts = collections.Counter()
        chatdir = generated / 'chats' / label
        chatdir.mkdir(parents=True)
        handles = {k: (chatdir / (k + '.jsonl')).open('w', encoding='utf-8') for k in ['messages', 'automatic_heartbeats', 'tool_evidence', 'retired_segment_tail']}
        md = (chatdir / 'CHAT.md').open('w', encoding='utf-8')
        md.write('# ' + THREADS[label][1] + '\n\n用户可见消息的本地快照；密码已脱敏。自动心跳和工具原输出另见同目录 JSONL。\n\n')
        segments = []
        for row in rows:
            original_prefix_sha = hashlib.sha256()
            # Bind the exact observed prefix, without copying internal session contents.
            with row['path'].open('rb') as f:
                left = row['snapshot_bytes']
                while left:
                    b = f.read(min(left, 1024**2))
                    assert b
                    original_prefix_sha.update(b)
                    left -= len(b)
            segments.append({'original_path': str(row['path']), 'observed_prefix_bytes': row['snapshot_bytes'], 'observed_prefix_SHA256': original_prefix_sha.hexdigest(), 'canonical_prefix_limit': row['canonical_limit'], 'history_base': row['meta'].get('history_base')})
            for line, start, end, r in records(row['path'], row['snapshot_bytes']):
                v = visible(r)
                if not v:
                    continue
                v['text'], changed = redact(v['text'], 'chat')
                v.update(timestamp=r.get('timestamp'), segment=row['path'].name, line=line, byte_offset=start, credential_redacted=changed)
                if end > row['canonical_limit']:
                    target = 'retired_segment_tail'
                elif v['kind'] != 'message':
                    target = 'tool_evidence'
                elif '<heartbeat>' in v['text']:
                    target = 'automatic_heartbeats'
                else:
                    target = 'messages'
                counts[target] += 1
                handles[target].write(json.dumps(v, ensure_ascii=False) + '\n')
                if target == 'messages':
                    md.write('## ' + str(v['timestamp']) + ' — ' + v['role'] + '\n\n' + v['text'] + '\n\n')
        for f in handles.values():
            f.close()
        md.close()
        coverage['threads'].append({'label': label, 'id': THREADS[label][0], 'title': THREADS[label][1], 'counts': dict(counts), 'segments': segments})
        print('CHAT_EXPORTED', label, dict(counts), flush=True)
    (generated / 'CHAT_COVERAGE.json').write_bytes(jsonbytes(coverage))

    files = []
    for label, root in ROOTS.items():
        assert root.is_dir()
        for d, dirs, names in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIR and not x.startswith(('research_linux_handoff', 'complete_research_archive')))
            for name in sorted(names):
                p = Path(d) / name
                rel = p.relative_to(root).as_posix()
                if p.is_symlink():
                    continue
                st = p.stat()
                sensitive = p.name.lower() in {'auth.json', 'credentials', 'credentials.json', 'history.jsonl', '.env', 'id_rsa', 'id_ed25519'} or p.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}
                ext = p.suffix.lower()
                eligible = ext in TEXT_EXT | ARRAY_EXT or (ext == '.zip' and st.st_size <= MAX_ZIP)
                reason = 'sensitive_file_not_exported' if sensitive else ('included' if eligible else 'large_original_or_other_binary_index_only')
                files.append({'root': label, 'relative_path': rel, 'original_path': str(p), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'eligible': eligible and not sensitive, 'status': reason})
    assert shutil.disk_usage(out).free > MIN_FREE + 1024**3
    archive = out / 'complete_research_archive.zip'
    manifest = []
    objects = {}
    included_bytes = 0
    archive_member_redactions = 0
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        def put(name, data):
            z.writestr(name, data)
            manifest.append({'path': name, 'bytes': len(data), 'SHA256': digest(data)})

        def obj(data):
            h = digest(data)
            if h not in objects:
                name = 'objects/' + h[:2] + '/' + h
                put(name, data)
                objects[h] = len(data)
            return 'objects/' + h[:2] + '/' + h

        for i, q in enumerate(files):
            if not q['eligible']:
                continue
            p = Path(q['original_path'])
            data = p.read_bytes()
            if p.stat().st_mtime_ns != q['mtime_ns'] or len(data) != q['bytes']:
                q['status'] = 'changed_during_snapshot_not_included'
                continue
            q['original_SHA256'] = digest(data)
            ext = p.suffix.lower()
            changed = False
            if ext in TEXT_EXT:
                try:
                    text = data.decode('utf-8-sig')
                except UnicodeDecodeError:
                    text = data.decode('utf-8', errors='replace')
                    q['non_UTF8_original'] = True
                text, changed = redact(text, 'research_file')
                if changed:
                    data = text.encode('utf-8')
            elif ext == '.zip':
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as source:
                        ns = source.namelist()
                        assert len(ns) == len(set(ns))
                        assert source.testzip() is None
                        bad = []
                        for member in source.infolist():
                            if Path(member.filename).suffix.lower() in TEXT_EXT and member.file_size <= 128*1024**2:
                                b = source.read(member)
                                s, c = redact(b.decode('utf-8', errors='replace'), 'archive_member')
                                if c:
                                    bad.append((member.filename, s.encode('utf-8'), digest(b)))
                        if bad:
                            # Preserve every original archive unchanged; export only sanitized derivatives of affected text.
                            q['status'] = 'archive_contains_credentials_original_index_only'
                            q['redacted_text_members'] = [{'member': n, 'object': obj(b), 'original_member_SHA256': h} for n,b,h in bad]
                            archive_member_redactions += len(bad)
                            continue
                        q['ZIP_CRC_unique_members_verified'] = True
                        q['ZIP_members'] = len(ns)
                except (zipfile.BadZipFile, AssertionError) as e:
                    q['archive_format_problem'] = type(e).__name__
                    # Invalid/partial historical capture remains an exact byte copy, labeled as such.
            q['object'] = obj(data)
            q['exported_SHA256'] = digest(data)
            q['exported_bytes'] = len(data)
            q['status'] = 'redacted_derivative' if changed else 'exact_original_copy'
            included_bytes += len(data)
            if i % 300 == 0:
                assert shutil.disk_usage(out).free > MIN_FREE + 128*1024**2
                print('FILE_PROGRESS', i, 'of', len(files), 'unique_objects', len(objects), 'ZIP_bytes', archive.stat().st_size, flush=True)
        for p in sorted(generated.rglob('*')):
            if p.is_file():
                put(p.relative_to(generated).as_posix(), p.read_bytes())
        mapping = [{k:v for k,v in q.items() if k != 'eligible'} for q in files]
        put('FILE_MAP.json', jsonbytes(mapping))
        s = io.StringIO(newline='')
        cols = ['root','relative_path','original_path','bytes','status','original_SHA256','exported_SHA256','object']
        writer = csv.DictWriter(s, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(mapping)
        put('FILE_MAP.csv', s.getvalue().encode('utf-8-sig'))
        summaries = collections.Counter(q['status'] for q in files)
        counts = collections.Counter(q['root'] for q in files if q.get('object'))
        summary = {'schema': 1, 'actual_UTC': datetime.now(timezone.utc).isoformat(), 'scope': 'Existing research records, results, source and visible local chat history', 'file_status_counts': dict(summaries), 'included_files_by_root': dict(counts), 'unique_objects': len(objects), 'unique_object_bytes': sum(objects.values()), 'logical_included_bytes': included_bytes, 'credential_redaction_occurrences': dict(redactions), 'archive_redacted_text_members': archive_member_redactions, 'chat_coverage': coverage, 'no_scientific_execution': True, 'no_SSH': True, 'no_new_once_consumed': True, 'no_public_upload_of_chat': True, 'large_PT_and_large_ZIP_are_indexed_not_embedded': True, 'target_Linux_restore_has_not_occurred': True}
        put('COVERAGE.json', jsonbytes(summary))
        readme = '''# 实验研究、结果与聊天迁移包

这是实际文件快照和三个相关本地聊天的可见内容导出。先看 COVERAGE.json 和 CHAT_COVERAGE.json；FILE_MAP.csv 可用表格软件打开。

## Linux 使用

1. 用 U 盘或你自己的文件传输方式复制整个 ZIP 和同目录 .sha256 文件到 Linux。
2. 在 Linux 运行 `sha256sum -c complete_research_archive.zip.sha256`。
3. 解压到一个新空目录：`unzip complete_research_archive.zip -d research_archive`。
4. `cd research_archive`，运行 `python3 VERIFY_TRANSFER.py`；只校验字节，不导入研究代码。
5. 阅读 `chats/main/CHAT.md`、`chats/original/CHAT.md`、`chats/advice/CHAT.md`。同目录的 tool_evidence.jsonl 保存工具调用和实际输出；automatic_heartbeats.jsonl 保存自动协议；retired_segment_tail.jsonl 标明旧分段中未纳入当前主链的尾部记录，不冒当前状态。
6. 如需按原目录使用源码和结果，运行 `python3 RESTORE_FILES.py`。它只校验并复制文件到 `restored_records/C_main`、`C_original`、`C_advice`、`D_preserved`，不会启动训练、审核、推理、评分、SSH或自动化。

## 收录方式

- 原研究报告、记录、源码、日志、JSON、CSV、NPY/NPZ及不超过64MiB的原 ZIP，跨四个根目录收录。相同字节只存一份 objects，FILE_MAP 保留全部原路径和 SHA256。数组仅复制字节，未数值解码。
- 每个复制文件记录原 SHA256；脱敏衍生件另记导出 SHA256。敏感凭据不随包迁移；原文件未修改。自动心跳与聊天全文分开保存，未用摘要替换原可见内容。
- 超过64MiB的完整 ZIP、独立 PT 大模型和其他二进制只列原位置和实际大小；它们没有全部装进此包。相关原 SHA、Release 地址/分片和整件还原证明保留在收录的 publication/preservation receipt、pointer、index 和研究记录中。索引条目不代表本次重新取得或核验大原件。用 FILE_MAP 搜索原文件名定位对应回执。
- `.git`、虚拟环境、node_modules、凭据目录及本次/上次迁移包的重复目录不递归收录。运行依赖须在目标节点另外按原版本和原资产证明装配。原源码中的 Windows 路径保留原样，不能直接冒 Linux 执行资格。
- 云端“实验结果总结”聊天访问返回 cloudflare_challenge，内容未取得；此缺口已列 CHAT_COVERAGE。其它未被识别为这项研究的聊天未擅自收录。

## 最新结论及继续约束

最新科学状态以 D 原保存记录及最新 publication receipt 为准，旧历史快照仍全部保留。已有旧 A/B 100轮/4000更新、额外重放、CPU核验、五指标和异节点复核均不得重复；旧 A 只有 Acc2/F1 改善，旧 B 五项落后，没有整体五项胜出或稳定显著收益证明。历史 TEST 已见。

新全数据五折比较仍只保存1/25组未评分外折预测，评分0/25；35训练阶段未完成。CaReFlow fold0 完成100轮4700更新，INNER 最早选81；fold1只有native/CPU记录，未正式训练。新增恢复审核器、driver及10项依赖 capsule 是源码/合成准备，真实非最终epoch审核、完整CPU及CPU/CUDA连续状态资格仍待。迁移并不消费once或提升资格。

旧三节点租期已过且未延长。本设备网络不同与另一设备在内网，不构成续租或新节点授权。收到人类实际新节点资料后再核身份、来源、资产、runtime、fresh资源和实际租期以及至少2h保存门；禁止连接过期节点、重复已完成once或按外折结果改候选。

## 可粘贴到 Linux 新研究聊天的接续请求

“请先读这份迁移包的 README、COVERAGE、CHAT_COVERAGE，校验 MANIFEST，再通过 RESTORE_FILES 还原。研究内容以三个 CHAT.md 和原工具证据、D 保存记录为准。区分历史失败、准备、实际执行和实际保存；不要重新运行已完成实验、评分、once或连接过期节点。先核对实际新节点资料、原资产与资源门，报告可接续范围及缺口。”

本包含私人研究聊天，未公开上传。OpenAI 官方 app-server 的 thread/read/includeTurns 是读取聊天历史的接口说明：https://learn.chatgpt.com/docs/app-server 。本包采用本地分段可见消息导出，不宣称支持原生跨设备导入聊天数据库。
'''
        put('README.md', readme.encode('utf-8'))
        put('VERIFY_TRANSFER.py', VERIFIER.encode('utf-8'))
        put('RESTORE_FILES.py', RESTORER.encode('utf-8'))
        put('INDEX.html', ('<!doctype html><meta charset="utf-8"><title>研究迁移包</title><h1>实验研究迁移包</h1><p><a href="README.md">先读说明</a> · <a href="COVERAGE.json">覆盖范围</a> · <a href="FILE_MAP.csv">全部文件索引</a></p><ul>' + ''.join('<li><a href="chats/'+l+'/CHAT.md">'+t[1]+'</a></li>' for l,t in THREADS.items()) + '</ul><p>工具原输出和自动心跳位于每个聊天同目录。结果与源码用 RESTORE_FILES.py 还原；本页面不会运行研究代码。</p>').encode('utf-8'))
        # Manifest binds all preceding members; its own hash is bound by the outer ZIP SHA256.
        z.writestr('MANIFEST.json', jsonbytes(manifest))
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    (out / (archive.name + '.sha256')).write_text(h.hexdigest() + '  ' + archive.name + '\n', encoding='ascii')
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(z.namelist()) == {q['path'] for q in manifest} | {'MANIFEST.json'}
        for q in manifest:
            data = z.read(q['path'])
            assert len(data) == q['bytes'] and digest(data) == q['SHA256']
        assert z.testzip() is None
    receipt = {'actual_UTC': datetime.now(timezone.utc).isoformat(), 'archive': str(archive), 'SHA256': h.hexdigest(), 'bytes': archive.stat().st_size, 'ZIP_entries': len(manifest)+1, 'whole_ZIP_CRC_unique_exactset_all_member_SHA256_verified': True, 'coverage': summary, 'C_free_bytes_after': shutil.disk_usage(out).free, 'D_copy_pending': True}
    (out / 'EXPORT_RECEIPT.json').write_bytes(jsonbytes(receipt))
    (out / 'README.md').write_text(readme, encoding='utf-8')
    (WS / 'work' / 'complete_research_archive_pointer.json').write_bytes(jsonbytes({k:v for k,v in receipt.items() if k != 'coverage'}))
    print('COMPLETE', json.dumps({k:v for k,v in receipt.items() if k != 'coverage'}, ensure_ascii=False), flush=True)
    print('COUNTS', json.dumps(summary['file_status_counts']), 'REDACTIONS', dict(redactions), flush=True)


VERIFIER = '''import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
for q in manifest:
 p=(root/q['path']).resolve()
 assert root.resolve() in p.parents
 h=hashlib.sha256();size=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b);size+=len(b)
 assert size==q['bytes'] and h.hexdigest()==q['SHA256'],q['path']
print('VERIFIED',len(manifest),'members. No research code executed.')
'''
RESTORER = '''import hashlib,json,shutil
from pathlib import Path,PurePosixPath
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
for q in manifest:
 p=(root/q['path']).resolve();assert root.resolve() in p.parents
 h=hashlib.sha256();size=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b);size+=len(b)
 assert size==q['bytes'] and h.hexdigest()==q['SHA256'],q['path']
dest=root/'restored_records'
assert not dest.exists(),'Use a fresh empty destination; existing files are preserved.'
rows=json.loads((root/'FILE_MAP.json').read_text(encoding='utf-8'))
count=0
for q in rows:
 if not q.get('object'):continue
 rel=PurePosixPath(q['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts
 p=dest/q['root']/Path(*rel.parts);assert dest.resolve() in p.resolve().parents
 p.parent.mkdir(parents=True,exist_ok=True)
 shutil.copyfile(root/q['object'],p);count+=1
print('RESTORED',count,'existing record/source/result files. No research code executed.')
'''


if __name__ == '__main__':
    main()
