from pathlib import Path
import datetime, hashlib, json

r = Path(__file__).resolve().parents[1]
receipt = sorted((r / 'outputs').glob('Jacobian完整研究与接续资料永久保存核验_*.json'))[-1]
record = json.loads(receipt.read_text(encoding='utf-8'))
assert record['status'] == 'ADDITIVE_PERMANENT_RECORDS_ALL_SHA_VERIFIED'
for row in record['rows']:
    assert hashlib.sha256(Path(row['destination']).read_bytes()).hexdigest() == row['sha256']
now = datetime.datetime.now(datetime.timezone.utc)
state = r / 'outputs/研究接续状态.md'
line = '\n最新报告保存UTC' + now.isoformat() + '：32份实际报告/源/三节点机制预检/关闭证据已追加永久D逐文件SHA核验，回执' + receipt.name + '；目录jacobian_reports_20261005T140239Z。此前1355部分复制因引用不存在的单独机制文件失败，部分目录保留，不能称完整；实际机制证据为冻结plan与三原preflight，未改科学协议。91816独立CPU七文件原回执仍待执行，保存依赖未全部结束。\n'
state.write_text(state.read_text(encoding='utf-8') + line, encoding='utf-8')
dest = Path(record['rows'][0]['destination']).parent / ('研究接续状态_保存后_' + now.strftime('%Y%m%dT%H%M%SZ') + '.md')
dest.write_bytes(state.read_bytes())
assert hashlib.sha256(dest.read_bytes()).hexdigest() == hashlib.sha256(state.read_bytes()).hexdigest()
proof = dict(utc=now.isoformat(), status='FINAL_SHORT_STATE_PERMANENT_SHA_MATCH',
    source=str(state), destination=str(dest), sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
    records_receipt=str(receipt))
with (dest.parent / ('state_addendum_' + now.strftime('%Y%m%dT%H%M%SZ') + '.json')).open('x', encoding='utf-8') as f:
    json.dump(proof, f, ensure_ascii=False, indent=2)
print(json.dumps(dict(status=proof['status'], files_verified=len(record['rows'])), ensure_ascii=True))
