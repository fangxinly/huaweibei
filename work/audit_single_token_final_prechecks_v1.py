from pathlib import Path
import datetime,json,hashlib
root=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_risk_preflight_20261005T1434Z');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
w=json.loads((root/'c/single_token_warmup_receipt.json').read_text());wp=json.loads((root/'c/single_token_warmup_plan.json').read_text());f=json.loads((root/'a/single_token_full_raw_receipt.json').read_text());fp=json.loads((root/'a/single_token_full_raw_plan.json').read_text())
assert w['source_sha256']==wp['source_sha256']==sha('work/check_single_token_warmup_v1.py')
assert w['all_tensor_max_error']==0 and w['shared10_tensor_sha256']==w['original_shared10_tensor_sha256'] and w['after20_matches_original']
plan=json.loads(Path('work/finite_formal_plan_v1.json').read_text());assert w['initial_tensor_sha256']==plan['initial_tensor_sha256'] and w['orders_sha256']==plan['orders_sha256']
assert f['source_sha256']==sha('work/check_single_token_full_raw_v1.py')==fp['source_sha256']['check_single_token_full_raw_v1.py']
assert f['reader_source_sha256']==sha('work/finite_single_token_reader_v1.py')==fp['source_sha256']['finite_single_token_reader_v1.py']
assert f['train_row_ids']==list(range(31))+[620]
for v in f['modes'].values():assert v['full_vs_cached_error']<2e-5 and v['label_replacement_error']==0
assert not w['test_requested'] and not w['dev_requested'] and not f['test_requested'] and not f['dev_requested']
out=dict(status='SINGLE_TOKEN_INIT_ORDERS_SHAREDWARMUP_FULLRAW_ORIGINAL_RECEIPTS_INDEPENDENTLY_AUDITED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),warmup=w,warmup_plan=wp,full_raw=f,full_raw_plan=fp,remaining=['single-token-specific double finite difference/main-aux isolation extended check','new formal amended source/protocol/budget freeze before C2 launch'],scope='1281TRAIN shared10 gradients and200pilot proof separate; original C remains failed, no revised100 training started.')
Path('outputs/单token解析分支初始化订单共享期与原输入独立核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(f['modes']))
