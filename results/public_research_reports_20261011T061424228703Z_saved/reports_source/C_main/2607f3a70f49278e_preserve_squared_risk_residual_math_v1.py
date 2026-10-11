import hashlib,json,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'work/squared_risk_residual_math_20261006T1259Z'
utc=datetime.now(timezone.utc)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
space={p:shutil.disk_usage(p).free for p in ('C:/','D:/')}
assert space['D:/']>100*1024**2
receipt=json.loads((SRC/'synthetic_math_receipt.json').read_text())
audit=json.loads((SRC/'independent_synthetic_audit.json').read_text())
assert sha(SRC/'synthetic_witnesses.npz')==receipt['arrays_sha256']
assert sha(SRC/'synthetic_math_receipt.json')==audit['original_receipt_sha256']
assert not receipt['new_task_scores'] and not receipt['official_data_or_labels_read']
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/('squared_risk_residual_math_'+utc.strftime('%Y%m%dT%H%M%SZ'))
D.mkdir(exist_ok=False)
original=D/'original';original.mkdir()
files=list(SRC.iterdir())+[ROOT/'work/squared_risk_residual_identity_v1.py',ROOT/'work/audit_squared_risk_residual_identity_v1.py',Path(__file__),ROOT/'outputs/平方误差收益与残差估计的条件信息及匹配对照决策.md']
members={}
for p in files:
 assert p.is_file() and p.name not in members
 q=original/p.name;shutil.copyfile(p,q)
 assert sha(q)==sha(p)
 members[p.name]={'bytes':q.stat().st_size,'sha256':sha(q)}
pkg=D/'original_package.zip'
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for name in sorted(members):z.write(original/name,name)
with zipfile.ZipFile(pkg) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(members)
 for name,m in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==m['sha256']
proof={'status':'LOCAL_SYNTHETIC_MATH_ORIGINALS_D_SHA_ZIP_CRC_MEMBERS_PASSED','actual_utc':utc.isoformat(),'free_bytes_before':space,'members':members,'zip_sha256':sha(pkg),'new_task_scores':False,'new_GPU_or_Torch':False,'other_node_CPU_or_remote_capture':False,'whole_research_or_lease_complete':False}
(D/'preservation_receipt.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(SRC/'permanent_D_location.json').write_text(json.dumps({'directory':str(D),'receipt_sha256':sha(D/'preservation_receipt.json')},indent=2)+'\n',encoding='utf-8')
state=ROOT/'outputs/研究接续状态.md'
short=state.read_text(encoding='utf-8')
shutil.copyfile(state,D/'preceding_short_state.md')
short=short.replace(short.splitlines()[0],f'更新UTC {utc.isoformat()}。只读短接续/按需原证据，不重注入历史。研究、后续实验和租期保存未整体完成。',1)
line=f'7a) 新平方误差Q/残差条件信息数学决策，D {D.as_posix()}，源{receipt["source_sha256"]}、合成数组{receipt["arrays_sha256"]}。10000合成float64与独立本地重建通过，非真实任务分数/官方标签/GPU/异节点CPU。Q=δ²-2δr，结构化Q的平方训练恰为4δ²残差加权；完整选择信息和真实δ不可省。未来先同输入同容量普通/加权残差、直接输出/同池Q对照，暂缓无约束Q扩头，当前未训练或正式协议。文献与条件详见平方误差收益与残差估计的条件信息及匹配对照决策.md。\n'
assert '7a) 新平方误差' not in short
short=short.replace('8) 第六',line+'8) 第六',1)
state.write_text(short,encoding='utf-8')
ledger_path=ROOT/'outputs/研究建议交流接续.json'
ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
assert 'squared_risk_residual_math_decision' not in ledger
ledger['squared_risk_residual_math_decision']={'batch':'squared_risk_residual_math_20261006T1259Z','phase':'MATHEMATICAL_ANALYSIS_SYNTHETIC_LOCAL_NUMPY_NOT_TASK_EXPERIMENT','permanent_D':str(D),'report':str(ROOT/'outputs/平方误差收益与残差估计的条件信息及匹配对照决策.md'),'source_sha256':receipt['source_sha256'],'arrays_sha256':receipt['arrays_sha256'],'report_sha256':sha(ROOT/'outputs/平方误差收益与残差估计的条件信息及匹配对照决策.md'),'sent_to_review':False,'actual_GPU':False,'new_task_scores':False,'decision':'First match information/capacity/roles for ordinary and amplitude-weighted residual, direct output and structured Q acceptance; unconstrained Q expansion deferred.'}
ledger['updated_at_utc']=utc.isoformat();ledger['updated_utc']=utc.isoformat()
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'directory':str(D),'members':len(members),'receipt_sha256':sha(D/'preservation_receipt.json'),'report_sha256':sha(ROOT/'outputs/平方误差收益与残差估计的条件信息及匹配对照决策.md'),'new_task_scores':False}))
