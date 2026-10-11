import hashlib,json,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'work/locked_T0_constant_outer_development_20261006T1324Z'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
space={p:shutil.disk_usage(p).free for p in ('C:/','D:/')}
assert space['D:/']>=1073741824
rec=json.loads((SRC/'execute/receipt.json').read_text())
aud=json.loads((SRC/'independent_local_audit.json').read_text())
tools=json.loads((SRC/'original_tool_receipts.json').read_text(encoding='utf-8'))
assert all(tools[k]['exit_code']==0 for k in ('freeze','execute','independent_local_audit'))
assert aud['original_receipt_sha256']==sha(SRC/'execute/receipt.json')
assert rec['plan_sha256']==sha(SRC/'plan.json')
assert not aud['cost_heuristic_passed'] and not rec['new_fit']
utc=datetime.now(timezone.utc)
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105').resolve()
D=BASE/('locked_T0_constant_outer_development_actual_'+utc.strftime('%Y%m%dT%H%M%SZ'))
assert D.resolve().is_relative_to(BASE)
D.mkdir(exist_ok=False);original=D/'original';original.mkdir()
report=f'''# 同T0锁定常数外折开发测量实际结果

实际完成UTC {rec['actual_utc']}。同T0原OUTER433行、18视频的冻结常数校正，视频等权MSE从0.551755514降至0.534415092，降3.1428%；只有9/18视频改善，未达预声明12/18门槛，停止本标量分支扩张。平均收益不能替代跨视频覆盖条件。

| 方法 | 视频等权MSE | 片段MSE | 片段MAE | 改善视频 |
| --- | --- | --- | --- | --- |
| 原T0 F | 0.551755514 | 0.580429148 | 0.576066254 | — |
| F加原FIT常数 | 0.534415092 | 0.559269909 | 0.567529600 | 9/18 |

常数精确沿用原FIT695/30拟合结果c=-0.07411910583740507。本轮没有重拟合、换符号、调幅度、改fold/seed/岭/阈值；只比较F与F+c。预声明主比较为视频等权paired平方损失差Q，结果-0.0173404216957215。删任一视频后Q仍负，范围-0.02183685至-0.01115956；最大改善视频占净收益39.22%。三项成本启发条件为平均Q负、至少12/18视频改善、任意删一视频后Q负；覆盖项失败。它是成本停止规则，不是显著性检验或风险保证。

协议实际在2026-10-06T13:28:00.432307+00:00冻结，plan SHA f4a1a2d48d3dec88a5fc4850840be5a448f1fdd5c01839511eccbbf57044f1a0。先检查52视频资产/标签角色图及NPZ头，随后读取已保存T0 OUTER标量p_F，预测写盘SHA后才解码精确433行y。T0 OUTER未参与T0 FIT695/30、INNER153/4选模或原常数拟合；全TRAIN先前探索历史保留，因此只是后续开发测量，不是全新确认或全流程crossfit。旧CAL/EVAL行可能重叠，未使用其协议或数组；不能声称这些标签从未在过去实验被读过。本轮未解码FIT/INNER标签、混折OOF mu或DEV/TEST。

沿用T0 fold0完整教师best85，原checkpoint SHA338df042d648d3bdcb221bc0cacbd3d5574a1dd1efac49b1823f091a75706695；原OUTER预测SHA b02469e0fbd3329b7d3f8f115f04bfeddc9896f6cda5d9581c7b9eaaf7cfc9a4。它们为既有资产，不是本轮教师更新、新前向或weight下载。新预测SHA d82a608a16270f3507f6b1d7d3af2c85001345dc7f25e16f5ac406e954d901da。

CPU实际函数耗时{rec['elapsed_seconds']:.6f}秒；原工具自然exit0单独保存。没有测量新的进程峰值，不沿用旧实验35.8MB。独立脚本从原角色、原预测和精确行原y重建Q，最大差{aud['max_Q_reconstruction_error']:.3g}；SHA/行序/先冻结后读标签/逐视频及删组指标审核通过。这是独立实现的本地CPU数组审核，不是异节点CPU或CPU整模型前向。

决定：保留已锁定常数为T0历史/开发简单基线，停止本批常数、仿射扩容及救分；不据这个平均降幅启动Q头、消息100或无约束Q网络。更丰富合法信息需要不同科学问题、标签角色与停止标准的新协议。这不是否定全部残差或消息研究。新完整流参考的可行性是另一项工作，不能用本结果替代其实际梯度、严格重放及预算依赖。

原件永久D：{D.as_posix()}。本批没有新GPU、Torch、参数更新或远端capture，研究和租期保存未整体完成。对第八审视的采纳与证据范围补充见第八审视采纳与最小匹配对照修订.md。
'''
(ROOT/'outputs/同T0锁定常数外折开发测量实际结果.md').write_text(report,encoding='utf-8')
entries=[(p,p.relative_to(SRC).as_posix()) for p in SRC.rglob('*') if p.is_file()]
names=['locked_T0_constant_outer_development_v1.py','freeze_locked_T0_constant_outer_development_v1.py','audit_locked_T0_constant_outer_development_v1.py','preserve_locked_T0_constant_outer_development_v1.py']
entries += [(ROOT/'work'/n,'sources/'+n) for n in names]
entries += [(ROOT/'outputs'/n,'reports/'+n) for n in ('同T0锁定常数外折开发测量实际结果.md','第八审视采纳与最小匹配对照修订.md')]
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第八次独立审视_结构化收益条件与最小匹配对照.md')
entries.append((review,'received_review/'+review.name))
members={}
for p,name in entries:
 assert name not in members
 q=original/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
 assert sha(p)==sha(q)
 members[name]={'bytes':q.stat().st_size,'sha256':sha(q)}
pkg=D/'original_package.zip'
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for name in sorted(members):z.write(original/name,name)
with zipfile.ZipFile(pkg) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(members)
 for name,m in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==m['sha256']
proof={'status':'ACTUAL_LOCAL_CPU_LOCKED_CONSTANT_D_SHA_ZIP_CRC_UNIQUE_MEMBERS_PASSED','actual_utc':datetime.now(timezone.utc).isoformat(),'free_bytes_before':space,'members':members,'package_sha256':sha(pkg),'package_bytes':pkg.stat().st_size,'actual_local_CPU':True,'new_GPU':False,'other_node_CPU':False,'new_fit':False,'new_remote_capture':False,'new_confirmation':False,'whole_research_or_lease_complete':False}
(D/'preservation_receipt.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(SRC/'permanent_D_location.json').write_text(json.dumps({'directory':str(D),'receipt_sha256':sha(D/'preservation_receipt.json')},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'directory':str(D),'members':len(members),'receipt_sha256':sha(D/'preservation_receipt.json'),'report_sha256':sha(ROOT/'outputs/同T0锁定常数外折开发测量实际结果.md'),'eighth_review_sha256':sha(review),'status':proof['status'],'free_bytes_before':space},ensure_ascii=False))
