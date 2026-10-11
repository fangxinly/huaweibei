import datetime as dt, hashlib, json, shutil, zipfile
from pathlib import Path

base=Path.cwd();r=base/'work/plain_scalar_residual_fold0_20261006T1157Z';stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/('plain_scalar_residual_actual_'+stamp)
assert shutil.disk_usage('D:/').free>=1073741824;D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((r/'plan.json').read_text(encoding='utf-8'));results=json.loads((r/'execute/results.json').read_text(encoding='utf-8'));receipt=json.loads((r/'execute/receipt.json').read_text(encoding='utf-8'));audit=json.loads((r/'independent_local_array_audit.json').read_text(encoding='utf-8'))
assert audit['status']=='SEPARATE_LOCAL_CPU_NORMAL_EQUATION_AND_SCOPED_ARRAY_AUDIT_PASSED'
for p in r.rglob('*'):
    if p.is_file():dest=D/'original'/p.relative_to(r);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
for name in ['plain_scalar_residual_fold0_v1.py','audit_plain_scalar_residual_fold0_v1.py','plain_scalar_residual_fold0_v1_selfcheck.json','preserve_plain_scalar_fold0_v1.py']:
    shutil.copy2(base/'work'/name,D/'original'/name)
for name in ['研究接续状态.md','研究建议交流接续.json','租期末强化保存与下一阶段研究决策.md']:
    target=D/'preceding_state'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(base/'outputs'/name,target)
names=['zero','constant','affine'];table=[]
for arm in ['A','B']:
    for name in names:
        x=results[arm][name];table.append(f"| {arm} | {name} | {x['video_mse']:.9f} | {x['pooled_mse']:.9f} | {x['pooled_mae']:.9f} | {x['improved_videos']}/4 | {x['stable_heuristic']} |")
report=f'''# 同折标量残差：固定fold0实际CPU探索

实际完成UTC {receipt['utc']}。本地双精度闭式CPU拟合，耗时{receipt['seconds']:.3f}秒，Windows全进程峰值{receipt['peak_process_working_set_bytes']} bytes；自然进程exit0另存原工具回执。没有新GPU前向、教师参数更新、新100或异节点CPU执行。永久原件目录：{D.as_posix()}。

协议在真实标签拟合前冻结，SHA {sha(r/'plan.json')}；源SHA {plan['source_sha256']}。输入仅同一T_0已保存标量p_F；视频ID只用于分组及等权，不是模型特征。不用混折OOF的mu、旧C2、OUTER/CAL/EVAL或DEV/TEST。官方TRAIN标签容器只经角色守卫取出FIT695和INNER153对应值，完整容器仅作为不透明字节核SHA，不把其余标签数组整体物化。

A臂：30个FIT视频695行，拟合零/常数/仿射，固定岭0.01、截距不惩罚、FIT视频等权标准化。A预测与规则SHA实际写盘后才读取4个INNER视频153行标签统计。B臂事先同时冻结，在每个INNER视频留出时仅用另外三个视频拟合同样校正；同一教师不重训。B头排除对应留出视频标签，但该视频曾用于教师选checkpoint，且A评估已经读取整个INNER，本臂仅留一视频诊断，不能称新独立验证。

| 臂 | 方法 | 视频等权MSE | 片段MSE | 片段MAE | 改善视频 | 成本启发门槛通过 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(table)}

A常数c=-0.07411910583740507，INNER视频等权MSE降2.5612%，片段MSE降3.1793%；3/4视频改善，删任一视频后的冻结Q均值仍为负。这个预声明的门槛只是扩容成本启发式，不是显著性或风险保证。A仿射相对零也通过门槛，但相对常数视频MSE更差、仅2/4视频改善，不支持增加复杂度。B常数与仿射均未通过，视频等权MSE分别升约0.89%和4.45%。所有逐视频Q和删组值均保留，不挑fold/seed/岭系数重试。

FIT原残差RMS0.21385，INNER0.87951，相差明显；FIT仿射视频MSE0.017427而INNER0.731849。方向和幅度的分布迁移仍是瓶颈。A有小常数信号、B无稳定收益，不能归结为残差整体不可学习，也不能宣称样本内目标坍缩已被证明。应保留常数为未来公平参考的简单对照，暂停这批仿射扩容和消息/utility新100；独立确认与更丰富合法输入需新协议。

本表只属于T_0自身INNER探索。它不能与旧C2全TRAIN误差0.0161、旧固定EVAL消息成绩、1281行OOF MSE0.6294混成同一个排行榜。全TRAIN先前探索以及INNER教师选模复用都保留。新的完整流v2仍只有草案。

独立实现用加权正规方程重建两臂，最大预测差A{audit['A_B_prediction_max_errors']['A']:.3g}/B{audit['A_B_prediction_max_errors']['B']:.3g}；原预测/角色标签/行序/先冻结后评估/每个B头的留出组排除和所有指标核过。这是另一脚本的本地CPU审核，不冒异GPU节点审核或CPU整模型前向。当前GPU最新保存仍是UTC11:39；没有新增远端状态查询，租期平台未核。
'''
(base/'outputs/同折标量残差固定fold0实际探索结果.md').write_text(report,encoding='utf-8');(D/'original/同折标量残差固定fold0实际探索结果.md').write_text(report,encoding='utf-8')
manifest={str(p.relative_to(D/'original')).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in (D/'original').rglob('*') if p.is_file()}
with zipfile.ZipFile(D/'original_package.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for name in manifest:z.write(D/'original'/name,name)
with zipfile.ZipFile(D/'original_package.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest)
    for name,entry in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256']==sha(D/'original'/name)
proof=dict(status='ACTUAL_LOCAL_CPU_SOURCE_PLAN_ORIGINAL_ARRAYS_RESULTS_AND_SEPARATE_AUDIT_D_SHA_ZIP_CRC_MEMBERS_PASSED',utc=dt.datetime.now(dt.timezone.utc).isoformat(),directory=str(D),members=manifest,package_sha256=sha(D/'original_package.zip'),package_bytes=(D/'original_package.zip').stat().st_size,other_node_CPU_execution=False,remote_capture=False,lease_latest_real_capture_utc='2026-10-06T11:39:11Z',research_complete=False)
(D/'preservation_receipt.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
(r/'permanent_D_location.json').write_text(json.dumps({'directory':str(D),'receipt_sha256':sha(D/'preservation_receipt.json')},ensure_ascii=False),encoding='utf-8')
print(json.dumps({'directory':str(D),'members':len(manifest),'ZIPCRC_SHA_member_passed':True,'package_bytes':proof['package_bytes']},ensure_ascii=False))
