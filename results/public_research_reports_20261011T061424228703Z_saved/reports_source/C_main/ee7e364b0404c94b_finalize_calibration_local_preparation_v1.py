"""Preserve new local preparation and a short continuation; no remote operation."""
from pathlib import Path
import datetime, hashlib, json, shutil, zipfile

workspace = Path(__file__).resolve().parent.parent
work = workspace/'work'
outputs = workspace/'outputs'
prepared = work/'correction_calibration_preparation_20261006T0354Z'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
free = {'c': shutil.disk_usage('C:/').free, 'd': shutil.disk_usage('D:/').free}
assert free['d'] >= 1024**3 and free['c'] >= 100*1024**2
proof = json.loads((outputs/'纠错幅度校准角色与消息回退本地机制核验.json').read_text(encoding='utf-8'))
assert proof['status'] == 'LOCAL_VIDEO_ROLES_GUARDS_AND_NONLINEAR_MESSAGE_BACKTRACK_CHECKED_NOT_GPU_EXECUTED'
pins = {'group_teacher_runtime_v2.py': '6f9074453ed15c32537efb7803210801f6123cc1647c6f4fe4d7126e2616dc9e',
        'train_group_teacher_v1.py': '7c94f90ff648909e0abe1a6ccdb7a88543650142c6633ebadb8a0062068b8791',
        'diagnose_oof_teacher_utility_v1.py': '5a38424e72429b88ea3ddeba4c5dc7fc036652c37b6a78ac52fefc90d07c3087',
        'capture_soft_vector_v19.py': '6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'}
for name, expected in pins.items():
    assert sha(work/name) == expected
prior = outputs/'研究接续状态_本地校准准备前_20261006T0400Z.md'
assert not prior.exists()
shutil.copy2(outputs/'研究接续状态.md', prior)
new_state = f'''最新本地状态UTC{now}；最后成功远端C/B实际UTC03:38:53，本轮未连接或发送远端命令。

# 多模态流研究短接续

当前远端发送限制：用户进展查询轮UTC03:48，已认证SSH A17373/B56428/C11935的只读查询均被write_stdin自动审批以“approval required by policy, but AskForApproval is set to Never”拒绝，未传远端；三会话退出未知、当前UUID/compute/argv/空间未知。证据进展查询实时核查拒绝_20261006T034834Z.json。拒绝后没有换命令/工具/客户端/新连接绕过，本轮也未重复试探。不归因服务器/密码或用户没授权，不称工具已修复。此前03:38:53 SSH91834/1575、SFTP17019/85493真实exit0只属于旧ID，禁复用。

新本地准备：纠错幅度校准本地实现接续.md。源与计划在work/correction_calibration_preparation_20261006T0354Z已冻结；同T_k FIT/INNER零标签采集器、标签角色守卫和实际消息减半/终端输出上限/回退原语已实现本地草稿。采集plan SHAefb55291d836c850ab8b34d375466940fa163fe274efc342d8e94928a1ce1903；角色plan SHA785c27b7bfb53796e3ee5d6cc9737009c85a29c1124a6633538f0e526722f0cc。新分离CAL/EVAL按视频固定SHA顺序，无标签或指标读入：三折136/297、148/276、134/290行；合18视频418行校准、34视频863行评估。原FIT/INNER/OUTER695/153/433、715/142/424、714/143/424不改；学生S_k目标只同T_k，禁混不同老师导致间接外折泄漏。

本地独立NumPy/AST、毒化标签/四越权请求、非线性真实消息重放、零上限/反方向回退、跨样本耦合拒绝已过，纠错幅度校准角色与消息回退本地机制核验.json。capture19递归子根/后缀静态覆盖检查过（初版把__pycache__目录当文件的exit1保留，v2只排除解释器缓存，不改科学/捕获源）。这是本地准备，不是已部署、真实GPU机制通过、FIT/INNER新采集、λ/ε/输出上限校准、消息收益或新100轮。所有这些实际步骤仍未执行。新预算草案20min/4GiB峰值/1GiB新资料/远端2GiB与本地1GiB空闲/2h保存余量须真实预检。现有原模型构造优化器对象随后丢弃、零步骤，报告不冒称从未构造。实际INNER磁盘原输入重放≤1e-6、fit统计缓冲exact、换标签0、整参数SHA不变和fresh源/UUID/空compute/空间依赖未过不得执行。新子根须实际capture覆盖核验/原始argv与退出/ZIP审核，静态覆盖不冒实际capture。

科学最新已完成：OOF教师驱动消息修改实验分析.md、教师OOF效用短实验执行状态.md。C新诊断PID12392 UTC03:33:17自然exit0，预检12257退休；固定C2best37/原β65294.20088076077/原尺度/三步.25/消息信赖域.25，1281TRAIN/52视频。rho=p0−μOOF，y只预测写盘SHA后统计，参数无更新，无DEV/TEST。原F MSE.0161276872，旧学习.0152679208，新OOF末步.0767956564害74.89%，μ最佳.0789019134害75.10%；新路径负结果，旧候选池选择4.00%不冒新生成收益。F误差RMS.127、教师分歧.8007、新rho.8616/旧rho.1749，线性风险均值−.007983+平方幅度.068651净+.060668。全TRAIN事后λ.0217仅解释，不作正式超参数；经验ε不是条件均值真界。

新短诊断21原文件永久D oof_utility_completed_20261006T033054Z/B独立NumPy原回执已下载。B/C实际03:35:43.880070/.883579 capture19 CAPTURE_COMPLETE+exit0后下载，全SHA/CRC/member及旧349/325成员、大引用不变联合过：OOF教师效用新诊断D与独立CPU联合快照核验.json。A无新材料仅原03:08完成快照，不能冒新三节点capture。大weight引用不是新下载；新诊断非CPU整模型前向。

三91818视频老师各100自然exit0、best85/45/68，各739126305bytes/301状态张量严格INNER原输入磁盘重放0，原节点训练组装、CPU A→B/B→C/C→A三原回执下载/D group_teacher_completed_20261006全SHA/CRC/选模/init/订单保留不重传。公共DeBERTa+随机任务init、fit-only统计、INNER最早最小选100、OUTER零标签；184749003保留参数291全微调，与旧流冻结骨干只520506供体头不同。1281行52视频OOF已合并、B19文件CPU原回执下载/D group_teacher_oof_20261006T0308Z，μ NPZ SHAcff1a52f7e828c96cacf070ae4e051a5c34042ebb9435f44f5b3242cad91a7ea，MSE.6293898611/MAE.5965080822，对fit均值降72.76%，仅教师质量非新控制收益。现有A参考仍全TRAINfit/DEVselected，已探索全TRAIN不冒未触碰确认集或全流程crossfit。

九91815/91816/91817 A/B/C2完成100/full/20诊断/D/独立CPU及TRAIN Oracle65.11%标签已知有限步诊断已完成不重跑重传。原C仅10自然失败，C2不是旧optimizer恢复；旧v5C81/临时31不冒最终100，旧14:30缺口不回填。旧84/v2/v3/v4/v5A/B保留不重复；旧址/N/R永禁连。

唯一新A REDACTED_SERVER_HOST.invalid:53449 UUID GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7；B REDACTED_SERVER_HOST.invalid:53416 GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067；C REDACTED_SERVER_HOST.invalid:53458 GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa。正式允许发送恢复后先fresh核UUID/完整argv/compute/历史或完成/源/空间；实际password提示后仅本聊天人类原凭据，禁密码文件/命令/自动提示/猜测，明确exit/bye核实际结果。capture19 SHA6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad，公共/data/coding/soft_vector_research_20261005T1220Z/deployment Jacobian1327，真实clock唯一stamp、CAPTURE_COMPLETE/exit0后receipt/ZIP全SHA/CRC/member。

本轮本地新准备和短证据永久D correction_calibration_preparation_20261006T0400Z，保存清单另见纠错幅度校准本地准备永久保存.json；仅本地保存，不是新远端/独立CPU保存。fresh C/D空闲{free['c']}/{free['d']}bytes，D优先禁删旧文件。24h首次Oct5UTC12:08:17仅估Oct6UTC12:08:17/BJ20:08:17平台未核；UTC08:08/10:08/11:38仍需真实强化保存，受阻不伪称或回填。禁TEST选结构/五seed稳定/SOTA/语义真值/subagents/设置/浏览器/续租/关机/停健康训练/改完成冻结源。十分钟频率健康未变或不可行动状态静默；只实质结果/新失败/完成/必要行动通知，全部研究与后续实验/租期保存未完成。
'''
(outputs/'研究接续状态.md').write_text(new_state, encoding='utf-8')
backup = Path('D:/CodexBackups/selective_flow_20261003_1105/correction_calibration_preparation_20261006T0400Z')
assert not backup.exists()
backup.mkdir()
files = [(path, Path('prepared')/path.name) for path in prepared.iterdir() if path.is_file()]
files += [(work/name, Path('sources')/name) for name in
          ['audit_correction_calibration_preparation_v1.py', 'audit_new_collection_capture_coverage_v2.py',
           'finalize_calibration_local_preparation_v1.py']]
files += [(outputs/name, Path('evidence')/name) for name in
          ['纠错幅度校准本地实现接续.md', '纠错幅度校准角色与消息回退本地机制核验.json',
           '新校准采集子根capture19静态覆盖检查.json', '研究接续状态.md', prior.name,
           '进展查询实时核查拒绝_20261006T034834Z.json']]
members = {}
for source, relative in files:
    target = backup/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert sha(source) == sha(target)
    members[relative.as_posix()] = {'sha256': sha(target), 'bytes': target.stat().st_size}
archive = backup/'local_preparation_payload.zip'
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
    for relative in sorted(members):
        z.write(backup/relative, relative)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(members)
    for name, expected in members.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == expected['sha256']
record = {'status': 'LOCAL_NEW_CALIBRATION_PREPARATION_D_FILES_ZIP_CRC_MEMBER_SHA_VERIFIED_NOT_REMOTE_SAVE',
    'actual_saved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'directory': str(backup),
    'files': len(members), 'members': members, 'zip_sha256': sha(archive), 'zip_bytes': archive.stat().st_size,
    'fresh_free_bytes_before': free, 'completed_original_sources_unchanged_sha': pins,
    'new_remote_capture': False, 'new_independent_remote_cpu_copy': False,
    'new_gpu_execution': False, 'new_calibrator_fit': False, 'new_training': False}
(backup/'preservation_manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
(outputs/'纠错幅度校准本地准备永久保存.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({key: record[key] for key in ['status', 'actual_saved_utc', 'directory', 'files', 'zip_sha256', 'zip_bytes']}))
