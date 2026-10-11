import hashlib,json,shutil,sys,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]
clock=sys.argv[1]
out=base/'outputs/polarity_intensity_native_20261008T151107Z'
github=base/'outputs/github_polarity_update_20261008T153225Z_v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
native=json.loads((out/'native_joint_verification_v2.json').read_text(encoding='utf-8'))
integration=json.loads((out/'integration_joint_20261008T153052Z.json').read_text(encoding='utf-8'))
failed=json.loads((out/'integration_joint_20261008T152915Z.json').read_text(encoding='utf-8'))
publication=json.loads((github/'blob_verification_6d45d54757b6.json').read_text(encoding='utf-8'))
assert integration['status']=='DUAL_NATIVE_INTEGRATION_SYNTHETIC_D_PASSED' and not publication['index_blob_SHA_failures']
report=dict(status='FOURTH_LEASE_DUAL_NATIVE_CPU_SYNTHETIC_POLARITY_INTENSITY_AND_INTEGRATION_PASSED_NOT_REAL_TRAINING',
    actualclock_before_save_UTC=clock,human_lease_reply='24h后',platform_expiry_confirmed=False,
    conservative_cutoff_UTC='2026-10-09T15:00:00+00:00',conservative_cutoff_Beijing='2026-10-09 23:00',save_reserve_seconds=7200,
    new_servers_read_only_bootstrap={n:json.loads((out/n/'bootstrap.json').read_text(encoding='utf-8')) for n in ('A','B','C')},
    module_joint=dict(path=str(out/'native_joint_verification_v2.json'),SHA256=sha(out/'native_joint_verification_v2.json'),D=native['D']),
    integration_joint=dict(path=str(out/'integration_joint_20261008T153052Z.json'),SHA256=sha(out/'integration_joint_20261008T153052Z.json'),D=integration['D']),
    first_integration_failure=dict(path=str(out/'integration_joint_20261008T152915Z.json'),SHA256=sha(out/'integration_joint_20261008T152915Z.json'),D=failed['D'],
        cause='deepcopy of module holding non-leaf forward graph tensors; fixed with fresh constructor and strict state_dict load under fork_rng',originals_preserved=True),
    candidate=dict(original_flow_preserved=True,steps=2,directions=6,additional_parameters=2002,
        fixed_candidate_mode='factorized_aux',equal_capacity_control='regression_aux',both_modes_same_fixed_auxiliary_objective=True,
        encoder_freeze_or_finetune='Future official pipeline preserves end-to-end fine-tuned public encoder; not executed in current CPU synthetic checks',
        core_innovation_OOF_utility_controller=False,global_gain_calibration_shortcut_still_present=True,
        real_TRAIN_VAL_TEST=False,full_encoder_precheck=False,new_scores=False),
    pending=['permanent backup path with at least 20GB free','public asset/runtime staging and full encoder precheck on new nodes','complete source/asset/ID/budget freeze','real official TRAIN and VAL selection','fixed VAL/TEST five metrics aligned to CaReFlow'],
    storage={drive:dict(free=shutil.disk_usage(drive+':/').free,total=shutil.disk_usage(drive+':/').total) for drive in ('C','D','E')},
    no_additional_deletion=True,no_renewal_or_shutdown=True,previous_tests_not_repeated=True,
    sessions=json.loads((out/'session_closure.json').read_text(encoding='utf-8')),
    github=publication,public_data_labels_models_credentials_uploaded=False,
    latest_real_official=dict(VAL=dict(ours_MAE=.5972565370388166,CaReFlow_MAE=.6045238262758688),
        TEST=dict(ours_MAE=.650616015,CaReFlow_MAE=.619535294),new_candidate_results=False,all_five_improved=False))
(out/'fourth_lease_state.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
text='''# 原流极性／强度候选：新服务器实际验证

两台新服务器已真实通过PyTorch 2.1.0+cu121的CPU合成模块和集成验证；完整源码、计划、执行PID/argv、前后UUID/进程/空间capture、自然退出、stdout/stderr和原ZIP均已D保存与SHA/CRC/唯一成员联合核验。没有真实数据训练、GPU前向、完整编码器预检或新VAL/TEST分数。

保持原100维、两步Euler和六方向消息。两个新增零初始化线性头共2002参数，文本/音频/视觉均可纠正符号与幅度，不固定“文本决定符号、音视决定强度”。主目标是最终带符号预测的Huber；幅度/符号监督仅辅助。同容量原回归加辅助监督作固定对照，以区分乘积读出和额外监督的效果。全局gain与一维校准捷径仍在，未声称实现完整OOF残差效用控制。

原训练/预测入口已接入。双方模式各3步合成更新、所有被优化参数梯度有限、优化器无重复漏项、消息开关、完整读出保存与严格加载回放均通过。首次保存deepcopy计算图自然失败A340/B382，原件完整保留；改为重建结构+严格加载参数后A400/B457自然0。没有重复历史实验。

下一正式实验沿用官方TRAIN1281、VAL229、TEST685，共同seed128、100轮4000更新、batch32/drop_lastTrue和原订单、VAL128/101批均值MSE最早最小选模，固定方法后报告同checkpoint五项；不以INNER或TEST选择结构。公共资产、完整编码器新预检、新来源/预算计划和长期保存容量仍须落实，当前不能把合成验证当成正式训练完成。

人类只回复“24h后”。保守执行截止为北京时间2026-10-09 23:00（UTC15:00），不是平台确认，预留2小时保存。三机bootstrap为空GPU compute，C的P4为7680MiB、A/B为8192MiB，不能冒相同容量或当前实时状态。所有六个SSH/SFTP会话已明确exit/bye自然0，旧会话禁复用；凭据不落盘。无续租、额外删旧文件、关机或停止训练。

GitHub main已同步33份源码/固定订单及状态说明，commit `6d45d54757b6674101752a7f024fad811269ffaf`，1751文件；全部清单和Git blob SHA一致、远端main一致、checkout clean、秘密扫描0。Git换行转换已补正，实验原冻结资料不改。首次发布和失败证据留存。

目前没有20GB长期备份路径。新服务器各约47GB空闲属于临时租期空间；本地C约1.23GB、D约0.33GB，不能代替完整训练状态备份。已向人类询问路径，未得到该部分回复。既有官方VAL MAE .597256537、TEST .650616015；CaReFlow .604523826/.619535294，当前没有五项超过。
'''
(out/'REPORT.md').write_text(text,encoding='utf-8')
dest=Path('D:/CodexBackups/selective_flow_20261003_1105/fourth_lease_polarity_summary_20261008T153225Z');dest.mkdir(exist_ok=False)
files=[out/'fourth_lease_state.json',out/'REPORT.md',out/'session_closure.json',out/'local_reference.json',out/'first_audit_failed_source.py',
       out/'native_joint_verification.json',out/'native_joint_verification_v2.json',out/'integration_joint_20261008T152915Z.json',out/'integration_joint_20261008T153052Z.json',
       github/'precommit_verification.json',github/'blob_verification_63f466afdc78.json',github/'blob_verification_6d45d54757b6.json',github/'published_source.bundle']
for p in files:shutil.copyfile(p,dest/p.name);assert sha(p)==sha(dest/p.name)
with zipfile.ZipFile(dest/'small_evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in files:
        if p.suffix!='.bundle':z.write(p,p.name)
with zipfile.ZipFile(dest/'small_evidence.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
receipt=dict(status=report['status'],actualclock_before_save_UTC=clock,report=str(out/'REPORT.md'),D=str(dest),
    small_evidence_SHA256=sha(dest/'small_evidence.zip'),Git_bundle_SHA256=sha(dest/'published_source.bundle'),
    module_joint_SHA256=report['module_joint']['SHA256'],integration_joint_SHA256=report['integration_joint']['SHA256'],github_commit=publication['commit'])
(base/'outputs/第四租期原流极性强度候选实际接续.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copyfile(base/'outputs/第四租期原流极性强度候选实际接续.json',dest/'continuation.json')
state=base/'outputs/研究接续状态.md'
prefix=f'最新第四租期实际 {clock}：先读《第四租期原流极性强度候选实际接续.json》及polarity_intensity_native_20261008T151107Z/REPORT.md。人类新批24h后，保守Oct9BJ23:00/UTC15:00非平台确认；A/B原Torch2.1cu121 CPU合成模块279/306自然0，集成首次deepcopy图失败340/382完整D保留，结构重建严格加载修复400/457自然0；完整源码argv/UUID/capture/自然exit原ZIP SHA CRC唯一及D联合通过。只有合成代码资格、没有完整编码器或真实TRAIN/新VALTEST/OOF效用控制，原全球gain校准捷径仍在。新2002参数极性强度与同容量回归辅助对照、官方TRAIN/VAL/TEST入口已源码接入，100/4000预算与选模沿用，未新GPU或重复历史。C仅15:16bootstrap非当前fresh。六会话14209/55025/89747/62568/94006/15820明确exit/bye0禁复用；凭据不落盘，无续租/删旧资料。备份C约1.23GB/D.33GB，长期20GB路径仍未回复，完整训练门未过。GitHub33新源码/订单同步main {publication["commit"]}，1751文件全blob/manifest SHA与remote head一致/clean/无secret，完整Gitbundle与证据D保存。既有官方VAL/TEST五项不变，没有五项胜，原10min不改，整体研究/正式超过/全数据五折仍未完成。以下历史。\n\n'
state.write_text(prefix+state.read_text(encoding='utf-8-sig'),encoding='utf-8')
shutil.copyfile(state,dest/'research_state.md');assert sha(state)==sha(dest/'research_state.md')
print(json.dumps(receipt,ensure_ascii=False))
