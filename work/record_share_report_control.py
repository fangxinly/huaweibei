import hashlib,json,shutil,sys
from pathlib import Path
clock=sys.argv[1]
base=Path('D:/CodexBackups/selective_flow_20261003_1105')
dest=base/'share_research_report_20261007T131648Z'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=Path('outputs/多模态流与CaReFlow实验总结报告_20261007.md')
archive=Path('outputs/多模态流与CaReFlow总结报告及原始指标_20261007.zip')
dynamic=base/'lease_dynamic_actual_20261007T130645Z'
control=dict(actualclock_record_UTC=clock,status='FORWARDABLE_REPORT_SAVED_D_AND_ZIP_VERIFIED',
    report_path=str(report.resolve()),report_SHA=sha(report),report_ZIP=str(archive.resolve()),report_ZIP_SHA=sha(archive),
    dynamic_joint_SHA=sha(dynamic/'actual_D_dynamic_joint.json'),
    SSH_exit0_observed=[4264,78343,86833],SFTP_bye_exit0_observed=[17081,84814,86260],
    stale_SFTP_exit1_preserved=[73248,27032,54572],all_closed_session_ids_forbidden_to_reuse=True,
    new_TEST_prediction_or_scoring=False,new_training=False,new_runtime_Torch_verification=False,
    full_fivefold_and_five_metric_superiority_complete=False,original_automation_unchanged=True)
text=json.dumps(control,ensure_ascii=False,indent=2)
(dest/'actual_report_control.json').write_text(text,encoding='utf8')
Path('outputs/总结报告交付实际接续.json').write_text(text,encoding='utf8')
shutil.copy2(__file__,dest/Path(__file__).name)
for p in [Path('work/seal_lease1300_completed_refs.py'),Path('outputs/第二租期13点真实动态保存实际接续.md')]:shutil.copy2(p,dynamic/p.name)
status=Path('outputs/研究接续状态.md');old=status.read_text(encoding='utf8')
status.write_text(
    f'最新报告与保存 {clock}：先读《多模态流与CaReFlow实验总结报告_20261007.md》《总结报告交付实际接续.json》《第二租期13点真实动态保存实际接续.md/json》。全部固定VAL/TEST已完成，F TEST MAE.643699787/C.619535294五项负结果保留；旧约.598仅VAL，new20原TEST.544225混合FIT/INNER不能冒保留TEST。新报告与原未舍入JSON/candidate源码ZIP全SHA/CRC/唯一完整D封存，可人工转发；未发外部联系人。原标签强弱比TRAIN弱40.20297%/VAL36.68122%/TEST37.22628%，原100共同订单实际弱51460/128000暴露40.203125%，不是独立样本；无重评分。新弱损失仅源码，原Torch验证与训练未完成，路径升级仅方法设计。\n\n'
    '13:00计划动态采集实际13:07，A/B/C完整小ZIP D SHA/CRC/唯一/source/fullargv/UUID/公共runtime资产均过，原整模型fresh SHA联结已存D，不回填13:00/重下旧大件。三个旧待认证SFTP实际关闭1保留，fresh三连接成功下载，六新SSH/SFTP全exit/bye0禁复用。平台到期未核，保守13:30。D约2.45GB/无新训练+2h保存预算，不删旧件。原10min不改，无subagents/新自动任务/续租/释放/关机。整体五项超过/完整五折仍未完成。以下全部历史。\n\n'+old,encoding='utf8')
shutil.copy2(status,dest/'研究接续状态_交付时原件.md')
print(json.dumps(dict(status=control['status'],control_SHA=sha(dest/'actual_report_control.json')),ensure_ascii=False))
