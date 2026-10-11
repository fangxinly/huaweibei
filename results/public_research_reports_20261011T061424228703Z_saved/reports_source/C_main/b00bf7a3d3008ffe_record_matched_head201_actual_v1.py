import datetime,hashlib,json,shutil,zipfile
from pathlib import Path
base=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');out=base/'outputs'
D=Path('D:/CodexBackups/selective_flow_20261003_1105/matched_head_score201_actual_20261006T200020Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,x:Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
j=read(D/'complete_matched_head201_scores_D_B_CPU_joint.json');r=read(D/'a/run/out/all_nine_development_results.json');cpu=read(D/'b/run/cpu_original_receipt.json');receipt=read(D/'a/run/out/actual_matched_head_evaluation_receipt.json')
assert j['status']=='ACTUAL_HEADEVAL201_ALL_NINE_SCORES_D_OTHER_CPU_JOINT_PASSED_DEVELOPMENT_ONLY'
assert len(r['outputs'])==9 and not r['readout_or_model_selection'] and len(receipt['task_label_journal'])==1
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
record={'status':j['status'],'actual_record_utc':now,'D':str(D),'joint_sha256':sha(D/'complete_matched_head201_scores_D_B_CPU_joint.json'),'plan_sha256':j['plan_sha256'],'head_state_sha256':j['head_state_sha256'],'score_child_pid':receipt['pid'],'actual_score_natural_exit':read(D/'a/run/natural_exit.json')['actual_exit_utc'],'other_CPU_child_pid':cpu['pid'],'CPU_five_metric_max_error':cpu['independent_five_metric_max_error'],'CPU_video_risk_max_error':cpu['independent_video_risk_max_error'],'all_nine':r['outputs'],'matched_comparisons':r['predeclared_matched_comparisons'],'all_five_strictly_better_than_F':{n:all(x['five_metrics_same_prediction'][k]>r['outputs']['F']['five_metrics_same_prediction'][k] for k in ['Acc7','Acc2','F1','Corr']) and x['five_metrics_same_prediction']['MAE']<r['outputs']['F']['five_metrics_same_prediction']['MAE'] for n,x in r['outputs'].items()},'decision':'Stop this weighted-affine head expansion and do not choose201 readout. Retain full fixed reference and zero/constant/matched simple baselines. Future different inputs or official CaReFlow benchmark require independent frozen protocol; no TEST tuning.','development_not_confirmation_or_CaReFlow_benchmark':True,'overall_research_and_lease_preservation_complete':False,'sessions_closed_actual_exit0':{'A_SSH':27749,'A_SFTP':52161,'B_SSH':28761,'B_SFTP':92646}}
write(out/'完整流匹配U_W头201一次开发五指标实际结果.json',record)
lines=['# 匹配 U/W 头：201 行一次开发评价实际结果','',f'记录 UTC {now}。完整冻结协议、232 FIT 真实拟合、201 九预测物理 SHA、D 与异节点 CPU 原件联合门均先于本轮一次标签评价。','', '结论：没有校正同时严格改善五项；W 在相同 free/interval/discrete 三种读出下的视频 MSE 均差于 U。本批加权仿射扩容暂停，不用201挑部署读出。','', '| 固定输出 | Acc7 % | Acc2 % | F1 % | MAE | Corr | 视频等权 MSE |','|---|---:|---:|---:|---:|---:|---:|']
for n,x in r['outputs'].items():
 m=x['five_metrics_same_prediction'];lines.append(f"| {n} | {100*m['Acc7']:.4f} | {100*m['Acc2']:.4f} | {100*m['F1']:.4f} | {m['MAE']:.9f} | {m['Corr']:.9f} | {x['video_equal_MSE']:.9f} |")
lines+=['', '常数校正 MAE 下降，但 Acc2/F1 下降且 Corr 不变；U_free 提高 Acc7/Corr 但 MAE/Acc2/F1 更差。约束读出相对各自 free 降低 MSE，却都未胜常数的视频 MSE；不能据此宣称消息无效或所有残差不可学习。W_discrete 的 MAE 仅比 F 低约0.0000162，Acc2/F1 完全打平，无统计功效或跨seed证据。','', '201/9视频只作一次 development，官方 TRAIN 历史已探索、参考 INNER 已选模；不是全新确认、全流程 crossfit、DEV/TEST 或 CaReFlow benchmark。此表不能与另一划分的 CaReFlow 数字横比。保留原全部九结果，不拼模型指标、不调整 lambda/特征/幅度/阈值或另择读出救分。', '', f"真实评分 child{receipt['pid']} 自然exit0；异节点原CPU child{cpu['pid']} 自然exit0，独立五项最大误差 {cpu['independent_five_metric_max_error']:.3g}，视频风险最大误差 {cpu['independent_video_risk_max_error']:.3g}。A73/B136完整原成员 SHA/CRC/唯一成员及 source/argv/标签顺序联结通过，无新教师/参考更新、CPU模型前向或新weight下载。", '', f"D：{D}；完整joint SHA {record['joint_sha256']}。全部本轮 SSH/SFTP 明确exit/bye actual0，旧ID禁止复用。整体研究和租期保存仍未完成。", '', '后续：先审阅本批固定匹配反证与五项 tradeoff，若推进正式 CaReFlow 比较，独立冻结单方法与单固定baseline完整官方流程和选择规则，不在 TEST 上选结构。租期保存继续按原真实时刻执行。']
(out/'完整流匹配U_W头201一次开发五指标实际结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
state=read(out/'完整流匹配U_W头实际接续.json');state.update(status=j['status'],actual_record_utc=now,predict_completed_or_scores=True,score=record,sessions_closed=True,next='Pause this weighted-affine expansion; no repeats or201 selection. Preserve originals, review evidence and freeze future matched official benchmark separately.')
write(out/'完整流匹配U_W头实际接续.json',state)
ledger=read(out/'研究建议交流接续.json');ledger['latest_matched_head201_actual']=record;write(out/'研究建议交流接续.json',ledger)
p=out/'研究接续状态.md';p.write_text(f"最新实际UTC {now}：匹配U/W232真实FIT→201九预测全SHA先冻→一次开发九输出五指标完成，A/B原CPU及双capture全D联合通过。没有校正同时严格改善五项；W同读出三视频MSE均差于U，暂停此加权仿射扩容，不在201择读出。先读完整流匹配U_W头201一次开发五指标实际结果.md/json、实际接续.json。全部四会话实际exit0，禁复用；整体超过CaReFlow与租期保存未完成。下方旧时刻仅历史非当前状态。\n\n"+p.read_text(encoding='utf-8'),encoding='utf-8')
control=D/'control';control.mkdir(exist_ok=True)
for n in ['完整流匹配U_W头201一次开发五指标实际结果.md','完整流匹配U_W头201一次开发五指标实际结果.json','完整流匹配U_W头实际接续.json','研究建议交流接续.json','研究接续状态.md']:
 shutil.copy2(out/n,control/n)
write(control/'control_original_SHA.json',{p.name:sha(p) for p in control.iterdir() if p.is_file()})
print(json.dumps({k:record[k] for k in ['joint_sha256','CPU_five_metric_max_error','CPU_video_risk_max_error','all_five_strictly_better_than_F']},ensure_ascii=False))
