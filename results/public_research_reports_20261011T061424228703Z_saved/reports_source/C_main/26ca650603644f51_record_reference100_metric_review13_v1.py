import json,hashlib,datetime,shutil,zipfile
from pathlib import Path
R=Path(__file__).parent.parent
D=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference100_complete_actual_20261006T183917Z')
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第十三次独立审视_五指标全面超过的统一协议与功效边界.md')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
joint=read(D/'complete_reference100_GPU_D_B_CPU_joint_audit.json')
assert joint['formal_updates']==2200 and joint['original_CPU_natural_exit']['exit_code']==0
metrics=read(R/'outputs/完整流100唯一best五指标INNER选择用途实际结果.json')
assert metrics['checkpoint_best_epoch']==41 and metrics['original_receipt_metric_error']<1e-12
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
failure=read(D/'b/capture_v1_failed_actual_exit.json')
assert failure['exit_code']==1 and failure['natural_wait_verified']
capture_note={'actual_utc':now,'original_failed_capture':failure,'cause':'v1 allowed staged_reference_ prefix but new root was staged_reference100_cpu_audit; assertion rejected before capture destination creation. Actual traceback at capture_staged_CPU_audit_v1.py line6.','repair':'Separate capture tool v2 additionally permits the exact authorized new CPU root; completed scientific training and old source unchanged.','v2_source_sha256':sha(R/'work/capture_staged_CPU_audit_v2.py'),'actual_new_capture_receipt':read(D/'b/capture_receipt.json'),'actual_new_capture_exit':read(D/'b/capture_actual_exit.json'),'completed_science_or_weights_modified':False}
write(D/'b/capture_failure_and_actual_repair.json',capture_note)
l=read(R/'outputs/研究建议交流接续.json')
decision={'actual_utc':now,'report':str(review),'sha256':sha(review),'full_read':True,
 'accepted':['五指标统一口径与单checkpoint全部Delta为正判定','既有CaReFlow固定配置seed128缓存，非CLI全默认或论文同配方','旧DEV为描述性观察；分类净5/7/5与3/4/3行非显著性证据','锁定100原seed/loss/日程/INNER MSE选择不变','唯一best补五项后，按原delta和完整Z成本门推进小头','零标签候选记录0和clip-round边界覆盖，非收益或扩大幅度授权'],
 'deferred':['最终双方完整官方benchmark协议/共同seed批与最终一次TEST评价','功效/视频组配对区间及跨seed稳定性，当前无有效估计'],
 'rejected':['以旧DEV五项占优宣称全面超过论文','逐指标拼A/B/seed/checkpoint','常数或正仿射能严格提高同pF Corr'],
 'cost':'本轮只读全文与本地记录，无训练/标签新增/远端实验；统计与正式benchmark须另冻协议。',
 'suggestions_are_not_new_measurements':True}
l['review_thread'].update(status='THIRTEENTH_FULL_REVIEW_READ_AND_INDEPENDENTLY_DECIDED',latest_report=str(review),latest_report_sha256=sha(review))
for b in l['sent_batches']:
 if b.get('batch')=='five_metric_DEV_actual_20261006T183811Z':b.update(status='COMPLETE_REPORT_FULLY_READ_AND_CONSIDERED',review_report_sha256=sha(review),independent_decision=decision)
l['five_metric_DEV'].update(current_100_same_best_five_metrics_pending=False,baseline_scope='existing fixed CaReFlow configuration seed128, not all CLI defaults or verified paper recipe')
l['reference100_INNER_five_metrics']={'report':str(R/'outputs/完整流100唯一best五指标INNER选择用途实际结果.md'),'report_sha256':sha(R/'outputs/完整流100唯一best五指标INNER选择用途实际结果.md'),'result':metrics,'scope':'INNER selection only, not official DEV/Test comparison'}
l['review13_independent_decision']=decision;l['updated_at_utc']=now
l['latest_actual_GPU_access_and_continuation']['control_sessions']={'A_SSH36491':'actual_exit0','A_SFTP5536':'actual_bye_exit0','B_SSH21582':'actual_exit0','B_SFTP62714':'actual_bye_exit0','all_older_ids':'do_not_reuse'}
write(R/'outputs/研究建议交流接续.json',l)
p=R/'outputs/研究接续状态.md'
p.write_text('最新五指标UTC '+now+'：旧DEV229三臂各五项胜指定固定CaReFlow配置seed128缓存，仅描述性；新完整流100唯一best41五项已补算，仅INNER153/4选择用途。完整D-B原CPU/双capture总门已过；第13完整审视已读并独立记录，正式全面超过目标未完成。先读完整流100唯一best五指标INNER选择用途实际结果.md/json及CaReFlow五指标统一评价与实际DEV结果.md/json。所有旧会话禁复用，新头/OUTER/最终Test尚未启。\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
note='\n\n第13次独立审视后的来源补充（'+now+'）：这里的对照应称“既有固定CaReFlow配置、seed128缓存”。seed128为作者默认seed，但100epoch/batch32/inter_dim150/loss_f .2/loss_b .1/eps1e-3并非固定旧源CLI全默认；不冒论文同配方。旧缓存训练源与归档指标源分别固定。原数值与原冻结计划不变；正式benchmark须另固定完整训练与选模参数、样本ID和唯一输出，不能拼指标或用TEST改策略。建议为分析，不是新实测。\n'
p=R/'outputs/CaReFlow五指标统一评价与实际DEV结果.md';p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8')
l['five_metric_DEV']['report_sha256']=sha(p);write(R/'outputs/研究建议交流接续.json',l)
dest=D/'metric_review13_final_records';dest.mkdir(exist_ok=False)
files=[review,R/'outputs/研究建议交流接续.json',R/'outputs/研究接续状态.md',p,R/'outputs/完整流100唯一best五指标INNER选择用途实际结果.md',R/'outputs/完整流100唯一best五指标INNER选择用途实际结果.json',Path(__file__)]
for f in files:shutil.copyfile(f,dest/f.name)
write(dest/'independent_review_decision.json',decision)
write(dest/'capture_failure_and_actual_repair.json',capture_note)
mf={f.name:sha(f) for f in dest.iterdir()};write(dest/'manifest.json',{'actual_utc':now,'members':mf,'fresh_C_D':{x:shutil.disk_usage(x+':/').free for x in ['C','D']}})
with zipfile.ZipFile(dest/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in mf:z.write(dest/n,n)
 z.write(dest/'manifest.json','manifest.json')
with zipfile.ZipFile(dest/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in mf.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
print(json.dumps({'status':'ACTUAL_METRICS_AND_REVIEW13_STATE_D_SAVED','actual_utc':now,'ZIP_SHA':sha(dest/'records.zip')}))
