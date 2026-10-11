"""Metadata/source-only preparation. Never imports a model or opens prediction/data arrays."""
import ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path
b=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');old=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen');o=b/'outputs'
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ');root=b/'work'/('official_benchmark_provenance_preparation_'+stamp);root.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
protocol=old/'outputs/repeat5_experiments/completed/careflow_seed128/protocol.json';p=read(protocol)
assert sha(protocol)=='8b094cc669f36966f8026a65b4a8ab6c5730a604ec5a1aacde4660f49eaea00d'
official=old/'outputs/careflow_reproduction/official';runner=old/'outputs/careflow_reproduction/run_careflow.py'
checks={}
for name,h in p['source_sha256'].items():
 path=official/name
 if path.is_file():checks[name]={'actual_sha256':sha(path),'expected_sha256':h,'match':sha(path)==h}
assert checks['train_reflow_new.py']['match'] and all(x['match'] for x in checks.values())
assert sha(runner)==p['runner_sha256'];ast.parse(runner.read_text(encoding='utf-8'));ast.parse((official/'train_reflow_new.py').read_text(encoding='utf-8'))
inventory={'status':'LOCAL_SOURCE_AND_PROTOCOL_PROVENANCE_PREPARATION_NOT_FORMAL_EXECUTION_FREEZE','actual_utc':now.isoformat(),'protocol_sha256':sha(protocol),'baseline_runner_sha256':sha(runner),'baseline_training_source_sha256':checks['train_reflow_new.py']['actual_sha256'],'archived_metric_source_sha256':'d88739d1d1a224a2f159f29fa723cc24c347eadcf921e93bf57bd0e2cdaa3cb7','training_and_metric_versions_are_distinct':True,'source_files_checked':checks,'baseline_declared_arguments':p['arguments'],'train_rows':1281,'DEV_rows':229,'baseline_declared_trainable_parameters':p['trainable_parameters'],'baseline_drop_last':True,'baseline_batches_per_epoch':1281//32,'baseline_updates_100_epochs':100*(1281//32),'baseline_nominal_rows_per_epoch':32*(1281//32),'selection':'mean of DEV batch MSE with fixed batch128, strict improvement and earliest tie','baseline_existing_wrapper_TEST_access':'only after training and validation-only checkpoint selection; already executed historically; no arrays opened now','baseline_complete_checkpoint_local_presence':(protocol.parent/'best.pt').is_file(),'cached_artifacts_reused_for_formal_benchmark':False,'method_selection_or_201_readout_choice':False,'no_pickle_or_NPZ_or_TEST_labels_read':True,'Torch_GPU_import_or_execution':False,'formal_training_or_final_TEST_authorized_by_this_inventory':False,'unresolved_before_formal_execution':['one method fixed independently of201 output ranking; no new201 threshold or readout choice','precise common officialTRAIN/DEV roles, shared assets and row-ID guards without constructing TEST loader','same documented epoch/update/drop_last/selection policy or explicit nonmatched deviations','public backbone verification and each random task initialization; equal seed is not identical initial state across architectures','complete source/runtime/order/scheduler freeze and actual GPU peak/time/save-space validation','eligible full baseline artifact source and selected whole checkpoint availability; no weak-baseline selection from DEV','final paired model/prediction physical locks and one final evaluation protocol; historical baseline TEST access must be disclosed']}
write(root/'source_provenance_inventory.json',inventory);shutil.copy2(protocol,root/'original_cached_baseline_protocol.json');shutil.copy2(runner,root/'original_cached_baseline_runner.py');shutil.copy2(official/'train_reflow_new.py',root/'original_cached_baseline_training_source.py')
doc='''# 正式 CaReFlow 比较：来源与预算本地准备

这份材料只核查源码和原协议，不是新训练、模型选择、201重评分或最终TEST授权。

既有指定baseline采用TRAIN1281/DEV229、100轮、batch32、drop_last=True，因此每轮40批、4000次更新；DEV选择是batch128的批MSE算术均值。当前新fold参考只有FIT695/INNER153、2200次更新和视频等权选择。两者不能直接称相同训练预算或选模协议；新fullTRAIN协议必须明确并统一这些规则，不能把fold模型当正式fullTRAIN模型。

原baseline训练源SHA d0266a55931fbae4329c2123b3b0f0d475799f31b8e202fa393ecef406ce841e；旧五指标归档时使用另一已固定指标源SHA d88739d1d1a224a2f159f29fa723cc24c347eadcf921e93bf57bd0e2cdaa3cb7。原runner来源与protocol pin匹配；不把两份版本冒同源码。缓存不是作者CLI全默认，也不冒论文同配方。

缓存的原runner在训练及DEV选模完成后曾执行一次TEST；本轮未打开任何预测NPZ、pickle或TEST标签。主聊天未读不抹去历史TEST访问。现有本地缓存目录没有best.pt，不能只凭protocol/预测缓存冒拥有可完整重放的baseline checkpoint。未来可核实永久原件位置；没有合格整模型就保持未通过，不因某个低DEV结果选baseline。

下一准备应固定单方法（不根据201九读出挑赢家）、共同官方角色/数据尺度/预训练资产/预算/选模规则、实际完整源与安全标签守卫。保持TEST不可访问；先做实际资源与保存门。不同架构即使同seed也不是共享全部初始权重，容量和训练成本须如实列出。当前未选择常数/U/W作为部署方法、未启动fullTRAIN/MOSEI或重新跑baseline。

第16建议turn在remote compact阶段因usage limit实际failed，只有部分commentary，无完整报告或独立源审阅。主线独立准备继续；不重复发送同批或把部分建议冒完成，不把该失败当GPU/服务器故障。
'''
(root/'provenance_and_budget_preparation.md').write_text(doc,encoding='utf-8');shutil.copy2(root/'provenance_and_budget_preparation.md',o/'正式CaReFlow比较来源与预算本地准备接续.md');write(o/'正式CaReFlow比较来源与预算本地准备接续.json',inventory)
ledger=read(o/'研究建议交流接续.json');batch=ledger['sixteenth_actual_batch'];batch.update(status='FAILED_USAGE_LIMIT_NO_COMPLETE_SIXTEENTH_REPORT',actual_observed_utc=now.isoformat(),cursor='c9fdcd3f-7d4b-4678-809d-0742a5ab961e:9',complete_report_reviewed=False,partial_commentary_not_full_report=True,new_advice_read_or_adopted=False,error='Error running remote compact task: usage limit',retry_sent=False)
ledger['updated_at_utc']=now.isoformat();ledger['official_benchmark_source_preparation']={'local':str(root),'status':inventory['status'],'real_fit_or_score':False};write(o/'研究建议交流接续.json',ledger)
state=read(o/'完整流匹配U_W头实际接续.json');state['advisor_batch']=batch;state['formal_benchmark_preparation']=str(root);write(o/'完整流匹配U_W头实际接续.json',state)
short=o/'研究接续状态.md';short.write_text(f'准备UTC {now.isoformat()}：正式CaReFlow来源/预算仅本地核查，baseline TRAIN1281 drop_last100×40=4000更新与当前fold2200/videoequal选模不同，须独立全冻共同协议；无新模型/标签/成绩。第16建议remote compact usage-limit失败，只有部分commentary，无完整审阅，不当GPU故障、不重复发同批。先读正式CaReFlow比较来源与预算本地准备接续.md/json；201原结果不变，五项目标与租期保存未完成。\n\n'+short.read_text(encoding='utf-8'),encoding='utf-8')
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/('official_benchmark_provenance_preparation_'+stamp);D.mkdir();shutil.copytree(root,D/'source_and_protocol')
for n in ['正式CaReFlow比较来源与预算本地准备接续.md','正式CaReFlow比较来源与预算本地准备接续.json','研究建议交流接续.json','完整流匹配U_W头实际接续.json','研究接续状态.md']:shutil.copy2(o/n,D/n)
shutil.copy2(__file__,D/Path(__file__).name)
members={str(p.relative_to(D)).replace('\\','/'):sha(p) for p in D.rglob('*') if p.is_file()};write(D/'member_SHA.json',members)
with zipfile.ZipFile(D/'snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in [*members,'member_SHA.json']:z.write(D/n,n)
with zipfile.ZipFile(D/'snapshot.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)+1
 for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
write(D/'actual_local_archive_audit.json',{'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'zip_sha256':sha(D/'snapshot.zip'),'members':len(members)+1,'source_only_not_remote_capture_or_new_score':True})
print(json.dumps({'local':str(root),'D':str(D),'runner_sha':sha(runner),'baseline_100_updates':4000,'source_only':True},ensure_ascii=False))
