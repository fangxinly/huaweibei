import datetime,hashlib,json,pathlib,shutil,zipfile

base=pathlib.Path(__file__).resolve().parents[1]
D=pathlib.Path('D:/CodexBackups/selective_flow_20261003_1105/paired_fulltrain100_complete_actual_20261007T030311Z')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
score=D/'paired_fixed_selected_DEV_five_20261007T034922Z';result=read(score/'result.json');ex=read(score/'natural_exit.json');launch=read(score/'actual_child_launch.json')
assert ex['natural_exit'] and ex['exit_code']==0 and ex['result_SHA']==sha(score/'result.json') and ex['child_pid']==launch['pid'] and ex['full_argv']==launch['full_argv']
assert result['argv']==ex['full_argv'][1:] and not result['all_five_strict_on_already_explored_DEV'] and not any(result['strict_improvement_each'].values())
assert sha(score/'plan.json')==result['frozen_plan_sha256']=='a23ba23e62b49534b739e77d0eeee8a0ae7c9582868dceadabcea9c840b089cd'
methods={};files={};large={}
for m in ('minimal_fixed_F','careflow'):
 root=D/m;t=root/'actual_training_joint_after_history_serialization_repair.json';f=root/'actual_fresh_public_selected_saved_joint.json';tr,fr=read(t),read(f)
 assert fr['whole_training_joint_sha256']==sha(t) and fr['all_original_model_and_array_gates_passed']
 gpu=read(root/'a/run/out/actual_stage_receipt.json');cpu=read(root/'b/run/out/actual_stage_receipt.json');fresh=read(root/'fresh/run/out/actual_stage_receipt.json')
 assert cpu['checks']['arrays_history_orders']['independent_DEV_batch_MSE_max_error']==0 and cpu['checks']['resume']['steps']==4000
 assert fresh['original_prediction_replay_error']==fresh['dummy0vs7_error']==0 and fresh['state_and_allRNG_unchanged']
 assert not fresh['new_label_or_final_TEST_access'] and not cpu['final_TEST_access'] and not cpu['CPU_model_forward']
 methods[m]={'training_joint_SHA':sha(t),'fresh_complete_joint_SHA':sha(f),'best_epoch':gpu['metadata']['best_epoch'],'best_state_SHA':gpu['metadata']['best_state_SHA'],'CPU_child_pid':read(root/'b/run/natural_exit.json')['child_pid'],'CPU_actual_natural_exit_UTC':read(root/'b/run/natural_exit.json')['actual_utc'],'CPU_history_serialization_supplement_SHA':tr['CPU_history_repair_plan_sha256'],'CPU_state_Adam_counts':cpu['checks']['resume'],'CPU_no_forward':True,'fresh_child_pid':read(root/'fresh/run/natural_exit.json')['child_pid'],'fresh_actual_natural_exit_UTC':read(root/'fresh/run/natural_exit.json')['actual_utc'],'fresh_replay_error':0,'dummy0vs7_error':0,'state_and_allRNG_unchanged':True,'fresh_cumulative_peak_no_reset':fresh['cumulative_peak_no_reset'],'whole_D_files':tr['original_complete_checkpoints_D'],'fixed_DEV_five':result['rows'][m]['all_five_same_fixed_prediction']}
 for role in ('a','b','fresh'):
  for n in ('capture_receipt.json','capture_actual_exit.json','snapshot.zip'):
   p=root/role/n;files[str(p.relative_to(D)).replace('\\','/')]=p
 for n in ('actual_training_joint_after_history_serialization_repair.json','actual_fresh_public_selected_saved_joint.json','actual_D_GPU_whole_SHA_CRC_source_capture_receipt.json'):
  p=root/n;files[str(p.relative_to(D)).replace('\\','/')]=p
 for n in ('original_operator_newline_gate_failure.json','original_operator_fresh_wrapper_launch.json','original_operator_fresh_gate.json'):
  p=root/'fresh'/n;files[str(p.relative_to(D)).replace('\\','/')]=p
 for key,rec in tr['original_complete_checkpoints_D'].items():
  p=root/'a/run/out'/(key+'.pt');assert p.is_file() and p.stat().st_size==rec['bytes']
  large[str(p.relative_to(D)).replace('\\','/')]=dict(rec,path=str(p),physically_D_full_saved=True,SHA_CRC_already_checked_by_training_joint_not_new_download=True)
for sub in (score,D/'paired_CPU_history_serialization_repair_20261007T034000Z',D/'CPU_history_failure_original_20261007T033805Z'):
 for p in sorted(sub.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(D)).replace('\\','/')]=p
for n in ('actual_original_training_cost_and_fixed_selection_only.json','actual_both_training_completion_progress.json','actual_original_CPU_history_failures_and_supplemental_pending.json'):
 p=D/n;files[n]=p
record={'status':'ACTUAL_BOTH_FULLTRAIN100_ORIGINAL_GPU_D_OTHER_CPU_FRESH_COMPLETE_FIXED_DEV_ALL_FIVE_NEGATIVE_NO_FINAL_TEST','actual_local_record_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'D':str(D),'training_plan_SHA':'8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb','CPU_history_supplement_plan_SHA':'c4f6485b43d9045f0222ddb3e2206c4a25cada3d753f3142ec724c4e1e6e92fc','fixed_DEV_plan_SHA':result['frozen_plan_sha256'],'fixed_DEV_result_SHA':sha(score/'result.json'),'fixed_DEV_child_pid':ex['child_pid'],'fixed_DEV_actual_natural_exit_UTC':ex['actual_utc'],'methods':methods,'F_minus_C_same_DEV':result['F_minus_C_same_DEV'],'all_five_strict_on_already_explored_DEV':False,'fullTRAIN100_preservation_current_two_models_complete':True,'overall_research_formal_TEST_superiority_or_all_lease_preservation_complete':False,'prior_original_CPU_natural1_failures_preserved':True,'history_repair_only_comparison_view_dropped_TRAIN_rows_tuple_to_list':True,'GPU_training_or_original_model_changed':False,'operator_newline_failure_before_fresh_root_or_child_preserved':True,'no_TEST_or_201_rescore':True,'closed_ids_no_reuse':[83561,29114,36457,17139,42147,58037,93385,14448,72143,34370,11741,47872,36738,6701,67854,62809],'current_all_new_interactive_sessions_actual_exit_or_bye_zero':True,'no_active_remote_sessions_in_this_batch':True,'conservative_lease_end_UTC':'2026-10-07T13:30:00+00:00','not_platform_lease_confirmation':True,'strong_actual_save_windows_UTC_pending':['09:30','11:30','13:00'],'next':'Stop current minimal_fixed_F version expansion/readout/seed rescue on explored DEV; independently review fixed matched five-metric negative and cost. Keep the two selected complete models immutable. No final TEST until separate fixed source/protocol/role/ID/asset/runtime/once-only metric rule is frozen; disclose old TEST access. No formal all-five claim. Lease dynamic saves remain required at09:30/11:30/13:00 actual UTC, fresh native identity/source/fullargv/compute/space, receipt-first D evidence. Advisor16 still failed, no same-batch resend or poll without recovery evidence.'}
p=D/'actual_both_fulltrain100_complete_CPU_fresh_fixed_DEV_summary.json';assert not p.exists();p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');files[p.name]=p
lines=['双方新fullTRAIN100完整保存及固定best一次DEV五项实际结果','',f"实际评分子进程 {ex['child_pid']} UTC {ex['actual_utc']} 自然0。双方先完成GPU原训练/整selected及resume D保存/异节点CPU/真实fresh重放capture与联合门，再固定评分来源和两方父SHA，一次全五项同时评分，不选择新checkpoint或读出。",'', '| 固定模型 | epoch | Acc7↑ % | Acc2↑ % | F1↑ % | MAE↓ | Corr↑ |','|---|---:|---:|---:|---:|---:|---:|']
for m in ('minimal_fixed_F','careflow'):
 v=methods[m]['fixed_DEV_five'];lines.append(f"| {m} | {methods[m]['best_epoch']} | {100*v['Acc7']:.6f} | {100*v['Acc2']:.6f} | {100*v['F1']:.6f} | {v['MAE']:.12f} | {v['Corr']:.12f} |")
lines+=['','F在全部五项点值上均差于本次新CaReFlow；当前版本未达到五项全面超过。此DEV229已探索并用于预固定batch MSE选模，Acc2/F1排除13个0真值后216行，不是独立确认或论文TEST。不同架构的参数与实耗仍不相同；名义共同seed128/100轮/batch32/drop1/4000实际更新/共同订单/同批MSE严格earliest选模。旧弱缓存不能取代本次指定CaReFlow。','', '全部7415841507字节完整模型在D+异节点B原CPU字节/SHA/ZIP CRC/唯一通过。F CPU child31944/C31863自然0，374/329模型state、364/323 Adam状态及各4000 step、scheduler、全部RNG、原100轮数组/预测冻结先于label/订单核过，独立DEV选择MSE误差0。不是CPU模型前向。原CPU31642/31714自然1是JSON list/Torch tuple尾批字段比较的表示错误，原失败与严格只修此比较视图的补充源/协议已完整保存；没有删改训练/旧冻结源/模型。','', 'F fresh child3173/C1162自然0；独立公共实例整selected重放、dummy0/7、自己的统计及全部参数/RNG不变均过。各38成员真实capture，CPU C208/F206成员，GPU各156成员；actual UTC和完整argv均由原件关联，不用目录后缀当实际时间。后续静态本地总封装不是新的remote capture。','', '本轮Windows文本读取消去CRLF导致回执传递SHA门失败，发生在fresh根/模型子进程创建前，改为原始字节传递且原失败保留。所有本批SSH/SFTP均明确exit/bye实际0关闭，ID禁复用。','', '停止当前minimal_fixed_F版本基于该DEV的seed/feature/readout救分或扩矩阵；仅独立审阅反证与合理后续成本，不外推否定所有flow/消息/残差。最终TEST流程尚未冻结/执行，历史旧baseline TEST访问不能抹去；正式五项目标未完成。09:30/11:30/13:00 UTC三机动态强化保存仍待实际执行；租期保守13:30非平台确认。', '',f'D原件：{D}',f'评分plan SHA {result["frozen_plan_sha256"]}',f'评分result SHA {sha(score/"result.json")}',f'全记录 SHA {sha(p)}']
report=D/'actual_both_fulltrain100_complete_CPU_fresh_fixed_DEV_summary.md';assert not report.exists();report.write_text('\n'.join(lines)+'\n',encoding='utf-8');files[report.name]=report
manifest={'actual_local_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'LOCAL_COMPLETE_PARENT_PROVENANCE_SEAL_NOT_NEW_REMOTE_CAPTURE','members':{n:{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in files.items()},'large_complete_D_refs_not_new_downloads':large}
mp=D/'actual_both_fulltrain100_complete_local_seal_manifest.json';assert not mp.exists();mp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
archive=D/'actual_both_fulltrain100_complete_CPU_fresh_fixed_DEV_seal.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for n,p in files.items():z.write(p,n)
 z.write(mp,mp.name)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(files)+1
 for n,rec in manifest['members'].items():assert hashlib.sha256(z.read(n)).hexdigest()==rec['sha256']
seal={'actual_local_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':manifest['kind'],'ZIP':str(archive),'SHA':sha(archive),'bytes':archive.stat().st_size,'members':len(files)+1,'full_CRC_unique_all_member_SHA_passed':True,'whole_models_refs_bytes':sum(v['bytes'] for v in large.values()),'D_free_bytes':shutil.disk_usage(D).free,'summary_SHA':sha(p),'no_new_remote_capture_claim':True}
assert seal['D_free_bytes']>=6*1024**3
sp=D/'actual_both_fulltrain100_complete_local_seal_receipt.json';assert not sp.exists();sp.write_text(json.dumps(seal,indent=2)+'\n',encoding='utf-8')
for n in ('正式双方fullTRAIN100固定best一次DEV五项实际结果','正式双方fullTRAIN100完成与完整保存实际结果'):
 (base/'outputs'/(n+'.md')).write_bytes(report.read_bytes());(base/'outputs'/(n+'.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
record['local_complete_seal_receipt']=str(sp);record['local_complete_seal_receipt_SHA']=sha(sp)
(base/'outputs/正式双方fullTRAIN100整保存与CPU待接续.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
training=base/'outputs/正式双方官方fullTRAIN100实际训练接续.json';old=read(training);old['status']=record['status'];old['latest_complete_CPU_fresh_and_DEV_actual_summary']=str(p);old['latest_complete_CPU_fresh_and_DEV_actual_summary_SHA']=sha(p);old['latest_complete_local_seal']=seal;training.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state=base/'outputs/研究接续状态.md';oldtext=state.read_text(encoding='utf-8');head='最新UTC03:50实际：双方新fullTRAIN100/4000训练、全部整selected/resume7415841507字节D+异节点CPU+独立fresh整重放+全部原capture联合门已过。F补充CPU31944/C31863 natural0，旧31642/31714自然1完整保留，仅tuple/list dropped_TRAIN_rows比较视图另冻补充plan c4f6485b...，未改训练/权重/旧源；原100轮选择复算误差0。fresh F3173/C1162 natural0，预测重放/dummy/全参数RNG/stat误差0；真实capture已D，所有本批会话exit/bye0关闭禁复用。\n固定Fbest89/Cbest93 DEV一次5项 child28316 UTC03:50:17.953086自然0：F=.49344978165938863/.875/.8742439348620805/.6210114891238218/.8575564162961984；C=.4978165938864629/.8796296296296297/.8792475014697237/.6045238262758688/.8594938469285841。F五项均差，当前版本未过全面超过，停此版本DEV救分/扩容，不否定全部flow/消息。新一次评分plan a23ba23e.../result6e7e0d62...已D全source/父joint/argv/natural0。DEV已探索/已选模非独立TEST，禁旧弱缓存替换新C。\n先读《正式双方fullTRAIN100整保存与CPU待接续.json》《正式双方fullTRAIN100固定best一次DEV五项实际结果.md/json》。完整当前两模型保存已完成，研究/正式TEST五项超过及租期09:30/11:30/13:00实际强化保存仍未完成。第16建议failed未恢复，不重发同批/轮询。以下全部为历史记录，旧pending/训练局部状态不得冒当前。\n\n';state.write_text(head+oldtext,encoding='utf-8')
print(json.dumps({'seal':seal,'status':record['status'],'report':str(report),'five':record['F_minus_C_same_DEV']},ensure_ascii=False))
