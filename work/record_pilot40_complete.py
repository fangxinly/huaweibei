import json,hashlib,sys,shutil,zipfile
from pathlib import Path
root=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');out=root/'outputs'
base=Path('D:/CodexBackups/selective_flow_20261003_1105/anchored_flow_pilot40_complete_actual_20261007T103742Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
clock=sys.argv[1]
r=json.loads((base/'A_original_capsule/run/out/actual_stage_receipt.json').read_text())
natural=json.loads((base/'A_original_capsule/execution/natural_exit.json').read_text())
history=json.loads((base/'A_original_capsule/run/out/history.json').read_text())
joint=json.loads((base/'actual_D_CPU_joint.json').read_text())
assert joint['status']=='PILOT40_D_ORIGINAL_CPU_CAPTURE_COMPLETE' and natural['natural_exit']==0
cpu=json.loads((base/'B_original_capsule/run/out/actual_stage_receipt.json').read_text())
sc=r['selected_INNER_five_development_only']
curve=[{k:h[k] for k in ['epoch','FIT_parts_batch_means','INNER_selection_MSE','INNER_five_development_only','flow_gain_tanh']} for h in history if h['epoch'] in [1,10,20,30,40]]
record={'status':joint['status'],'actualclock_record_UTC':clock,'D':str(base),
 'original_actual_exit_UTC':natural['actual_exit_utc'],'GPU_child_pid':natural['pid'],
 'epochs':r['epochs'],'optimizer_steps':r['optimizer_steps'],'best_epoch':r['best_epoch'],
 'selected_INNER_five_development_only':sc,'CPU_original_receipt':cpu,
 'joint_sha256':sha(base/'actual_D_CPU_joint.json'),'complete_checkpoint':r['complete_resume_and_selected'],
 'curve_observations':curve,'OUTER_scored':False,'full_fivefold_complete':False,
 'same_scope_CaReFlow_pilot_comparison':False,'proved_better_than_old0_598':False,
 'overall_goal_complete':False,'current_frozen_source_or_coefficients_changed':False,
 'closed_interactive_sessions_explicit_zero':[77620,87092,14458,55993],
 'next_dynamic_saves_UTC':['11:30','13:00'],'not_platform_lease_confirmation':True,
 'historical_alignment_report':dict(path=str(out/'TEST与既有0.59结果对齐分析.json'),sha256=sha(out/'TEST与既有0.59结果对齐分析.json'))}
target=out/'新方案40轮实际完成与完整保存接续.json'
target.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
md=f'''新方案40轮真实完成：A child{natural['pid']} 于{natural['actual_exit_utc']}自然exit0，40轮1880更新。按预定INNER全行MSE严格最小、平局最早，选择第20轮。

| 固定选中模型 INNER264 | Acc7 % | Acc2 % | F1 % | MAE | Corr |
|---|---:|---:|---:|---:|---:|
| anchored_flow 第20轮 | {sc['Acc7']*100:.6f} | {sc['Acc2']*100:.6f} | {sc['F1']*100:.6f} | {sc['MAE']:.9f} | {sc['Corr']:.9f} |

这是该折的选模开发集成绩，尚未证明优于旧0.598，也没有同范围CaReFlow对照。OUTER437预测已冻结且仍未评分，不是完整五折完成。

训练曲线提供新的实际诊断：第20→40轮，FIT主预测batch均值MSE从0.409323降到0.070148、MAE从0.492499降到0.199722，而INNER MSE从0.779180变为0.810987、Acc2从86.4%降到84.0%。这支持后半程训练拟合继续增强而开发收益停滞/部分指标退化，与泛化差距扩大一致；FIT为训练时batch均值，不是同一checkpoint eval纯TRAIN风险，不能证明唯一过拟合原因。第40轮MAE0.671141略低但MSE更差，仍保留预定第20轮，不能按指标拼checkpoint。

第20轮tanh(gain)=-0.795143，表明当前修正沿流差方向反向外推，并非正向凸插值；tanh仅限制单标量，不限制预测差幅度，不提供风险保证。现有history没有同checkpoint流前/流后INNER单独风险，不冒称已证明流修正有益或有害。

完整2966458931字节model/347Adam两矩/scheduler/四RNG/订单/FIT统计/40轮history/selected state已D与B原CPU审核。B child{cpu['pid']}自然0，1880步、独立strictmin最早选择误差{cpu['selection_max_error']}，选中预测及全部五项一致，OUTER dummy不变。CPU只核整Torch原件与保存数组，没有CPU模型前向。A94成员真实capture、B真实capture与源/argv/PID/receipt、D整SHA/ZIPCRC/唯一联结通过。

D：{base}。joint SHA {record['joint_sha256']}。

下一优先核对旧0.598冻结骨干/仅供体与头训练同新全量微调的差异；保留旧完整模型资格和正向DEV信号。不能在无同范围对照时声称新loss已成功。新的受控训练应固定相同划分、初始化、预算、选模，先隔离冻结范围或流修正的一项变化，预先声明五项判据；当前不扩矩阵或延长已完成pilot。十模型全五折完整保存容量与整队列+至少2h保存余量尚未落实。原TEST685已一次评分，F五项均输；不按TEST改结构/系数。11:30/13:00UTC仍需真实动态保存，不回填时间。

实际本地记录clock：{clock}。源、旧失败和此前第10轮整快照均保留。
'''
(out/'新方案40轮实际完成与完整保存接续.md').write_text(md,encoding='utf-8')
old=out/'新方案40轮实际训练接续.json'
prior=json.loads(old.read_text(encoding='utf-8'))
shutil.copy2(old,base/'prior_running_continuation_original.json')
prior.update(status=joint['status'],actualclock_local_record_utc=clock,actual_completed_epochs=40,
 actual_step_prefix=1880,natural_exit_exists_at_observation=True,completion_record=str(target),completion_record_sha256=sha(target),detached_training_continues=False,
 actual_last_live_observation_utc='2026-10-07T10:37:16.473966+00:00',actual_training_exit_utc=natural['actual_exit_utc'],
 latest_progress_in_development_not_final_score=history[-1],selected_checkpoint_INNER_development_five=sc,
 complete_D_other_CPU_joint_sha256=record['joint_sha256'],actual_D_free_bytes=shutil.disk_usage('D:/').free,
 closed_interactive_session_ids_no_reuse=[77620,87092,14458,55993])
old.write_text(json.dumps(prior,ensure_ascii=False,indent=2),encoding='utf-8')
state=out/'研究接续状态.md';prior_text=state.read_text(encoding='utf-8')
head=f'最新完整实测{clock}：新anchored_flow固定第0折40轮/1880更新A4695自然0，INNER预定MSE选20，五项48.106061/86.4/86.386016/.679452407/.818676823；D整2966458931B+异节点B原CPU{cpu["pid"]}自然0/347Adam1880步/40历史/选择/数组/双真实capture联合完成。先读《新方案40轮实际完成与完整保存接续.md/json》《TEST与既有0.59结果对齐分析.md/json》。旧A/B/C2完整DEV229约.598不可抹去，旧冻结骨干只供体/头不同于新全量F；新固定F TEST.6436998/C.6195353五项负结果保留，无重评分/TEST调参。新pilot未证明改善/无同范围CaReFlow/OUTER未评分/完整五折未完成，不重跑扩矩阵。四会话77620/87092/14458/55993全exit/bye0关闭禁复用。09:30真实动态保存已完成，11:30/13:00仍待，保守13:30非平台确认；整体目标未完成。以下均历史。\n\n'
state.write_text(head+prior_text,encoding='utf-8')
payload=base/'completion_local_seal';payload.mkdir()
for n in ['新方案40轮实际完成与完整保存接续.md','新方案40轮实际完成与完整保存接续.json','新方案40轮实际训练接续.json','研究接续状态.md','TEST与既有0.59结果对齐分析.md','TEST与既有0.59结果对齐分析.json','保留旧0.598信号的下一项受控训练方向.md']:
 shutil.copy2(out/n,payload/n)
shutil.copy2(Path(__file__),payload/Path(__file__).name)
shutil.copy2(root/'work/audit_pilot40_completion.py',payload/'audit_pilot40_completion.py')
refs={p.name:sha(p) for p in base.iterdir() if p.is_file() and p.suffix!='.pt'}
refs['whole_full_state_sha_ref']=r['complete_resume_and_selected']['sha256']
(payload/'original_evidence_refs.json').write_text(json.dumps(refs,indent=2),encoding='utf-8')
manifest={p.name:sha(p) for p in payload.iterdir()}
(payload/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
archive=base/'completion_local_seal.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(payload.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
 for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
print(json.dumps({'status':joint['status'],'local_seal_sha256':sha(archive),'completion_record_sha256':sha(target)},ensure_ascii=False))
