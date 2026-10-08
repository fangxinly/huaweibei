import json,hashlib,datetime,zipfile,shutil
from pathlib import Path
R=Path(__file__).parent.parent
D=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference100_complete_actual_20261006T183917Z')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
a=read(D/'complete100_D_original_audit.json');gpu=read(D/'a/run/out/actual_training_receipt.json');gexit=read(D/'a/run/natural_exit.json')
B=D/'b';c=read(B/'capture_receipt.json');ex=read(B/'capture_actual_exit.json')
assert ex['exit_code']==0 and ex['natural_wait_verified']
assert sha(B/'snapshot.zip')==c['snapshot_sha256'] and (B/'snapshot.zip').stat().st_size==c['snapshot_bytes']
with zipfile.ZipFile(B/'snapshot.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==c['members']
 m=json.loads(z.read('member_manifest.json'))
 for n,s in m['small_members'].items():assert hashlib.sha256(z.read(n)).hexdigest()==s['sha256']
 cpu=json.loads(z.read('audit_root/cpu_original_receipt.json'));ce=json.loads(z.read('audit_root/cpu_original_receipt.actual_exit.json'))
 assert ce['exit_code']==0 and ce['natural_wait_verified'] and ce['child_pid']==cpu['pid'] and ce['child_full_argv'][1:]==cpu['argv']
 assert ce['original_receipt_sha256']==hashlib.sha256(z.read('audit_root/cpu_original_receipt.json')).hexdigest()
 assert cpu['source_sha256']==sha(R/'work/audit_staged_reference100_CPU_v2.py')
 assert cpu['steps']==2200 and cpu['epochs']==100 and cpu['best_epoch']==41 and not cpu['GPU_used'] and not cpu['CPU_model_forward'] and not cpu['OUTER_used']
 assert cpu['original_training_receipt_sha256']==a['training_receipt_sha256'] and cpu['original_natural_exit_sha256']==a['training_natural_exit_sha256']
 for n,r in a['whole'].items():
  assert cpu['states'][n]['file_sha256']==r['sha256'] and cpu['states'][n]['bytes']==r['bytes']
  br=m['large_references_not_downloads']['audit_root/run/out/'+n];assert br['sha256']==r['sha256'] and br['bytes']==r['bytes'] and br['stable_before_after']
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 joint={'status':'ACTUAL_REFERENCE100_GPU_D_B_COMPLETE_STATE_SOURCE_ORDER_ARRAY_CAPTURE_JOINT_PASSED','actual_utc':now,'GPU_natural_exit_utc':gexit['actual_exit_utc'],'GPU_child_pid':gpu['pid'],'epochs':100,'formal_updates':2200,'best_epoch':41,'INNER_video_MSE_selection_only':gpu['INNER_video_equal_MSE_selection_only'],'strict_fresh_full_pt_replay_error':gpu['fresh_instance_strict_full_disk_replay_error'],'dummy_label_replacement_error':gpu['dummy_label_replacement_error'],'original_D_full_files':a['whole'],'original_other_training_node_CPU':cpu,'original_CPU_natural_exit':ce,'A_capture':read(D/'a/capture_receipt.json'),'B_capture':c,'D_audit_sha256':sha(D/'complete100_D_original_audit.json'),'CPU_model_forward':False,'OUTER_or_new_head_used':False,'new_generalization_or_donor_gain_claimed':False,'fresh_space':{k:shutil.disk_usage(k+':/').free for k in ['C','D']}}
 write(D/'complete_reference100_GPU_D_B_CPU_joint_audit.json',joint)
 report='完整流单参考100轮完成与整状态联合保存\n\n'
 report+='GPU child1486 UTC '+gexit['actual_exit_utc']+' 自然exit0，固定100轮2200更新，前10完整model/Adam/scheduler/RNG恢复后继续11–100，未重启或重复前10。全部正常32/尾23均正式更新，已捕获1980续训batch订单和100轮历史原件；B独立CPU审核完整原文件374状态/364参数Adam两矩与step2200/scheduler/所有RNG/全history/best状态、FIT统计、数组、source和订单通过。原CPU child '+str(cpu['pid'])+' UTC '+ce['actual_exit_utc']+' 自然exit0，不是CPU模型前向。\n\n'
 report+='固定INNER153/4视频MSE严格更小选模且平局最早，best第41轮，MSE '+str(gpu['INNER_video_equal_MSE_selection_only'])+' 只选模。自己的整pt同实例/fresh公共实例重放误差0，dummy标签0/7误差0，FIT-only统计未变。没有OUTER/head/CAL/EVAL/DEV/TEST新分数、全流程crossfit或供体收益对照。\n\n'
 report+='本续阶段含全量存盘/重放 '+str(gpu['final_budget']['elapsed_seconds'])+'秒，累计allocated/reserved '+str(gpu['final_budget']['peak_allocated_bytes'])+'/'+str(gpu['final_budget']['peak_reserved_bytes'])+'字节，均低于原6GiB且未重置peak。预定四FIT零标签供体关闭终端变化最大 '+str(gpu['donor_mechanism']['terminal_scalar_change_max'])+'，恢复误差0，只是可达性，不是收益。\n\n'
 report+='完整resume2967047320bytes SHAdd1acc8e83e54fcd533b6fc365239a1e90aaf9ab49843d940fe26da901d627da；best741731206bytes SHAda42330af1d9c7eb39f58744949adda0af53c62c77b367e1f302d6b1d4ab3f62。真实原件已D下载并转存B，不把大SHA引用冒新下载。A原capture132成员/B原capture'+str(c['members'])+'成员实际COMPLETE/natural0后receipt→ZIP，全部SHA/CRC/唯一成员/源/完整argv/原GPU-CPU回执/full关联通过。\n\n'
 report+='永久D '+str(D)+'。人类新24h仍仅估Oct7UTC13:35，保守13:30非平台确认；后续强化实际动态保存09:30/11:30/13:00待执行。旧缺口不回填。\n\n'
 report+='后续先为唯一固定best补Acc7/Acc2/F1/MAE/Corr，所有指标共用预测/选模规则，不为指标单独挑checkpoint。旧DEV五项补算见CaReFlow五指标统一评价与实际DEV结果.md，不能与此INNER或论文Test横比。未来头232/9FIT、201/9EVAL，一预指定真实供体关断delta及完整Z/非退化/ULP/有效视频/成本与标签角色先冻结过门；δ全0不加epsilon救，U/W同信息同3参数岭.01/free-interval-discrete匹配；仍未实施新头。目标是同协议同一模型五项同时超过CaReFlow，尚未达成最终Test或多种子确认。\n'
 (R/'outputs/完整流单参考100完成与完整保存实际结果.md').write_text(report,encoding='utf-8');write(R/'outputs/完整流单参考100完成与完整保存实际结果.json',joint)
 l=read(R/'outputs/研究建议交流接续.json');old=l['latest_actual_GPU_access_and_continuation'];new=dict(old,status=joint['status'],record_actual_utc=now,last_observed_complete_epoch=100,last_observed_formal_updates=2200,formal100_complete=True,complete_joint_audit=str(D/'complete_reference100_GPU_D_B_CPU_joint_audit.json'),complete_joint_audit_sha256=sha(D/'complete_reference100_GPU_D_B_CPU_joint_audit.json'),whole_checkpoint_only_at_phase_end=True,OUTER_or_head_evaluation_started=False,new_generalization_or_donor_benefit_result=False)
 l['latest_actual_GPU_access_and_continuation']=new;l['second_lease_new_assets']['staged_reference_training']['continuation']=new
 t=l['second_lease_new_assets']['staged_reference_training'];t.update(status=joint['status'],epochs_completed=100,formal_updates=2200,formal100_complete=True,updated_actual_utc=now,INNER_best_epoch=41,INNER_video_MSE_selection_only=gpu['INNER_video_equal_MSE_selection_only'])
 l['updated_at_utc']=now;l['five_metric_DEV']={'report':str(R/'outputs/CaReFlow五指标统一评价与实际DEV结果.md'),'report_sha256':sha(R/'outputs/CaReFlow五指标统一评价与实际DEV结果.md'),'D':'D:/CodexBackups/selective_flow_20261003_1105/five_metric_DEV_actual_20261006T183811Z','scope':'descriptive repeatedly explored DEV only','all_three_own_arms_better_than_predeclared_cached_seed128_each_five':True,'paper_Test_superiority_or_goal_complete':False,'current_100_same_best_five_metrics_pending':True}
 dispatch=read(R/'work/review13_actual_dispatch.json');snap=read(R/'work/review13_actual_wait_snapshot.json');poll=json.loads(snap['content'][0]['text'])['polls'][0]
 if not any(b.get('batch')==dispatch['batch'] for b in l.get('sent_batches',[])):
  l.setdefault('sent_batches',[]).append({'batch':dispatch['batch'],'prompt_sha256':hashlib.sha256(dispatch['prompt'].encode()).hexdigest(),'original_dispatch':str(R/'work/review13_actual_dispatch.json'),'turn':poll['latestTurn']['id'],'last_wait_cursor':poll['cursor'],'status':'SENT_ONCE_FULL_REVIEW_NOT_YET_READ'})
 l['review_thread'].update(status='THIRTEENTH_METRIC_PROTOCOL_REVIEW_PENDING_FULL_READ',current_turn=poll['latestTurn']['id'],last_wait_cursor=poll['cursor'])
 write(R/'outputs/研究建议交流接续.json',l)
 for n in ['完整流续训恢复实际核验.json','完整流单参考分段训练实际接续.json']:
  p=R/'outputs'/n;j=read(p);j['latest_actual_reference100_completion']=joint;j['five_metric_protocol']=l['five_metric_DEV'];write(p,j)
 for n in ['完整流续训恢复实际核验.md','完整流单参考分段训练实际接续.md']:
  p=R/'outputs'/n;p.write_text(p.read_text(encoding='utf-8')+'\n\n最新实际'+now+'：完整100/2200已自然0，D完整权重/B原TorchCPU及A-B动态capture总门通过；best41，仅INNER选模。最新见完整流单参考100完成与完整保存实际结果.md。旧阶段时刻保留，不冒其旧状态为当前。\n',encoding='utf-8')
 p=R/'outputs/研究接续状态.md';text=p.read_text(encoding='utf-8');p.write_text('最新覆盖状态UTC '+now+'：完整流100/2200自然0，D与B原TorchCPU/双capture总门通过，best41/INNER仅选模，禁止重启。先读完整流单参考100完成与完整保存实际结果.md/json、CaReFlow五指标统一评价与实际DEV结果.md/json；旧段时刻保留，不是实时。新头/OUTER/最终Test/五指标整体确认尚未实施。D '+str(D)+'。\n\n'+text,encoding='utf-8')
 dest=D/'joint_records';dest.mkdir(exist_ok=False)
 files=[D/'complete_reference100_GPU_D_B_CPU_joint_audit.json',Path(__file__),R/'outputs/完整流单参考100完成与完整保存实际结果.md',R/'outputs/完整流单参考100完成与完整保存实际结果.json',R/'outputs/研究接续状态.md',R/'outputs/研究建议交流接续.json',R/'outputs/CaReFlow五指标统一评价与实际DEV结果.md',R/'outputs/CaReFlow五指标统一评价与实际DEV结果.json']
 for p in files:shutil.copyfile(p,dest/p.name)
 mf={p.name:sha(p) for p in dest.iterdir()};write(dest/'manifest.json',{'actual_utc':now,'members':mf})
 with zipfile.ZipFile(dest/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
  for n in mf:z.write(dest/n,n)
  z.write(dest/'manifest.json','manifest.json')
 with zipfile.ZipFile(dest/'records.zip') as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for n,h in mf.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 print(json.dumps({'status':joint['status'],'actual_utc':now,'joint_sha256':sha(D/'complete_reference100_GPU_D_B_CPU_joint_audit.json'),'D':str(D),'ZIP_SHA':sha(dest/'records.zip')}))
