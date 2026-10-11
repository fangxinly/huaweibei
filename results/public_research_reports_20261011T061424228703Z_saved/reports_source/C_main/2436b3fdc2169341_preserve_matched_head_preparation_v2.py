import ast,datetime,hashlib,json,shutil,sys,zipfile
from pathlib import Path
R=Path(__file__).parent.parent;O=R/'outputs';W=R/'work'
stamp=sys.argv[1];actual=datetime.datetime.strptime(stamp,'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).isoformat()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
local=W/('matched_head_fit_preparation_v2_'+stamp);D=Path('D:/CodexBackups/selective_flow_20261003_1105')/local.name
local.mkdir(exist_ok=False);D.mkdir(exist_ok=False)
sources=['matched_fixed_residual_heads_v1.py','matched_fixed_residual_heads_v2.py','fit_matched_fixed_heads_original_v1.py','audit_matched_head_fit_original_CPU_v1.py','matched_head_fit_stage_wrapper_v1.py','check_matched_head_preparation_v2.py','check_matched_head_preparation_v2.json',Path(__file__).name]
for name in sources:
 if name.endswith('.py'):ast.parse((W/name).read_text(encoding='utf-8'))
 shutil.copyfile(W/name,local/name)
report=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第十五次独立审视_UW闭式协议的权重标准化与冻结顺序.md')
decision={'status':'FIFTEENTH_FULL_REPORT_READ_AND_INDEPENDENTLY_CONSIDERED','actual_utc':actual,'report':str(report),'report_sha256':sha(report),'full_read':True,
 'accepted':['共同FIT视频等权人口mu/std在标签前保存，U/W共享X；W仅改变全局归一损失q','截距不惩罚，两斜率ridge .01；同3x3求解，不二次mean或每视频W归一','W真实加权截距条件与独立增广最小二乘核验；rawQ ridge系数为c*.01','真实pC离散选择、负delta同规则、tie回F；九读出同五项和同视频MSE','完整协议先于232监督和201输入；全九预测物理SHA先于201一次标签评分'],
 'deferred':['精确201输入守卫/原输入collector、冻结九预测与一次统一评价runner/独立CPU指标审核/新capture源全覆盖，完成全协议后才实际拟合','官方匹配CaReFlow完整benchmark与跨seed功效；201仅探索开发'],
 'rejected':['用建议全文或本地合成核验冒实际源已获独立代码审核、真实残差信号或新五指标收益','按232端点覆盖/ESS/拟合目标挑特征、lambda、幅度或201读出','新增教师/候选/seed扩容以替代固定匹配小实验'],
 'cost_and_stopping':'本轮仅本地源和合成数据；未来固定两个3参数头+一个常数，当前不因审视增加科学预检。完整protocol和资产/保存门未齐，真实拟合尚未启动。',
 'advisor_reviewed_new_fit_source':False,'new_task_experiment_or_score':False}
write(local/'review15_independent_decision.json',decision);shutil.copyfile(report,local/report.name)
wait=W/'review15_actual_once_wait.json'
if wait.is_file():shutil.copyfile(wait,local/wait.name)
test=read(W/'check_matched_head_preparation_v2.json')
plan={'status':'LOCAL_MATCHED_HEAD_FIT_RUNNER_AUDIT_WRAPPER_PREPARATION_NOT_EXECUTION_FREEZE','actual_utc':actual,'source_and_record_SHA':{p.name:sha(p) for p in local.iterdir()},'review15_report_sha256':sha(report),'same_predeclared_recipe':True,'complete_execution_protocol_ready':False,'true_headFIT232_labels_or201_inputs_accessed':False,'synthetic_check':test,'missing':['exact201-only original-input collector and role/source/checkpoint guard','nine-readout physical prediction freeze then once201 label metric runner','independent othernode original prediction/metric audit and all roots/source/fullargv capture','complete protocol source freeze and real fresh identity/assets/space/human lease budget with two-hour preservation reserve']}
write(local/'preparation_plan.json',plan)
files={p.name:sha(p) for p in local.iterdir()};space={x:shutil.disk_usage(x+':/').free for x in ('C','D')}
write(local/'manifest.json',{'actual_utc':actual,'files':files,'fresh_C_D_free_bytes':space,'remote_capture':False,'new_other_node_CPU':False})
for p in local.iterdir():shutil.copyfile(p,D/p.name)
with zipfile.ZipFile(D/'preparation.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in local.iterdir():z.write(p,p.name)
with zipfile.ZipFile(D/'preparation.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in files.items():assert sha(D/n)==h and hashlib.sha256(z.read(n)).hexdigest()==h
prep={'status':plan['status'],'actual_utc':actual,'local':str(local),'D':str(D),'plan_SHA':sha(local/'preparation_plan.json'),'ZIP_SHA':sha(D/'preparation.zip'),'source_SHA':{n:sha(local/n) for n in sources if n.endswith('.py')},'synthetic':test,'actual_task_fit_or_GPU_or201_evaluation':False,'next_required':plan['missing']}
write(O/'完整流匹配U_W头协议本地准备接续.json',prep)
(O/'完整流匹配U_W头协议本地准备接续.md').write_text('完整流匹配U/W头：本地准备接续\n\n'+actual+'。新增v2共同设计在标签前固定、原232-only CPU拟合runner、自然退出wrapper和异节点原数组独立正规方程审核候选；未执行真实拟合。两头同视频等权FIT标准化、全局W归一、ridge .01斜率/自由截距，离散直接用真实pC。\n\n合成独立增广最小二乘差 '+str(test['independent_augmented_lstsq_max_error'])+'，Q尺度恒等误差 '+str(test['structured_Q_data_identity_error'])+'；自己JSON磁盘重放0，4个核心负门与2个不完整协议入口拒绝通过。无官方数据、标签、Torch/GPU、201输入或新成绩。AST及合成检查不能冒未来环境实际通过。\n\n第15全文已审阅，接受数值匹配和时间顺序；建议未审核新实际拟合源。现仍缺精确201原输入collector、九预测冻结/一次评分runner、独立CPU指标/capture和完整源协议冻结。没有complete protocol标志，不启动真实标签拟合。\n\n本地 '+str(local)+'；完整D '+str(D)+'，全成员SHA/ZIP CRC/唯一成员过，是本地准备保存，不是remote capture或异节点CPU实测。旧完成实验/旧v1准备保留。\n',encoding='utf-8')
(O/'第十五次建议独立决策与匹配头准备.md').write_text('第十五次建议独立决策\n\n'+actual+'，全文 '+str(report)+'，SHA '+sha(report)+'。接受共同FIT统计、两个归一q与正确截距/ridge、同一评价分布及标签时间顺序，见D review15_independent_decision.json。暂缓正式benchmark和201执行，拒绝把建议/合成核验当真实收益或实际源独立审核。下一步完成最小执行协议再拟合；没有新增科学门槛或扩容。\n',encoding='utf-8')
ledger=read(O/'研究建议交流接续.json');ledger['updated_at_utc']=actual;ledger['review15_independent_decision']=decision;ledger['latest_review']={'status':decision['status'],'batch':'fixed_head_candidate_actual_20261006T191103Z','review':decision};ledger['next_matched_heads_local_preparation']=prep
ledger['received_reviews'].append(decision)
for b in ledger['sent_batches']:
 if b.get('batch')=='fixed_head_candidate_actual_20261006T191103Z':b.update(status='COMPLETE_REPORT_FULLY_READ_AND_CONSIDERED',report_sha256=sha(report),actual_read_record_utc=actual,cursor='c9fdcd3f-7d4b-4678-809d-0742a5ab961e:6')
write(O/'研究建议交流接续.json',ledger)
p=O/'研究接续状态.md';p.write_text('本地准备UTC '+actual+'：第15全文已读并独立采纳共同FIT统计/归一权重/截距/ridge/标签顺序。新v2 core+232-only FIT runner+自然退出wrapper+原数组CPU审核仅源候选；合成独立解差2.22e-16、磁盘头重放0、不完整协议两入口拒绝。D '+str(D)+'全SHA/ZIPCRC保存。尚无真实头标签/拟合/201输入或新五指标；先完成201原输入与一次评分/原CPU/capture完整冻结，不冒代码或真实预算通过。整体研究与租期保存未完成，全部旧session禁复用。\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
args=read(W/'candidate_existing_automation_update_args.json');lines=args['prompt'].splitlines()
for i,line in enumerate(lines):
 if line.startswith('第13/14完整建议'):
  lines[i]='第13/14/15完整建议均已全文审阅并独立决定，原第15批已发送一次，无待返回审视。第15接受共同a=1/(9*n_v)的FIT人口mu/std、qU=a/qW=a*4delta²/c全局归一、ridge .01仅两斜率/自由截距、同视频MSE与同五项评价、九预测SHA先于一次201标签；建议未审核新实际拟合源，不冒收益。仅分析聊天01a10fcb-6663-70a2-9a76-60e5634d0c03，人类授权双向，禁GPU/改主源/训练/子代理/凭据/新聊天。批次SHA去重，健康/准备/同证据不发，不等建议耽搁保存。'
lines.insert(1,'先补读完整流匹配U_W头协议本地准备接续.md/json及第十五次建议独立决策与匹配头准备.md。最新本地 '+str(local)+' 为源准备非完整执行冻结；v2 core/232-only CPU fit runner/自然退出wrapper/原CPU数组审核AST和合成核验通过，无真实FIT标签/201输入或新成绩。还缺精确201原输入collector、九预测锁定后一次评分、原CPU指标/capture覆盖及完整protocol冻结；依赖齐后按既有授权实际执行，不机械扩容或把本地准备当结果。')
args['prompt']='\n'.join(lines);write(W/'matched_head_preparation_existing_automation_update_args.json',args)
print(json.dumps({'status':prep['status'],'D':str(D),'ZIP_SHA':prep['ZIP_SHA'],'files':len(files),'fresh_C_D':space,'review15_recorded':True,'new_real_experiment':False}))
