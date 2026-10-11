import datetime,hashlib,json,shutil,zipfile
from pathlib import Path
R=Path(__file__).parent.parent;O=R/'outputs';D=Path('D:/CodexBackups/selective_flow_20261003_1105/fixed_head_candidate_actual_20261006T191103Z')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
j=read(D/'complete_label_free_candidate_GPU_D_B_CPU_joint_audit.json');diag=j['diagnostics'];close=read(R/'work/fixed_head_candidate_session_closure_actual.json');now=datetime.datetime.now(datetime.timezone.utc).isoformat()
assert close['all_four_actual_exit0']
report=f'''完整流唯一best41：单固定候选232零标签实际结果

固定候选为首次context更新时将六方向incoming反馈精确置零，其余公式、完整model/统计、输入mask和第二Euler仍重算。不是删flow、换任务权重、扫通道或幅度。源计划在执行前冻结SHA {j['plan_sha256']}。公共构造输入守卫只dummy标签，随后严格恢复唯一best41完整pt；trusted pickle包含未使用标签，但不索引行标签。只访问预留232头FIT输入/9视频，不取201输入或任何真标签，不拟合头或更新模型。

GPU child1742实际UTC19:12:56.393876启动，{j['GPU_actual_exit_utc']}自然exit0。按32×7+8固定批次F→C→C→F共32前向，首批另F/C dummy0/7两前向，总34。参数/缓冲/FIT统计/RNG不变，第一Euler状态相同；F重放、C重放与dummy0/7误差均0。

232/232行delta=pC-pF精确非零且至少一个原FP32 ULP；最大绝对位移{diag['delta_abs_max']:.10f}，视频等权平均绝对位移{diag['video_equal_abs_delta_mean']:.10f}。4delta²权重在9视频都有质量，有效视频质量{diag['W_effective_video_mass']:.6f}，只是集中程度诊断。二分类端点类别差1/232，clip-round七分类端点类别差15/232：说明有限区间存在改变类别的机会，不是正确率提升或实际收益。其余连续误差仍可能变化；free残差输出不受此区间约束。

本次实际耗时{j['GPU_budget']['elapsed_seconds']:.3f}秒，累计峰值allocated/reserved {j['GPU_budget']['peak_allocated_bytes']}/{j['GPU_budget']['peak_reserved_bytes']}字节；无构造后peak重置。完整原NPZ SHA {j['original_array_sha256']}，不含标签。原GPU全PID/argv/退出/source/角色/数组及A44成员捕获已D保存。另一B节点原NumPy child30944 {j['B_CPU_actual_exit_utc']}自然exit0，独立重建delta、4delta²、视频权重、ESS、分类端点和重放，并fresh SHA核既有B完整best原件。不冒本轮Torch模型CPU前向/权重更新或整权重重新传输。

A捕获UTC19:14:07.527147、B捕获UTC19:18:54.020423，各自COMPLETE/natural0后receipt再完整ZIP下载D。44/54成员全SHA/CRC/唯一/source/fullargv/原GPU与CPU回执/旧完整D best freshSHA联结通过。B capture目录后缀191902Z为命名抄写错误，真实开始19:18:49.559629、完成19:18:54.020423以原回执为准，不回填或改时间。大741731206字节best引用联结原D和B真实完整原件，非本轮新下载。

独立采纳第14报告：ESS3/活跃视频3硬门在任何新头数据观测前取消，因为两个3参数闭式拟合成本很小；v1本地准备原源和D保留，v2冻结后执行。exact delta0仍停止候选/W且不加epsilon救；exact absdelta常数使U/W代数重复，exact feature方差0标准化/系数0；集中、近共线、边界只诊断。

本轮联合门SHA {sha(D/'complete_label_free_candidate_GPU_D_B_CPU_joint_audit.json')}。D {D}。四SSH/SFTP明确exit/bye实际0禁复用；C未fresh核，不冒当前三机状态。汇总脚本初次错读actual_budget字段KeyError，改为原final_budget后自然0；原科学源/回执/数组不变。

下一步另冻结完整标签访问与匹配拟合/201一次development评价协议：共同Z=(pF,delta)，r=y-pF，U/W同3参数仿射、FIT-only标准化、video等权、固定岭.01斜率与W全局归一；F/C/常数、各自free/interval/discrete全报五项及视频MSE，不用201挑读出/lambda/结构。当前没有新头拟合、201推理/标签、残差信号或收益，正式全面超过CaReFlow尚未完成。历史TRAIN/INNER探索边界保留，不冒全新确认/wholecrossfit。
'''
(O/'完整流单候选232零标签实际结果.md').write_text(report,encoding='utf-8')
state=read(O/'完整流单候选232零标签采集实际接续.json');state.update(record_actual_utc=now,status=j['status'],actual_child_exit_utc=j['GPU_actual_exit_utc'],D=str(D),joint_sha256=sha(D/'complete_label_free_candidate_GPU_D_B_CPU_joint_audit.json'),diagnostics=diag,session_closure=close,next=j['next'],all_control_sessions_closed=True)
state.pop('active_A_SSH',None);state.pop('active_A_SFTP',None);write(O/'完整流单候选232零标签采集实际接续.json',state)
ledger=read(O/'研究建议交流接续.json');ledger['updated_at_utc']=now;ledger['latest_candidate_actual']={'report':str(O/'完整流单候选232零标签实际结果.md'),'report_sha256':sha(O/'完整流单候选232零标签实际结果.md'),'D':str(D),'joint_sha256':sha(D/'complete_label_free_candidate_GPU_D_B_CPU_joint_audit.json'),'scope':'232/9 label-free real GPU candidate; other-node original NumPy audit; no head fit,201 inputs or gain','diagnostics':diag,'closed_sessions':close}
ledger['next_label_free_candidate_preparation']['superseded_by_actual_collection']=ledger['latest_candidate_actual']['joint_sha256'];write(O/'研究建议交流接续.json',ledger)
p=O/'研究接续状态.md';p.write_text('最新实际UTC '+now+'：唯一best41单固定六反馈关断候选232/9零标签GPU采集自然0，F-C-C-F/dummy重放0、参数/FITstats/RNG不变；232精确非零、最大|delta|.1700737476、ESS7.9017诊断、端点Acc2/Acc7变化1/15非收益。完整D+B原NumPy/双实际capture总门过，四会话实际exit0禁复用。第14全文审阅/独立决定已过，ESS硬门在新数据前取消；头拟合/201输入标签/新五项收益尚未实施。先读完整流单候选232零标签实际结果.md/json及实际接续.json；旧段保留非实时。D '+str(D)+'；整体研究与租期保存未完成。\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
dest=D/'research_records';dest.mkdir(exist_ok=False)
files=[O/'完整流单候选232零标签实际结果.md',O/'完整流单候选232零标签实际结果.json',O/'完整流单候选232零标签采集实际接续.json',O/'研究接续状态.md',O/'研究建议交流接续.json',R/'work/fixed_head_candidate_session_closure_actual.json',R/'work/finalize_fixed_head_candidate_actual_v1.py',Path(__file__),Path(ledger['review14_independent_decision']['report'])]
for f in files:shutil.copyfile(f,dest/f.name)
write(dest/'review14_independent_decision.json',ledger['review14_independent_decision'])
write(dest/'local_aggregation_field_error_and_repair.json',{'first_attempt_exit_code':1,'original_error':'KeyError actual_budget at finalize line77','repair':'Read original final_budget field; source science, original receipts and arrays unchanged','successful_attempt_exit_code':0})
mf={f.name:sha(f) for f in dest.iterdir()};write(dest/'manifest.json',{'actual_utc':now,'members':mf,'fresh_C_D':{x:shutil.disk_usage(x+':/').free for x in ['C','D']}})
with zipfile.ZipFile(dest/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in mf:z.write(dest/n,n)
 z.write(dest/'manifest.json','manifest.json')
with zipfile.ZipFile(dest/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in mf.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
print(json.dumps({'status':'ACTUAL_CANDIDATE_REPORT_STATE_REVIEW14_D_RECORDS_SAVED','records_SHA':sha(dest/'records.zip'),'joint_SHA':sha(D/'complete_label_free_candidate_GPU_D_B_CPU_joint_audit.json'),'files':len(mf)}))
