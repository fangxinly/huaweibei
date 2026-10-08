from pathlib import Path
import datetime,hashlib,json,shutil
cwd=Path(__file__).parent.parent;out=cwd/'outputs';d=Path('D:/CodexBackups/selective_flow_20261003_1105/cal_video_equal_signal_actual_20261006T063535Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();now=datetime.datetime.now(datetime.timezone.utc).isoformat()
records=d/'research_records';records.mkdir(exist_ok=True);prior=records/'preceding_state';prior.mkdir(exist_ok=True)
for name in ['研究接续状态.md','研究建议交流接续.json','视频等权CAL信号执行接续.json']:
    if not (prior/name).exists():shutil.copy2(out/name,prior/name)
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第二次独立审视_符号幅度与候选池最小对照.md')
shutil.copy2(review,records/review.name)
r=json.loads((d/'execute/calibration_result.json').read_text());joint=json.loads((d/'joint_preservation_audit.json').read_text())
assert joint['status']=='CAL_SIGNAL_D_OTHER_NODE_CPU_AND_ACTUAL_CAPTURE_JOINTLY_VERIFIED'
report='''固定CAL视频等权残差信号：实际结果与下一步

实际CPU正式进程37008在UTC2026-10-06 06:36:27.943040自然exit0；预检15396自然exit0。固定18视频418行，只解码授权CAL真实标签；EVAL863行未新解码或计算指标。科学源/角色/拟合规则在真实CAL数值访问前冻结。原TRAIN此前已有探索，参考C2全TRAINfit/DEVselected，本次角色隔离不冒全新确认或全流程crossfit。

令r=y-pF、t=mu-pF，C=视频等权mean(r*t)，B=视频等权mean(t²)，lambda=clip(C/B,0,1)。六视频每折等权；删除一个视频仅敏感性，不改正式系数。输出变化上限仅来自CAL原旧控制|p_old-pF|第95百分位，非风险安全界。

|fold|C|B|lambda|删视频lambda范围|输出上限|
|---|---:|---:|---:|---:|---:|
'''
for f in r['folds']:report+=f"|{f['fold']}|{f['C_video_equal']:.9f}|{f['B_video_equal']:.9f}|{f['lambda_video_equal']:.9f}|{f['loo_lambda_range']}|{f['output_cap_old_native_abs_shift_quantile95']:.9f}|\n"
report+='''
第0折C为负，CAL教师收缩接受器必须精确回退原F，旧原生控制单列基线。第1、2折C为正且六个删除视频C均正，支持一次有限消息候选短实验。这里没有真实消息控制收益或新EVAL分数，C≤0也不否定任意非线性候选的价值。

第二次独立建议已全文审阅：采纳旧末端位移/教师末端位移与反向旧末端位移、固定F锚、同预算9对9、生成与接受交叉。下一13共享名义消息：F及三方向×四半径1/8,1/4,1/2,1；各臂真实终端前向、同一输出上限，float64原F相对Q严格改善否则回退。教师原mu只生成，CAL lambda只用于接受，不能把首步-J当反向旧三步位移。精确重复消息会识别并在选择保持最早来源；本版统一原批量重复前向13次以保证数值重放，预算两主池相同，不补额外方向。

永久D原24文件ZIP 57,164字节SHAad19c84ee5250a3a35007b88c0267583177977574fb81f9f8430227a350c4d13；不包含全TRAIN标签档案。B异节点NumPy逐数组/数值重建原回执已下载，非CPU整模型前向。B实际capture19 UTC06:40:13.961145，ZIP75,178,098字节SHA28e11eb7d9d1c38a350cbbd43d303a1b10ce9bb997c41c30dd5209cbd1ed1be7，530成员/CRC/SHA及491旧科学成员与10大引用未变联合过。只有B新capture；旧完整模型未重传。
'''
name='固定CAL视频等权残差信号实际结果.md';(out/name).write_text(report,encoding='utf-8');shutil.copy2(out/name,records/name)
for name in ['joint_preservation_audit.json','local_independent_audit.json']:shutil.copy2(d/name,out/('CAL_'+name))
coord=json.loads((out/'研究建议交流接续.json').read_text(encoding='utf-8'));coord['updated_at_utc']=now
rt=coord['review_thread'];rt.update(status='second_review_completed_and_read',last_wait_cursor='5c299d87-8d70-412a-94b9-bdc45953b3ac:8',last_completed_turn='01a10fe7-4cea-7b50-bc34-71d5c1d2871a')
if not any(x.get('sha256')==sha(review) for x in coord['received_reviews']):coord['received_reviews'].append(dict(path=str(review),sha256=sha(review),completed_utc='2026-10-06T06:36:10Z',read_and_considered=True,accepted='等预算9对9、末端位移而非首步Jacobian、固定F锚、生成接受隔离；统一原批量13次终端调用并识别重复，预算一致；暂缓100'))
coord['scientific_state']='仅CAL实际拟合及D/异节点CPU/capture完成；无新EVAL分数。新匹配消息池刚冻结，未GPU预检/正式执行。'
for item in coord['accepted_suggestions']:
    if 'CAL' in item['suggestion']:item['status']='ACTUAL_CAL_COMPLETED_WITH_OTHER_NODE_CPU_AND_CAPTURE';item['evidence']=str(d/'joint_preservation_audit.json')
(out/'研究建议交流接续.json').write_text(json.dumps(coord,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'视频等权CAL信号执行接续.json').write_text(json.dumps(dict(stage='ACTUAL_CAL_COMPLETE_D_CPU_CAPTURE_JOINT_PASSED',updated_utc=now,permanent_root=str(d),result_sha256=sha(d/'execute/calibration_result.json'),joint_sha256=sha(d/'joint_preservation_audit.json'),EVAL_metrics_computed=False),ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['研究建议交流接续.json','视频等权CAL信号执行接续.json']:shutil.copy2(out/name,records/name)
print(json.dumps(dict(report=str(out/'固定CAL视频等权残差信号实际结果.md'),joint_status=joint['status'])))
