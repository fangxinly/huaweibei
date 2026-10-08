"""Preserve preceding continuation and new completed research proofs on D."""
from pathlib import Path
import datetime,hashlib,json,shutil,zipfile
base=Path(__file__).parent.parent;out=base/'outputs';dest=Path('D:/CodexBackups/selective_flow_20261003_1105/scalar_geometry_actual_20261006T060719Z')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
joint=json.loads((dest/'joint_preservation_audit.json').read_text());assert joint['status']=='SCALAR_GEOMETRY_GPU_D_OTHER_NODE_CPU_AND_ACTUAL_CAPTURES_JOINTLY_VERIFIED'
assert shutil.disk_usage(dest).free>1073741824
records=dest/'research_records';prior=records/'preceding_state';prior.mkdir(parents=True,exist_ok=False)
for name in ['研究接续状态.md','研究建议交流接续.json']:shutil.copy2(out/name,prior/name)
coord=json.loads((out/'研究建议交流接续.json').read_text(encoding='utf-8'))
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review')
advice=review/'outputs/第一次独立审视与短实验优先建议.md'
coord['updated_at_utc']=now
coord['review_thread'].update(status='first_review_completed',last_wait_cursor='5c299d87-8d70-412a-94b9-bdc45953b3ac:4',last_completed_turn='01a10fcb-6769-7ba2-8a80-ed3f3c549d99')
coord['received_reviews']=[dict(batch=coord['first_sent_batch'],path=str(advice),sha256=sha(advice),completed_utc='2026-10-06T06:03:53Z',read_and_considered=True)]
coord['accepted_suggestions']=[dict(suggestion='锚点/Jacobian首步审计',reason='标量平方目标同点首步理论上只差符号/尺度；本轮真实GPU1281行已验证',evidence=str(out/'首步标量方向几何GPU与D_CPU快照联合核验.json'),limitation='仅首步几何；不是lambda0消息求解器回退或新控制分数'),dict(suggestion='每折CAL视频等权剩余误差信号/删除视频敏感性',reason='教师绝对质量不能替代对当前参考剩余误差的预测；每折仅6个CAL视频',status='next_to_freeze_not_fitted'),dict(suggestion='真实候选并集与同预算反向旧控制、生成/接受交叉',reason='区分新增候选能力和接受信号；不以代数改名或归一化抵消收缩冒方向创新',status='next_to_implement_and_gpu_precheck')]
coord['deferred_suggestions']=[dict(suggestion='按折完整隔离参考短可行性',reason='需合法公共初始化/FIT-only全链条及新训练预算；旧全TRAINfit不能替代',status='not_started'),dict(suggestion='新学生100轮',reason='尚无真实CAL信号、候选增量和公平参考隔离证据；不能为了填卡跳依赖',status='not_started')]
coord['rejected_suggestions']=[dict(suggestion='部署全TRAIN事后缩放/混合系数或以epsilon经验误差冒真实安全界',reason='标签已知探索或未证条件均值界；不构成可部署/泛化证据')]
coord['scientific_state']='新同点首步标量GPU几何完成且D/异节点CPU/actual capture保存；无新控制MSE、CAL拟合/EVAL或100轮'
batch='scalar_geometry_diagnostic_v1_20261006T060719Z_'+joint['arrays_sha256']
coord.setdefault('evidence_batches',[]).append(dict(batch=batch,status='ready_to_send_after_preservation',arrays_sha256=joint['arrays_sha256'],report=str(out/'首步标量方向几何GPU核验与优化决策.md'),report_sha256=sha(out/'首步标量方向几何GPU核验与优化决策.md'),joint_evidence=str(dest/'joint_preservation_audit.json'),joint_sha256=sha(dest/'joint_preservation_audit.json')))
coord['permanent_current_evidence_directory']=str(dest)
(out/'研究建议交流接续.json').write_text(json.dumps(coord,ensure_ascii=False,indent=2),encoding='utf-8')
state='''更新UTC@utc@。新同点首步标量方向GPU诊断已实际完成并D/异节点B CPU/两真实capture联合通过：C预检13766 UTC06:09:00 exit0；正式14010 UTC06:15:04.427385 exit0/退休，1281TRAIN/52视频/41批，23.19s/100131328bytes。缓存坐标终端/供体冻结，未重复原输入DeBERTa整模型前向；无真实y/DEV/TEST/CAL拟合/EVAL指标/消息路径修改/学生训练。参数SHA前后一致、无参数梯度/0optimizersteps；毒化标签0、单token双精度FD-HVP5.05e-9/4.51e-8。源SHAc84b97838d76d373aa4d1e65efd8ca39d9b929ba4da8b377306b820f8e44bcd1，planSHA d7e79462b68e19f491ac73997b68cc37beb8a27924e6f3cbfa03e87a29cb254e。

实质结果：同一点平方标量目标g_t=2w(p_F-t)J，旧/教师均非零1281行，同向666/反向615（51.9906%，不是纠错正确率），共线误差3.41e-14；正lambda收缩目标后归一化方向差0、g_lambda=lambda*g_1误差0。p_F-p0 RMS.18456148不能互换。仅首步，不能推所有三步路径；lambda0仅梯度0，不冒新求解器精确回退或学生主辅门控已过。下一不训练只重复归一化Jacobian的“新方向”网络。

永久D scalar_geometry_actual_20261006T060719Z，完整32,258,674bytes新数组SHAe0016c23628ca0584784e417465277e4ef516cc884ff22ea4f2113c26f9ead91和23原文件包实际下载；B原CPU NumPy/exit/argv/回执UTC06:19:37.392635完成已下载，非CPU模型前向。C/B capture19实际UTC06:19:37.379112/06:22:59.276574，全SHA/CRC/member/新源/原过程/回执和旧科学证据/大SHA不变过。新32MB NPZ是capture大引用，另D完整原件+CPU原包对应，不冒再次在小ZIP下载；旧完整weight未重传。自capture stdout原空→原完成行自然闭合单独按旧exit0核验；原审核拒绝保留，新v3/v2只联结大NPZ准确路径/字节/SHA，不改科学断言或capture19。A无本轮新查询/材料，保留此前三capture不冒新三节点。详见首步标量方向几何GPU核验与优化决策.md与首步标量方向几何GPU与D_CPU快照联合核验.json。

当前工具可正常发送；UTC06:25:08 C/B SSH22394/89258及SFTP5047/88395全部明确exit/bye真实exit0，旧ID禁write_stdin，下轮需fresh。不能沿用旧审批阻塞，不称底层修复。只授权新A53449 UUIDGPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7/B53416 GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067/C53458 GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa；域名见冻结plan/此前回执，旧址/N/R永禁连。真实password后仅人类原凭据，禁密码文件/命令/自动提示/猜测，先核UUID/argv/compute/完成/源/空间，结束核真实退出；新拒绝不换路绕过。

下一先冻结仅本折CAL真实残差信号拟合（视频等权C/B、lambda clip0..1、删除视频只敏感性），再实际消息候选旧池/教师池/并集及同预算旧+反向旧，方向生成和接受交叉。固定0,1/8,1/4,1/2,1消息半径，真实终端前向、输出上限预筛后float64最早最小/严格改善及F回退；不移入事后Oracle系数，不先加epsilon/eta。尚未CAL真实拟合、实现新候选求解器、预检学生或新100。新源/角色/标签/锚目标detach/lambda0消息回退/真实步长与拼接重放/单token二阶/预算空间/actual capture全核才执行。新100必须候选增量和非Oracle净收益、有公平参考隔离及初始化/容量/100订单/shared10等依赖，不填卡。

同T_k FIT/INNER实际采集v2已三折完成D same_teacher_collection_actual_20261006T053835Z/异节点A→B/B→C/C→A原回执/三05:51完成capture；FIT/INNER695/153、715/142、714/143，六NPZ row_ids/fold/mu，严格INNER原输入重放0/换标签0/fit缓冲exact/参数不变/零optimizersteps，临时optimizer构造丢弃。S_k只同T_k自身目标，禁跨T_j间接携带S_k外折标签。CAL18视频418/EVAL34视频863，三CAL/EVAL136/297、148/276、134/290固定，EVAL各臂预测SHA后统计；全TRAIN已探索且A/C2全TRAINfit/DEVselected不能冒全新确认或全流程crossfit。按折全链参考另需公共预训练随机任务初始化及新协议，不用旧A/C2冒隔离。

九旧流100/full/20诊断/D/CPU、三91818老师100/best85/45/68/full/D/CPU、1281/52 OOF和TRAIN Oracle不重跑重传。原C仅10失败，旧C最终/旧14:30缺口不回填。OOF老师MSE.6293898611/MAE.5965080822仅老师质量；固定C2 TRAIN原F.0161276872/旧学习.0152679208降5.33%/直接OOF末步.0767956564害74.89%/mu选步.0789019134害75.10%仍负。旧流仅520506供体头训练，老师184749003参数全微调。Oracle65.11%/事后输出组合6.61%不冒严格上界/泛化或部署。

独立建议聊天“多模态消息控制研究建议”01a10fcb-6663-70a2-9a76-60e5634d0c03第一次建议已完成/全文审阅，采纳锚/首步（本轮实测）、CAL信号及候选并集/等预算对照；暂缓新100与隔离参考。原报告在2026-10-06/multimodal-flow-research-review/outputs/第一次独立审视与短实验优先建议.md。新实质结果D/CPU保存后只向同聊天发送一次，无密码/SSH凭据，按研究建议交流接续.json批次SHA去重；建议只分析不连GPU/改主源/训练/子代理，主聊天执行，不等建议停健康训练。未完成第二次建议不冒已返回。

capture19固定SHA6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad，公共root/data/coding/soft_vector_research_20261005T1220Z、deployment/data/coding/jacobian_aligned_v2_deployment_20261005T1327Z，真实clock唯一stamp，实际CAPTURE_COMPLETE/exit0后receipt→ZIP审核。D优先每次fresh盘禁删；本轮C约8.16GB/D39.62GB。24h仅估Oct6UTC12:08:17/BJ20:08:17平台未核；UTC08:08/10:08/11:38强化实际保存仍未执行，不伪称或回填。禁subagents/设置/浏览器/续租/关机/停健康训练/改完成冻结源/TEST挑结构/五seed稳定/SOTA/语义真值。原十分钟频率、健康未变或无行动静默，只实质结果/新失败/完成/必要行动通知。整体研究/后续优化/租期保存未完。
'''.replace('@utc@',now)
(out/'研究接续状态.md').write_text(state,encoding='utf-8')
targets=[]
for path in sorted((base/'work').glob('*scalar*geometry*.py')):targets.append(('sources/'+path.name,path))
for name in ['首步标量方向几何GPU核验与优化决策.md','首步标量方向几何GPU与D_CPU快照联合核验.json','首步几何完成捕获大数组引用原审核拒绝与修订.json','首步几何本地准备原v1失败记录.json','首步几何捕获审核v1自记录自然闭合差异.json','首步标量方向几何短诊断冻结接续.json','研究接续状态.md','研究建议交流接续.json']:
    targets.append(('reports/'+name,out/name))
targets.extend([('review/'+advice.name,advice),('review/audit_unlabelled_baseline.py',review/'work/audit_unlabelled_baseline.py'),('review/unlabelled_baseline_audit.json',review/'work/unlabelled_baseline_audit.json'),('sources/run_same_teacher_capture_v1.py',base/'work/run_same_teacher_capture_v1.py')])
manifest={}
with zipfile.ZipFile(records/'research_proofs.zip','x',zipfile.ZIP_DEFLATED) as z:
    for name,path in targets:
        data=path.read_bytes();target=records/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        z.writestr(name,data);manifest[name]=dict(bytes=len(data),sha256=sha(path))
    z.writestr('member_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(records/'research_proofs.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,item in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==item['sha256']==sha(records/name)
r=dict(status='SCALAR_GEOMETRY_RESEARCH_REPORTS_REVIEW_AND_SOURCE_D_PRESERVED',actual_utc=now,files=len(manifest),zip_sha256=sha(records/'research_proofs.zip'),files_sha256=manifest,free_C_bytes=shutil.disk_usage('C:/').free,free_D_bytes=shutil.disk_usage('D:/').free,no_old_files_deleted=True)
(records/'preservation_receipt.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:r[k] for k in ['status','actual_utc','files','zip_sha256','free_C_bytes','free_D_bytes']}))
