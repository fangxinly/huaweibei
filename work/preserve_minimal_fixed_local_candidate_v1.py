import datetime as dt, hashlib, json, shutil, zipfile
from pathlib import Path

b=Path.cwd();r=b/'work/minimal_fixed_flow_v2_local_20261006T1214Z';now=dt.datetime.now(dt.timezone.utc);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert shutil.disk_usage('D:/').free>=1073741824
receipt=json.loads((r/'local_static_numpy_receipt.json').read_text(encoding='utf-8'));assert receipt['status']=='LOCAL_CANDIDATE_AST_AND_SYNTHETIC_NUMPY_SCHEDULE_ONLY_PASSED'
assert sha(r/'legacy_flow_model.py')==sha(b/'work/inflow_counterfactual_v5_deployment_20261005T0520Z/legacy_flow_model.py')
assert sha(r/'finite_single_token_reader_v1.py')==sha(b/'work/finite_single_token_reader_v1.py')
for name,value in receipt['source_sha256'].items():assert sha(r/name)==value
teacherplan=b/'work/group_teacher_plan_20261005T1650Z/group_teacher_plan_v3.json';assert sha(teacherplan)=='f0878166da09433c27119a08fdfd8682f9b9e29f2bfe0a758f29eee45cf131a4'
plan=dict(status='LOCAL_FLOW_SUBMODULE_CANDIDATE_ONLY_SOURCE_PINS_NOT_FORMAL_TRAINING_AUTHORIZATION',utc=now.isoformat(),new_source_sha256={p.name:sha(p) for p in r.glob('*.py')},teacher_role_plan_sha256=sha(teacherplan),seed_proposed_not_initialization_executed=91819,fold=0,rows=dict(FIT=695,INNER=153,OUTER=433),architecture='Direct retained module constructor; one pass, two Euler, six directed donors; fixed .5 weights each, .125 donor coefficient from zero context',objectives=dict(task_MSE=1.,flow_matching=.02,cycle_reconstruction=.01,unimodal=.05,variance_floor=.01),removed_without_constructing=['old_feedback','pair_heads','utility_heads','predicted_gate','10_counterfactual_reference_terminal_branches'],gradient_boundaries='Donor features detach; stage context and cycle terminal remain attached; targets retain original detach. Actual parameter gradients untested.',singleton_patch='Exact original analytic single-token source copied and attached to new reader; AST compiled, no new Torch/HVP replay.',constructor_RNG_equivalence_to_old=False,public_pretrained_initialization_verified=False,full_encoder_runtime_or_guarded_trainer_implemented=False,complete100orders_frozen=False,GPU_memory_time_budget_passed=False,at_least_two_hour_save_reserve_passed=False,new_training_allowed=False,new_scores=False,actual_local_checks='AST compile/call and parameter construction boundary, independent NumPy algebra and synthetic masking/two-stage schedule only',formal_execution_remaining=receipt['mandatory_real_remaining'])
(r/'local_candidate_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/('minimal_fixed_v2_local_candidate_'+now.strftime('%Y%m%dT%H%M%SZ'));D.mkdir(exist_ok=False)
for p in r.iterdir():
 if p.is_file():shutil.copy2(p,D/p.name)
shutil.copy2(Path(__file__),D/Path(__file__).name)
preceding=D/'preceding_state';preceding.mkdir()
for name in ['研究接续状态.md','研究建议交流接续.json','完整流fixed_v2本地候选源码接续.md']:
 p=b/'outputs'/name
 if p.exists():shutil.copy2(p,preceding/name)
report=f'''完整流fixed v2本地候选源码接续

实际UTC {now.isoformat()}。原完成科学源/协议均未改；新flow子模块独立候选目录{r.as_posix()}，永久D {D.as_posix()}。主源码SHA {plan['new_source_sha256']['minimal_fixed_flow_v2.py']}，候选计划SHA {sha(r/'local_candidate_plan.json')}。

直接构造有明确预测/辅助作用的forward/backward fields、解析单token reader、role/unimodal heads及六donor；从构造依赖链删除旧feedback、pair/utility/predicted gate，不执行10条参考终端。不是系数置零继续计算。新任务随机数消费因此不同，不能沿用旧init等价或task权重，未来须公共预训练/新clean initial单独存证。

一pass两Euler：第一步使用零context，第一步后根据stage slots更新context，第二次update返回原context。fixed权重.5、selected再乘.5、context再乘.5，因此各供体从零context的实际系数.125。新辅助目标严格MSE+.02FM+.01cycle+.05unimodal+.01variance；unimodal为source和stage1 observer各.5。cycle终端/context未detach，第二阶段FM context也未detach，不能称只训练backward；donor特征及原目标detach边界保留。

本地AST编译、声明模块/调用删减核验与NumPy独立上下文恒等式差0、synthetic双步/单token mask日程检查完成exit0。不导入或运行Torch模型，不读真实数据/标签，不核GPU、真实参数梯度、HVP、完整初始化、原输入或整pt重载，不产生任何分数。解析单token旧源码只是精确复制，旧验证不冒新整模型验证。

尚未实现完整encoder runtime/角色守卫与预检训练器；公共预训练资产、FIT-only统计、INNER原输入dummy标签严格重放、优化器覆盖/唯一性、正常两真实步骤/23尾批/单token梯度、clean initial与预检分离、完整100订单/选模/实际显存时间与完整D+异节点CPU/至少2h余量仍全待实现与实核。这不是正式源/预算冻结，不允许启动训练。A key和B认证限制未恢复，平台释放未核，租期估计经过也不能填卡或续租。

只本地准备按用户静默规则，不向建议聊天重发、不通知新成绩。当前真实标量/消息正负分数、最新三远端11:39保存、C12:06只读和A/B失败状态不变。下一步本地可准备完整runtime和角色守卫，数值检查须待可验证环境，不能把静态图当真实autograd。
'''
(b/'outputs/完整流fixed_v2本地候选源码接续.md').write_text(report,encoding='utf-8');(D/'完整流fixed_v2本地候选源码接续.md').write_text(report,encoding='utf-8')
manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in D.iterdir() if p.is_file()}
with zipfile.ZipFile(D/'source_candidate_package.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for name in manifest:z.write(D/name,name)
with zipfile.ZipFile(D/'source_candidate_package.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest)
 for name,item in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==item['sha256']
proof=dict(status='LOCAL_CANDIDATE_ONLY_SOURCE_PLAN_STATIC_SYNTHETIC_RECEIPT_D_SHA_ZIP_CRC_PASSED',utc=now.isoformat(),members=manifest,package_sha256=sha(D/'source_candidate_package.zip'),GPU_or_full_Torch_verified=False,remote_capture_or_other_node_CPU=False,new_scores=False,whole_research_complete=False)
(D/'preservation_receipt.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
sp=b/'outputs/研究接续状态.md';state=sp.read_text(encoding='utf-8');state+=f'\n本地源码准备UTC {now.isoformat()}：fixed v2新flow子模块候选/直接保留构造/删除pair、utility及额外参考支路/两Euler与每供体.125时序已本地实现，AST+NumPy synthetic通过；不冒Torch/autograd/GPU/完整runtime/预检/100。原科学源均未改。D {D.as_posix()} source_candidate_package.zip 全SHA/CRC/成员过，源码主SHA{plan["new_source_sha256"]["minimal_fixed_flow_v2.py"]}；完整runtime、标签守卫、公共init、FIT统计、实际梯度/重载/预算仍未完成。见完整流fixed_v2本地候选源码接续.md。仅准备，不发新成绩/重发建议；A/B连接限制无新可信信息，未重试任何节点。\n';sp.write_text(state,encoding='utf-8')
lp=b/'outputs/研究建议交流接续.json';ledger=json.loads(lp.read_text(encoding='utf-8'));ledger['updated_at_utc']=now.isoformat();ledger['local_fixed_flow_candidate']=dict(directory=str(D),candidate_plan_sha256=sha(r/'local_candidate_plan.json'),phase='SUBMODULE_AST_NUMPY_PREPARATION_ONLY',new_GPU=False,new_Torch=False,full_runtime_trainer=False,new_scores=False,sent_to_review=False);lp.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
records=D/'records';records.mkdir();updated=[sp,lp,b/'outputs/完整流fixed_v2本地候选源码接续.md',D/'preservation_receipt.json'];record_manifest={}
for p in updated:
 q=records/p.name;shutil.copy2(p,q);record_manifest[q.name]=dict(bytes=q.stat().st_size,sha256=sha(q))
(D/'updated_short_state_preservation.json').write_text(json.dumps(dict(status='UPDATED_STATE_FILES_D_SHA_EXACT',utc=now.isoformat(),files=record_manifest,source_plan_sha256=sha(r/'local_candidate_plan.json')),ensure_ascii=False,indent=2),encoding='utf-8')
(r/'permanent_D_location.json').write_text(json.dumps(dict(directory=str(D),preservation_receipt_sha256=sha(D/'preservation_receipt.json')),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'directory':str(D),'candidate_plan_SHA':sha(r/'local_candidate_plan.json'),'source_SHA':plan['new_source_sha256']['minimal_fixed_flow_v2.py'],'prepared_only':True,'new_GPU':False,'new_scores':False}))
