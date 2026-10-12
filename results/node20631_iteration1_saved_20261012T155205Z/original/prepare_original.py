import ast,datetime as dt,json,pathlib,shutil,sys,zipfile
P=pathlib.Path;B=P(__file__).resolve().parent.parent;W=B/'work';sys.path.insert(0,str(W))
from group5_release_transport_v1 import write,digest
prev=json.loads((W/'node20631_official_queue_pointer.json').read_bytes());old=P(prev['payload']);p=json.loads((old/'plan.json').read_bytes());assert not prev['training_dispatched'] and not prev['once_consumed']
now=dt.datetime.now(dt.timezone.utc);r=B/'outputs'/('node20631_iteration_prepared_'+now.strftime('%Y%m%dT%H%M%SZ'));r.mkdir();shutil.copytree(old/'source',r/'source')
for n in ['iterative_method_v1.py','node20631_train_iteration_v1.py','node20631_iteration_supervisor_v1.py']:shutil.copyfile(W/n,r/'source'/n)
for f in (r/'source').glob('*.py'):ast.parse(f.read_text(encoding='utf-8-sig'),filename=f.name)
p.update(execution_enabled=False,protocol='OFFICIAL_FIXED_SPLIT_VAL_ONLY_ITERATION',human_confirmation='不用去做之前内容的五折验证，只管把方法不断更新迭代',seeds=[4],methods=['careflow','adaptive_gated_aux','adaptive_gated_residual'],source_SHA={f.name:digest(f) for f in (r/'source').glob('*.py')},previous_five_seed_frozen_plan_SHA=prev['plan_SHA'],previous_five_seed_plan_never_dispatched=True,TEST_scoring_enabled=False,iteration_stage=1,scientific_execution_qualification=False)
p['training_recipe']['test']='No TEST role indexed, inferred or scored during iteration; trusted whole pickle materializes all roles; final overall protocol requires separate freeze before TEST'
p['admission']='Real training and first VAL pass before next process; max3; GPU/RAM/space/lease only, no TEST'
write(r/'prepared_plan.json',p);write(r/'prior_not_executed_pointer.json',prev)
(r/'实验协议.md').write_text('最新人类指令停止旧方案五折验证，转方法迭代。本阶段仅官方划分、seed4三组：CaReFlow必要参照；输入自适应方向门控+MSE辅助；同门控加有界残差读出。门控仅输入特征，训练时10%消息方向随机关闭；残差幅度上界0.5、零初始化。候选是预先固定的工程假设，不能声称已提升或机制成立。原TRAIN统计/两步消息骨干保持；MSE对齐VAL选择目标，原符号/强度辅助损失保留。每组100轮4000更新，VAL最早最优选择。迭代不索引TEST角色、不推理评分TEST；整个可信pickle会实体化其他角色字节，不能称从未接触其他角色内存。先保留所有负结果，再依据VAL决定下一有界版本，最终方案单独冻结后才TEST一次。此前15组计划只保存未派发，旧once不消费。新小CPU合成检查必须独立通过，不能冒实际模型训练资格；实际首40更新和首次VAL检查后增加并行。\n',encoding='utf8')
z=r/'iteration_bundle.zip'
with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED) as f:
 for path in (r/'source').iterdir():f.write(path,'source/'+path.name)
 for n in ['prepared_plan.json','实验协议.md']:f.write(r/n,n)
ptr=dict(root=str(r),source=str(r/'source'),bundle=str(z),bundle_SHA=digest(z),remote=prev['remote'].rsplit('/',1)[0]+'/iteration1',training_dispatched=False,once_consumed=False,prior_plan=prev['plan_SHA'],prior_queue_not_dispatched=True,protocol=p['protocol'])
write(W/'node20631_iteration_pointer.json',ptr);print(json.dumps(ptr))
