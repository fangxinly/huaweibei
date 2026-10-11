"""Preserve compiled runtime source and actual local rejection fixture; no GPU or real data execution."""
import argparse,ast,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=argparse.ArgumentParser();p.add_argument('--clock-utc',required=True);a=p.parse_args()
stamp=a.clock_utc[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
b=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');o=b/'outputs';backup=Path('D:/CodexBackups/selective_flow_20261003_1105')
assert shutil.disk_usage(b).free>6*1024**3 and shutil.disk_usage(backup).free>6*1024**3
parent=b/'work/paired_fulltrain_session_candidate_20261007T001943Z'
root=b/'work'/('paired_fulltrain_runtime_preparation_'+stamp);D=backup/root.name
assert not root.exists() and not D.exists();root.mkdir();D.mkdir()
for x in parent.iterdir():
 if x.is_file():shutil.copy2(x,root/x.name)
for name in ['paired_fulltrain_runtime_candidate_v1.py','paired_fulltrain_natural_wrapper_candidate_v1.py']:
 shutil.copy2(b/'work'/name,root/name)
source={}
for x in root.glob('*.py'):
 s=x.read_text(encoding='utf-8');tree=ast.parse(s);compile(tree,str(x),'exec')
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('kill','terminate','reset_peak_memory_stats') for n in ast.walk(tree))
 source[x.name]=sha(x)
fixture=root/'local_rejection_fixture';fixture.mkdir()
write(fixture/'paired_fulltrain_execution_plan.json',{'status':'LOCAL_SOURCE_PREPARATION_NOT_EXECUTION_FREEZE'})
forbidden_root=Path('/data/coding/paired_fulltrain_local_negative_fixture_'+stamp)
assert not forbidden_root.exists()
argv=[sys.executable,str(root/'paired_fulltrain_runtime_candidate_v1.py'),'--root',str(forbidden_root),
      '--bundle',str(fixture),'--assets',str(fixture),'--method','minimal_fixed_F','--stage','precheck',
      '--plan-sha',sha(fixture/'paired_fulltrain_execution_plan.json')]
result=subprocess.run(argv,capture_output=True,text=True,encoding='utf-8')
assert result.returncode!=0 and 'Local preparation cannot launch' in result.stderr and not forbidden_root.exists()
(fixture/'actual_expected_rejection.stdout.txt').write_text(result.stdout,encoding='utf-8')
(fixture/'actual_expected_rejection.stderr.txt').write_text(result.stderr,encoding='utf-8')
write(fixture/'actual_expected_rejection_receipt.json',{'clock_start_utc':a.clock_utc,'argv':argv,'exit_code':result.returncode,
      'expected_rejection':True,'no_root_created':True,'refused_before_Torch_or_model_or_real_asset_access':True,
      'not_a_GPU_or_approval_failure':True})
receipt={'status':'LOCAL_PAIRED_PRECHECK_TRAIN_NATURAL_WRAPPER_SOURCE_PREPARATION_NOT_EXECUTION_FREEZE',
         'clock_start_utc':a.clock_utc,'local':str(root),'D':str(D),'source_sha256':source,
         'AST_compilation_passed':True,'actual_LOCAL_status_rejection_before_Torch_passed':True,
         'new_precheck_or_training_executed':False,'Torch_or_model_imported':False,'official_data_or_labels_read':False,
         'original_source_or_completed_training_modified':False,
         'source_behavior':['Direct actual nvidia UUID/empty compute, full source/asset SHA and conservative lease/reserve before constructor',
                            'Precheck2 complete clean initial/model-Adam-scheduler-allRNG plus two-step checkpoint, DEV dummy and original whole-file replay',
                            'Formal100 requires original precheck natural0 and D-otherCPU/capture joint, and fresh exact clean initial model/RNG with emptyAdam/step0',
                            'Same physical100 orders/40 batches/drop1/4000 optimizer+scheduler, original row-ID identity guard',
                            'Every DEV prediction SHA/state/IDs frozen before selection-only target access and strict earliest selection',
                            'Complete final latest model-Adam-scheduler-RNG/history+best-state and own complete best saved/replayed, original CPU/fresh instance replay explicitly pending',
                            'Wrapper only natural wait and actual exit/receipt SHA; no timeout kill or cumulative peak reset'],
         'common_selection_precision_deviation':'float64 batch MSE from FP32 original predictions for both; differs from cached author float32 selection',
         'not_claimed':['ActualGPU budget/gradients/DEV label invariance/officialIDs/checkpoint arrays passed',
                        'Complete end-to-end preservation or formal benchmark source frozen','New five-metric results or final TEST'],
         'remaining':['Root-specific capture and original whole TorchCPU/array/metric audit sources plus joint associator',
                      'Actual officialTRAIN/DEV row-ID asset/scale binding; preserved field relationships require audit',
                      'Fresh selected-best public-instance replay source after natural completed training and originalD/CPU preservation',
                      'Full execution protocol freeze and credible human lease plus fresh exact GPU/assets/space/execution2h gate before any new labels']}
write(root/'local_runtime_preparation_receipt.json',receipt)
doc='''# 正式双方训练/预检/自然退出：源码准备

新增共同预检与100轮训练入口、自然等待wrapper，仅AST编译与本地非冻结协议拒绝检查已通过。该拒绝发生在Torch/模型/真实资产访问前，属于合成保护门预期结果，不是GPU或审批故障。本轮没有新的数据、标签、梯度、训练或分数。

预检将保存完整clean初态与2步后model/Adam/scheduler/所有RNG，并核229 DEV dummy0/7及自己的完整文件重放。正式训练要求原预检自然0与D/异节点CPU/capture总门，再新构造精确初态/全部RNG、空Adam与scheduler0；预检2步绝不冒100轮起点。双方共同物理订单40批/丢尾1、4000真optimizer/scheduler；DEV每轮预测SHA/state/ID先冻，再选模标签。共同批MSE使用float64原FP32预测评分，明确区别于旧缓存float32作者选模，不冒原缓存重现。

结束将保存完整latest model/Adam/scheduler/RNG/history及best-state、完整selected best，自己的整文件重放通过才写原receipt；originalCPU与fresh公共实例重放仍是下一门。wrapper只自然wait/exit原记录，无超时杀进程或peakreset。源码里的资源检查不是实际通过：真实UUID/fullargv/compute/source/assets/完整空间与execute+2h门仍须核。

尚缺root-specific capture、独立TorchCPU/数组/指标audit与总关联源，正式官方行ID/尺度/资产绑定及fresh实例重放。完整执行协议未冻，不能直接启动。既有201结果/旧完整原件不修改，TEST入口不存在于此训练runner；正式一次最终比较另门，五项目标与租期保存仍未完成。
'''
(root/'runtime_preparation.md').write_text(doc,encoding='utf-8')
shutil.copytree(root,D/'source_candidate');shutil.copy2(__file__,D/Path(__file__).name)
for ext,name in [('md','runtime_preparation.md'),('json','local_runtime_preparation_receipt.json')]:shutil.copy2(root/name,o/('正式双方训练预检与自然退出源准备接续.'+ext))
pro=o/'正式CaReFlow比较来源与预算本地准备接续.json';v=json.loads(pro.read_text(encoding='utf-8'));v['latest_paired_precheck_train_runtime_candidate']={'status':receipt['status'],'local':str(root),'D':str(D),'clock_start_utc':a.clock_utc};write(pro,v)
short=o/'研究接续状态.md';previous=short.read_text(encoding='utf-8');short.write_text('本地准备actualclock '+a.clock_utc+'：双方预检/训练/natural wrapper候选AST过，LOCAL非冻结协议真实本地拒绝在Torch前，非GPU/审批失败。共同100×40更新、cleanmodel+allRNG新构造不继承2步、DEV先冻后label、完整best/resume保存仅源码。批MSE共同float64与旧缓存float32不同已声明。尚缺原CPU/root-specific capture/联合与fresh replay以及完整官方ID/资源门，不允许启动；先读正式双方训练预检与自然退出源准备接续.md/json及正式CaReFlow比较来源与预算本地准备接续.md/json。未新模型/标签/成绩，整体目标及租期保存未完成。\n\n'+previous,encoding='utf-8')
control=D/'control';control.mkdir()
for n in ['研究接续状态.md','正式CaReFlow比较来源与预算本地准备接续.json','正式双方训练预检与自然退出源准备接续.md','正式双方训练预检与自然退出源准备接续.json']:shutil.copy2(o/n,control/n)
members={str(x.relative_to(D)).replace('\\','/'):sha(x) for x in D.rglob('*') if x.is_file()};write(D/'member_SHA.json',members)
with zipfile.ZipFile(D/'snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in [*members,'member_SHA.json']:z.write(D/n,n)
with zipfile.ZipFile(D/'snapshot.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)+1
 for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
write(D/'preservation_receipt.json',{'clock_start_utc':a.clock_utc,'ZIP_SHA':sha(D/'snapshot.zip'),'members':len(members)+1,
      'CRC_unique_all_SHA':True,'scope':'Local source and negative fixture, not actualGPU/model/data/capture'})
print(json.dumps({'D':str(D),'members':len(members)+1,'status':receipt['status'],'expected_rejection_exit':result.returncode},ensure_ascii=False))
