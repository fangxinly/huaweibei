"""Build a source-only TRAIN/DEV adapter candidate. Never imports Torch or opens data arrays."""
import argparse,ast,hashlib,json,math,shutil,zipfile
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=argparse.ArgumentParser();p.add_argument('--clock-utc',required=True);a=p.parse_args()
stamp=a.clock_utc[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
b=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');o=b/'outputs'
src=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen/outputs/careflow_reproduction/official/train_reflow_new.py')
assert sha(src)=='d0266a55931fbae4329c2123b3b0f0d475799f31b8e202fa393ecef406ce841e'
backup=Path('D:/CodexBackups/selective_flow_20261003_1105');assert shutil.disk_usage(b).free>6*1024**3 and shutil.disk_usage(backup).free>6*1024**3
root=b/'work'/('official_train_dev_source_adapter_preparation_'+stamp);D=backup/root.name
assert not root.exists() and not D.exists();root.mkdir();D.mkdir()
text=src.read_text(encoding='utf-8');tree=ast.parse(text);lines=text.splitlines(keepends=True)
excluded={'set_up_data_loader','test_epoch','test_score_model','train','main'}
kept=[];removed=[];kept_functions={}
for n in tree.body:
    why=None
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in excluded:why='Excluded dataset/Test orchestration'
    elif isinstance(n,ast.If) and any(isinstance(q,ast.Name) and q.id=='__name__' for q in ast.walk(n.test)):why='Excluded executable main entry'
    elif isinstance(n,ast.Import) and any(q.name=='pickle' for q in n.names):why='Excluded direct dataset deserialization import'
    elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and any(isinstance(q,ast.Constant) and q.value=='--test_batch_size' for q in ast.walk(n)):why='Excluded TEST CLI option'
    segment=''.join(lines[n.lineno-1:n.end_lineno])
    if why:removed.append({'line':n.lineno,'end_line':n.end_lineno,'name':getattr(n,'name',None),'reason':why})
    else:
        kept.append(segment)
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):kept_functions[n.name]=hashlib.sha256(segment.encode('utf-8')).hexdigest()
adapter='"""Source-only derived module: no dataset loading or TEST orchestration; runtime not yet validated."""\n\n'+'\n\n'.join(kept)+'\n'
adapter_path=root/'careflow_train_dev_author_components_candidate.py';adapter_path.write_text(adapter,encoding='utf-8')
at=ast.parse(adapter);compile(at,str(adapter_path),'exec')
assert not any(isinstance(n,ast.FunctionDef) and n.name in excluded for n in ast.walk(at))
assert not any(isinstance(n,ast.Name) and n.id in ('pickle','test_data','test_dataset','test_dataloader','test_score_model') for n in ast.walk(at))
expected={'InputFeatures','convert_to_features','prepare_deberta_input','get_tokenizer','get_appropriate_dataset','set_random_seed','prep_for_training','batch_minmax','train_epoch','_forward_eval','eval_epoch','multiclass_acc'}
assert set(kept_functions)==expected
for n in at.body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
        original=next(q for q in tree.body if isinstance(q,type(n)) and q.name==n.name)
        assert ast.dump(n,include_attributes=False)==ast.dump(original,include_attributes=False)
common='''"""Preparation only: validated row permutations and author batch-MSE selection."""
import math
TRAIN_ROWS=1281
TRAIN_BATCH=32
EPOCHS=100
DEV_ROWS=229
DEV_BATCH=128
UPDATES_PER_EPOCH=40
TOTAL_UPDATES=4000

def train_batches(frozen_order):
    order=tuple(frozen_order)
    if len(order)!=TRAIN_ROWS or any(type(i) is not int for i in order) or set(order)!=set(range(TRAIN_ROWS)):
        raise ValueError('Expected complete unique official TRAIN positional permutation')
    used=UPDATES_PER_EPOCH*TRAIN_BATCH
    return tuple(order[i:i+TRAIN_BATCH] for i in range(0,used,TRAIN_BATCH)), order[used:]

def author_dev_batch_mse(batch_losses, batch_counts):
    losses=tuple(float(x) for x in batch_losses)
    if tuple(batch_counts)!=(128,101) or len(losses)!=2:
        raise ValueError('Expected fixed official DEV order and batches128/101')
    if any(not math.isfinite(x) or x<0 for x in losses):
        raise ValueError('Invalid DEV batch MSE')
    return sum(losses)/2

def improves_earliest(best, candidate):
    if not math.isfinite(candidate):
        raise ValueError('Nonfinite candidate')
    return candidate<best
'''
(root/'common_budget_selection_candidate.py').write_text(common,encoding='utf-8');ns={};exec(compile(common,'synthetic_common_budget','exec'),ns)
batches,omitted=ns['train_batches'](range(1281));assert len(batches)==40 and all(len(x)==32 for x in batches) and omitted==(1280,)
assert sum(len(x) for x in batches)==1280 and 100*len(batches)==4000
for invalid in [list(range(1280)),list(range(1280))+[0],list(range(1280))+[True]]:
    try:ns['train_batches'](invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid shared positional order accepted')
assert ns['author_dev_batch_mse']((0,100),(128,101))==50
assert not ns['improves_earliest'](1,1) and ns['improves_earliest'](1,.5)
for loss,count in [((0,100),(101,128)),((0,float('nan')),(128,101))]:
    try:ns['author_dev_batch_mse'](loss,count)
    except ValueError:pass
    else:raise AssertionError('Wrong/nonfinite DEV score accepted')
prior_guard=b/'work/official_baseline_artifact_guard_preparation_20261007T000208TZ/train_dev_access_guard_preparation.py'
shutil.copy2(prior_guard,root/prior_guard.name)
receipt={'status':'LOCAL_TRAIN_DEV_SOURCE_ADAPTER_AND_COMMON_BUDGET_PREPARATION_NOT_FORMAL_FREEZE',
         'clock_start_utc':a.clock_utc,'pinned_original_source_sha256':sha(src),'derived_candidate_sha256':sha(adapter_path),
         'kept_component_source_hashes':kept_functions,'removed_top_level_nodes':removed,
         'retained_component_AST_identical_to_pinned_source':True,'AST_compile_only_no_Torch_import':True,
         'synthetic_common_budget_guard_passed':True,'candidate_updates':4000,'candidate_drop_last':True,
         'candidate_DEV_selection':'Arithmetic mean of batch128/101 MSE, strict improvement and earliest tie',
         'official_positional_orders_generated_or_frozen':False,'official_row_ID_manifest_frozen':False,
         'single_method_formally_locked':False,'complete_runner_or_capture_frozen':False,
         'runtime_model_label_dummy_invariance_passed':False,'model_or_asset_or_true_labels_loaded':False,
         'formal_benchmark_executed':False,'new_score':False,
         'remaining':['One method independent of201 ranking plus common public-asset/official-role/ID/scale freeze',
                      'Physical shared epoch orders before training, linked to official TRAIN IDs and a declared isolated RNG source',
                      'Complete fresh-initialization runners for both methods, full-state preservation and original CPU/capture source',
                      'Actual runtime label-dummy invariance, complete state replay, cumulative resource/time/save-budget gates',
                      'Both methods and prediction/selection protocol frozen before one final TEST evaluation; historical baseline TEST disclosed']}
write(root/'local_source_adapter_receipt.json',receipt)
doc='''# 正式TRAIN/DEV来源与共同预算：局部适配准备

从固定CaReFlow训练源提取候选组件，保留tokenization、数据转换、模型/AdamW/scheduler构造、每批训练、batch-minmax和DEV批MSE函数的原AST；排除原pickle加载器、每轮TEST函数/训练总控/main及TEST CLI参数。原冻结源码未改。此候选没有启动入口或完整runner，编译只检查AST，未导入Torch或加载模型。

共同预算候选按既有固定baseline：TRAIN1281、batch32、drop_last=True、100轮×40正式更新=4000；每轮完整1281索引排列后丢尾1。DEV按固定顺序128/101两批MSE算术平均严格改善选模且平局最早。纯合成检查通过重复/短/非整数排列拒绝、尾批数量、错误DEV批结构/非有限MSE拒绝、严格平局保持。未生成真实官方ID/共同订单，也未把候选预算冒双方实际训练预算。

下一必须完成单方法固定与官方TRAIN/DEV-only角色/ID/尺度/公共资产，完整双方runner/共有实际订单/全状态/CPU与capture协议及5分钟资源和2h保存门；模型DEV真值与dummy替换不变须实际验证。未将201九读出排名用于方法选择，未读取pickle/NPZ/真实标签，没有新分数或正式TEST。旧baseline完整字节D保存通过不替代新协议下的完整模型/订单资格。
'''
(root/'preparation.md').write_text(doc,encoding='utf-8')
shutil.copytree(root,D/'source_candidate');shutil.copy2(__file__,D/Path(__file__).name)
for ext,n in [('md','preparation.md'),('json','local_source_adapter_receipt.json')]:shutil.copy2(root/n,o/('正式TRAIN_DEV共同预算源适配准备接续.'+ext))
pro=o/'正式CaReFlow比较来源与预算本地准备接续.json';v=json.loads(pro.read_text(encoding='utf-8'));v['latest_train_dev_source_adapter_candidate']={'D':str(D),'local':str(root),'status':receipt['status'],'clock_start_utc':a.clock_utc};write(pro,v)
short=o/'研究接续状态.md';previous=short.read_text(encoding='utf-8')
short.write_text('本地准备实际clock '+a.clock_utc+'：固定CaReFlow组件AST保真提取去除TEST总控/加载器；共同4000更新/drop尾1及DEV128/101批MSE严格最早选择仅候选，合成排列/结构/有限数守卫通过。未生成官方ID订单、未导Torch/模型/标签/新分数；完整双方runner/实际资源/capture协议未冻。先读正式TRAIN_DEV共同预算源适配准备接续.md/json及正式CaReFlow比较来源与预算本地准备接续.md/json。原baseline完整字节已D保存非正式cache重放通过；16建议failed不轮询/重发；五项目标与租期保存未完成。\n\n'+previous,encoding='utf-8')
control=D/'control';control.mkdir()
for n in ['研究接续状态.md','正式CaReFlow比较来源与预算本地准备接续.json','正式TRAIN_DEV共同预算源适配准备接续.md','正式TRAIN_DEV共同预算源适配准备接续.json']:shutil.copy2(o/n,control/n)
members={str(x.relative_to(D)).replace('\\','/'):sha(x) for x in D.rglob('*') if x.is_file()};write(D/'member_SHA.json',members)
with zipfile.ZipFile(D/'snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
    for n in [*members,'member_SHA.json']:z.write(D/n,n)
with zipfile.ZipFile(D/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)+1
    for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
write(D/'preservation_receipt.json',{'clock_start_utc':a.clock_utc,'ZIP_SHA':sha(D/'snapshot.zip'),'members':len(members)+1,'CRC_unique_all_SHA':True,'scope':'Local source candidate only; no remote capture or experiment'})
print(json.dumps({'D':str(D),'status':receipt['status'],'retained_components':len(kept_functions),'members':len(members)+1},ensure_ascii=False))
