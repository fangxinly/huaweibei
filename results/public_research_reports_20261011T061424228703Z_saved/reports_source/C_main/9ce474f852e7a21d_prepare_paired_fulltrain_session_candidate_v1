"""Candidate constructors/role gate/common positional orders only; no models or official arrays loaded."""
import argparse,ast,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=argparse.ArgumentParser();p.add_argument('--clock-utc',required=True);a=p.parse_args()
stamp=a.clock_utc[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
b=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');o=b/'outputs';db=Path('D:/CodexBackups/selective_flow_20261003_1105')
assert shutil.disk_usage(b).free>6*1024**3 and shutil.disk_usage(db).free>6*1024**3
root=b/'work'/('paired_fulltrain_session_candidate_'+stamp);D=db/root.name;assert not root.exists() and not D.exists();root.mkdir();D.mkdir()
parent=b/'work/minimal_fixed_staged_reference_v1_20261006T151117Z'
prior=b/'work/official_train_dev_source_adapter_preparation_20261007T000857Z'
plan=json.loads((parent/'runtime_candidate_plan.json').read_text(encoding='utf-8'))
copied={}
for name in ['encoder_adapter.py','minimal_fixed_flow_v2.py','legacy_flow_model.py','finite_single_token_reader_v1.py']:
 assert sha(parent/name)==plan['source_sha256'][name];shutil.copy2(parent/name,root/name);copied[name]=sha(root/name)
for name in ['careflow_train_dev_author_components_candidate.py','common_budget_selection_candidate.py']:
 shutil.copy2(prior/name,root/name);copied[name]=sha(root/name)
runtime=parent/'minimal_fixed_runtime_v1.py';assert sha(runtime)==plan['source_sha256'][runtime.name]
rt=ast.parse(runtime.read_text(encoding='utf-8'));lines=runtime.read_text(encoding='utf-8').splitlines(keepends=True)
parts=['import hashlib\nimport torch\nfrom encoder_adapter import content_mask, encode_masked\nfrom minimal_fixed_flow_v2 import objective\n']
for name in ('tensor_sha','forward_fixed','forward_batch','fit_objective'):
 node=next(x for x in rt.body if isinstance(x,ast.FunctionDef) and x.name==name)
 parts.append(''.join(lines[node.lineno-1:node.end_lineno]))
(root/'fixed_flow_components_candidate.py').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
guard='''"""Approved official TRAIN/DEV roles; labels become available only at explicit gates."""
import hashlib
from pathlib import Path
import numpy as np

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def canonical_id(value):
    if isinstance(value,bytes):return value.decode('utf-8')
    if isinstance(value,str):return value
    raise ValueError('Official segment ID must be a string or UTF8 bytes')

class FullTrainDevGuard:
    def __init__(self,train_records,dev_records):
        if len(train_records)!=1281 or len(dev_records)!=229:
            raise ValueError('Official TRAIN/DEV counts mismatch')
        self._records={'train':train_records,'dev':dev_records}
        self.ids={r:tuple(canonical_id(x[2]) for x in records) for r,records in self._records.items()}
        if any(len(set(ids))!=len(ids) for ids in self.ids.values()) or set(self.ids['train'])&set(self.ids['dev']):
            raise ValueError('Duplicate or overlapping official IDs')
        self.journal=[]

    def inputs_only(self,role,dummy=0):
        if role not in ('train','dev'):
            raise PermissionError('Unapproved role')
        records=[(r[0],np.full((1,1),dummy,dtype=np.float32),r[2]) for r in self._records[role]]
        self.journal.append({'role':role,'purpose':'inputs_only','labels_read':False,'rows':len(records)})
        return records

    def train_supervision(self):
        self.journal.append({'role':'train','purpose':'fullTRAIN_fitting','labels_read':True,'rows':1281})
        return list(self._records['train'])

    def dev_labels_after_frozen_prediction(self,path,expected_sha,state_sha):
        if len(expected_sha)!=64 or file_sha(path)!=expected_sha:
            raise PermissionError('DEV physical prediction SHA not frozen')
        if len(state_sha)!=64 or any(c not in '0123456789abcdef' for c in state_sha):
            raise PermissionError('Model state SHA invalid')
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'row_ids','prediction','model_state_sha256'}:
                raise PermissionError('DEV prediction schema mismatch')
            if tuple(z['row_ids'].tolist())!=self.ids['dev'] or str(z['model_state_sha256'].item())!=state_sha:
                raise PermissionError('DEV row/state identity mismatch')
            if z['prediction'].shape!=(229,) or not np.isfinite(z['prediction']).all():
                raise PermissionError('DEV prediction finite shape mismatch')
        values=np.asarray([np.asarray(r[1]).reshape(-1)[0] for r in self._records['dev']],dtype=np.float64)
        if not np.isfinite(values).all():raise ValueError('Nonfinite DEV target')
        self.journal.append({'role':'dev','purpose':'checkpoint_selection_only','labels_read':True,
                             'prediction_sha256':expected_sha,'model_state_sha256':state_sha,'rows':229})
        return values
'''
(root/'official_fulltrain_dev_guard_candidate.py').write_text(guard,encoding='utf-8')
construct='''"""Prospective paired constructors, no CLI or training. Runtime/preflight/full protocol remain pending."""
import importlib.util,json,pickle,sys
from pathlib import Path
from types import MethodType,SimpleNamespace
import torch
from encoder_adapter import fit_statistics,install
from minimal_fixed_flow_v2 import MinimalFixedFlow
from fixed_flow_components_candidate import tensor_sha,forward_fixed,forward_batch,fit_objective
from official_fulltrain_dev_guard_candidate import FullTrainDevGuard,file_sha

METHODS=('careflow','minimal_fixed_F')
REMOVED_AUTHOR=('reflow_a','reflow_v','reflow_a_b','reflow_v_b','rf_a','rf_v','rf_a_b','rf_v_b','pooler')

def check_sources_assets(bundle,assets,plan):
    for rel,h in plan['source_sha256'].items():
        if file_sha(bundle/rel)!=h:raise ValueError('Candidate source mismatch: '+rel)
    for rel,h in plan['asset_sha256'].items():
        if file_sha(assets/rel)!=h:raise ValueError('Public asset mismatch: '+rel)

def load_author_components(bundle,assets):
    original_argv=list(sys.argv)
    args=['careflow_train_dev_author_components_candidate.py','--model',str(assets/'assets/deberta-v3-base'),
          '--dataset','mosi','--n_epochs','100','--train_batch_size','32','--dev_batch_size','128',
          '--learning_rate','1e-5','--ratio','4','--dropout_prob','0.5','--inter_dim','150',
          '--share_dim','100','--transformer_layer','3','--step_size','2','--loss_b_ratio','0.1',
          '--loss_f_ratio','0.2','--eps','1e-3','--seed','128','--gradient_accumulation_step','1']
    sys.path.insert(0,str(assets/'assets/CaReFlow'))
    spec=importlib.util.spec_from_file_location('paired_author_train_dev_components',bundle/args[0])
    author=importlib.util.module_from_spec(spec)
    try:
        sys.argv=args;spec.loader.exec_module(author)
    finally:sys.argv=original_argv
    return author,args[1:]

def load_approved_official_roles(assets):
    # Trusted whole pickle materializes other-role bytes; only these two entries are indexed.
    # Runtime does not claim that unapproved bytes never existed in memory.
    with (assets/'assets/mosi.pkl').open('rb') as f:container=pickle.load(f)
    train,dev=container['train'],container['dev'];del container
    return FullTrainDevGuard(train,dev)

def verify_public_encoder(model,backbone):
    original=torch.load(backbone/'pytorch_model.bin',map_location='cpu')
    matched=0
    for name,t in model.dberta.model.state_dict().items():
        key=name if name in original else 'deberta.'+name
        if key in original:
            if not torch.equal(t.detach().cpu(),original[key].to(dtype=t.dtype)):
                raise ValueError('Public encoder initialization mismatch')
            matched+=1
    if matched<190:raise ValueError('Insufficient public encoder coverage')
    return matched

def new_optimizer(author,model):
    no_decay=('bias','LayerNorm.bias','LayerNorm.weight')
    named=list(model.named_parameters())
    groups=[{'params':[p for n,p in named if not any(x in n for x in no_decay)],'weight_decay':.01},
            {'params':[p for n,p in named if any(x in n for x in no_decay)],'weight_decay':0.0}]
    optimizer=torch.optim.AdamW(groups,lr=1e-5)
    ids=[id(p) for g in optimizer.param_groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters()}:
        raise ValueError('Optimizer duplicate/incomplete coverage')
    scheduler=author.get_linear_schedule_with_warmup(optimizer,num_warmup_steps=400,num_training_steps=4000)
    return optimizer,scheduler

def construct(method,bundle,assets,plan):
    if method not in METHODS:raise ValueError('Undeclared method')
    if plan.get('status')!='PAIRED_FULLTRAIN_RUNTIME_EXECUTION_PROTOCOL_FROZEN':
        raise PermissionError('Preparation candidate cannot construct a model')
    if plan.get('task_seed')!=128 or plan.get('orders_seed')!=128 or plan.get('updates')!=4000:
        raise ValueError('Common prospective budget/seed mismatch')
    bundle,assets=Path(bundle),Path(assets)
    check_sources_assets(bundle,assets,plan)
    author,arguments=load_author_components(bundle,assets);author.set_random_seed(128)
    guard=load_approved_official_roles(assets)
    train_inputs=author.get_appropriate_dataset(guard.inputs_only('train'))
    dev_inputs=author.get_appropriate_dataset(guard.inputs_only('dev'))
    model,temp_optimizer,temp_scheduler=author.prep_for_training(4000)
    matched=verify_public_encoder(model,assets/'assets/deberta-v3-base')
    if method=='minimal_fixed_F':
        stats=fit_statistics(train_inputs)
        for name in REMOVED_AUTHOR:
            if not hasattr(model.dberta,name):raise ValueError('Missing original component: '+name)
            delattr(model.dberta,name)
        model.dberta.own_flow=MinimalFixedFlow();install(model.dberta,stats)
        for encoder in (model.dberta.transa,model.dberta.transv):encoder.embed_positions._float_tensor.zero_()
        model.dberta.forward=MethodType(forward_fixed,model.dberta)
    else:stats=None
    model.to(author.DEVICE)
    del temp_optimizer,temp_scheduler
    for param in model.parameters():param.requires_grad_(True)
    optimizer,scheduler=new_optimizer(author,model)
    if optimizer.state:raise ValueError('Optimizer inherited state')
    train=author.get_appropriate_dataset(guard.train_supervision())
    clean_initial={'model':{n:t.detach().cpu().clone() for n,t in model.state_dict().items()},
                   'torch_rng':torch.get_rng_state().clone(),'cuda_rng':[r.clone() for r in torch.cuda.get_rng_state_all()]}
    receipt={'method':method,'fullTRAIN_rows':1281,'DEV_input_rows':229,'task_seed':128,
             'public_encoder_matched_tensors':matched,'optimizer_steps':0,
             'task_initialization':'Public pretrained backbone and new random task modules; no old task checkpoint',
             'parameters':sum(p.numel() for p in model.parameters()),'parameter_tensors':len(list(model.parameters())),
             'initial_state_sha256':tensor_sha(clean_initial['model']),
             'normalization':'TRAIN1281 fitted masked channels' if stats is not None else 'Author per-batch minmax',
             'same_seed_does_not_mean_identical_task_state_across_architectures':True,
             'guard_journal':guard.journal}
    return SimpleNamespace(model=model,optimizer=optimizer,scheduler=scheduler,author=author,guard=guard,
                           train=train,train_inputs=train_inputs,dev_inputs=dev_inputs,
                           statistics=stats,clean_initial=clean_initial,construction_receipt=receipt,arguments=arguments)

def training_loss(session,method,batch):
    if method=='minimal_fixed_F':return fit_objective(session.model,batch)
    if method!='careflow':raise ValueError('Undeclared method')
    logits,lf,lb=session.author._forward_eval(session.model,batch)
    return (logits.view(-1)-batch[3].view(-1)).square().mean()+.2*lf+.1*lb

def predictions(session,method,batch):
    if method=='minimal_fixed_F':return forward_batch(session.model,batch)
    if method=='careflow':return session.author._forward_eval(session.model,batch)[0].view(-1)
    raise ValueError('Undeclared method')
'''
(root/'paired_fulltrain_session_candidate.py').write_text(construct,encoding='utf-8')
for x in root.glob('*.py'):compile(ast.parse(x.read_text(encoding='utf-8')),str(x),'exec')
# Only the pure NumPy role guard is exercised. Sentinels make any label access fail.
ns={};exec(compile(guard,'synthetic_role_guard','exec'),ns)
class InputOnly:
 def __init__(self,key):self.key=key
 def __getitem__(self,i):
  if i==0:return ('synthetic_input',)
  if i==2:return self.key
  if i==1:raise AssertionError('Synthetic label touched before access gate')
  raise IndexError(i)
train=[InputOnly('train:'+str(i)) for i in range(1281)];dev=[InputOnly('dev:'+str(i)) for i in range(229)]
g=ns['FullTrainDevGuard'](train,dev);assert len(g.inputs_only('train'))==1281 and len(g.inputs_only('dev',7))==229
for bad in ('test','headEVAL201','inner'):
 try:g.inputs_only(bad)
 except PermissionError:pass
 else:raise AssertionError('Unapproved role accepted')
try:ns['FullTrainDevGuard'](train,[InputOnly('train:0')]+dev[1:])
except ValueError:pass
else:raise AssertionError('Overlapping row IDs accepted')
try:g.dev_labels_after_frozen_prediction(root/'absent.npz','', '0'*64)
except PermissionError:pass
else:raise AssertionError('Unfrozen prediction gate accepted')
rng=np.random.Generator(np.random.PCG64(128));orders=np.stack([rng.permutation(1281) for _ in range(100)]).astype(np.int64)
assert orders.shape==(100,1281) and all(np.array_equal(np.sort(x),np.arange(1281)) for x in orders)
np.save(root/'prospective_common_TRAIN_position_orders_seed128_100.npy',orders,allow_pickle=False)
orders_info={'status':'PROSPECTIVE_CANDIDATE_POSITIONAL_ORDERS_SAVED_NOT_OFFICIAL_ROW_ID_QUALIFIED',
             'shape':[100,1281],'seed':128,'generator':'NumPy Generator PCG64 isolated from task RNG',
             'numpy_version':np.__version__,'updates_per_epoch':40,'batch':32,'dropped_tail':1,'updates_total':4000,
             'sha256':sha(root/'prospective_common_TRAIN_position_orders_seed128_100.npy'),
             'shared_for_both_future_methods':True,'official_TRAIN_identity_mapping_checked':False,
             'matches_old_cached_baseline_real_orders':False,'selection_from_201_scores':False}
write(root/'prospective_orders_receipt.json',orders_info)
receipt={'status':'LOCAL_PAIRED_FULLTRAIN_SESSION_SOURCE_CANDIDATE_NOT_EXECUTION_FREEZE',
         'clock_start_utc':a.clock_utc,'local':str(root),'D':str(D),'task_seed_candidate':128,
         'method_candidate':'Same previously specified minimal fixed full-flow F, direct prediction; no new residual head/readout',
         'candidate_seed_rationale':'One common author-default seed128 prospective comparison, no seed sweep or201 selection',
         'old_completed_fold_seed91819_weights_reused':False,'method_architecture_parent_source_SHA':plan['source_sha256'],
         'copied_exact_source_SHA':copied,'AST_compile_passed':True,'synthetic_no_label_role_guard_passed':True,
         'prospective_positional_orders':orders_info,'source_sha256':{x.name:sha(x) for x in root.glob('*.py')},
         'asset_sha256':plan['asset_sha256'],'official_data_or_labels_decoded':False,'Torch_imported_or_GPU_executed':False,
         'formal_training_or_TEST_permitted':False,'end_to_end_runner_precheck_or_capture_implemented':False,
         'remaining':['Full train/precheck runners plus source/runtime independent audits, natural wrappers and root-specific capture',
                      'Actual official row-ID/scale/public-asset verification, shared positional-order binding and DEV dummy invariance',
                      'Cumulative construction/Adam/save/replay GPU peak, actual complete space and execute plus2h reserve gates',
                      'Full protocol freeze before new labels or training, each clean initial reset and full state preservation',
                      'Both final selected pipelines locked before final paired evaluation; old baseline TEST access disclosed']}
write(root/'local_paired_session_candidate_receipt.json',receipt)
doc='''# 正式双方fullTRAIN：构造与角色门候选

候选固定对比方向是既有完整minimal fixed F本身，不增加201校正头、不按九读出排名挑部署输出。共同作者默认seed128作为未来单次比较候选，未扫描seed；与旧fold91819不同，不复用旧任务权重、优化器或预检状态，也不声称相同seed产生相同架构初态。尚未正式执行冻结。

双方构造候选只索引可信官方pickle的TRAIN/DEV，未构造其他角色；可信整pickle反序列化会将其他split字节带入内存，不冒它们不存在。官方ID要独立唯一/不交叠，DEV先dummy原输入预测物理SHA与state/ID关联，再访问229标签用于每轮选模。纯合成哨兵已核输入入口不碰真标签、未知角色/重叠ID及未冻预测拒绝；尚无真实数据/模型验证。

共同100×1281位置订单已以独立PCG64 seed128保存SHA，每轮同40批×32并丢尾1，共4000更新，供未来双方共用。尚未绑定真实官方TRAIN行ID，不冒旧缓存订单一致。正式runtime还须锁定共同DEV128/101批MSE严格最早规则、完整训练/预检/自然退出/CPU/capture、actual资产/空间/累计GPU峰值与至少2h保存门。

新构造组件编译仅AST。自有核心和归一化/单token解析模块按已冻源码字节复制；统计将仅来自fullTRAIN1281原输入，旧fold695统计不沿用。CaReFlow保留原batch-minmax和.2/.1辅助目标，自有F保持既有完整目标；这些方法差异如实保留。未Torch导入、真实标签/NPZ/pickle访问、远程连接、新训练或分数。
'''
(root/'preparation.md').write_text(doc,encoding='utf-8')
shutil.copytree(root,D/'source_candidate');shutil.copy2(__file__,D/Path(__file__).name)
for ext,name in [('md','preparation.md'),('json','local_paired_session_candidate_receipt.json')]:shutil.copy2(root/name,o/('正式双方fullTRAIN构造与角色门准备接续.'+ext))
pro=o/'正式CaReFlow比较来源与预算本地准备接续.json';v=json.loads(pro.read_text(encoding='utf-8'));v['latest_paired_fulltrain_session_candidate']={'status':receipt['status'],'D':str(D),'local':str(root),'clock_start_utc':a.clock_utc};write(pro,v)
short=o/'研究接续状态.md';old=short.read_text(encoding='utf-8');short.write_text('本地准备实际clock '+a.clock_utc+'：双方fullTRAIN构造/角色门仅候选，既有minimal fixed F不加201读出；共同seed128一次候选非扫seed，PCG64位置订单100×1281同40批/丢尾1已存SHA但未绑定官方ID。原核心源码字节复制，标签哨兵/禁止角色/重叠ID/未冻预测门与AST过，未Torch/真实数据/模型/新成绩。完整训练预检自然wrapper/CPU/capture及资源门未齐、不允许启动。先读正式双方fullTRAIN构造与角色门准备接续.md/json；原201负结果/旧整件保留，整体五项目标与租期保存未完成。\n\n'+old,encoding='utf-8')
control=D/'control';control.mkdir()
for n in ['研究接续状态.md','正式CaReFlow比较来源与预算本地准备接续.json','正式双方fullTRAIN构造与角色门准备接续.md','正式双方fullTRAIN构造与角色门准备接续.json']:shutil.copy2(o/n,control/n)
members={str(x.relative_to(D)).replace('\\','/'):sha(x) for x in D.rglob('*') if x.is_file()};write(D/'member_SHA.json',members)
with zipfile.ZipFile(D/'snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in [*members,'member_SHA.json']:z.write(D/n,n)
with zipfile.ZipFile(D/'snapshot.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)+1
 for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
write(D/'preservation_receipt.json',{'clock_start_utc':a.clock_utc,'ZIP_SHA':sha(D/'snapshot.zip'),'members':len(members)+1,'CRC_unique_all_SHA':True,'scope':'Source/positional orders preparation; no actual model, labels, remote capture or training'})
print(json.dumps({'D':str(D),'members':len(members)+1,'status':receipt['status'],'candidate_orders_sha':orders_info['sha256']},ensure_ascii=False))
