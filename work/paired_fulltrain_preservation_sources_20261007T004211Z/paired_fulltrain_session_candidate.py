"""Prospective paired constructors, no CLI or training. Runtime/preflight/full protocol remain pending."""
import importlib.util,json,pickle,sys,hashlib
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
    identity=hashlib.sha256(json.dumps(guard.ids,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if identity!=plan['official_train_dev_ID_identity_sha256']:raise PermissionError('Official ID binding failed before model/TRAIN label access')
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
