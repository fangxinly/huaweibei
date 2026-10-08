"""Exactly official TRAIN1281/DEV229; no TEST entry is indexed here."""
import argparse,hashlib,json,math,os,pickle,random,shutil,sys,time
from pathlib import Path
from types import MethodType,SimpleNamespace
import numpy as np
import torch
from common import sha,read,write,utc,verify
from paired_fulltrain_session_candidate import load_author_components,verify_public_encoder,REMOVED_AUTHOR
from official_fulltrain_dev_guard_candidate import FullTrainDevGuard
from common_budget_selection_candidate import train_batches,author_dev_batch_mse
from encoder_adapter import fit_statistics,install
from fixed_flow_components_candidate import tensor_sha,forward_fixed,forward_batch
from official_upgrade import OfficialUpgrade
from training_health_v1 import gradient_summary,before_update,after_update,original_task_objective
from incremental_message import OriginalTail
from training_health_components_v2 import component_gradients,before_optimizer_step,after_optimizer_step

def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return x
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),cuda=[v.clone() for v in torch.cuda.get_rng_state_all()])
def source_features(core,batch):
    from encoder_adapter import content_mask,encode_masked
    ids,v,a,dummy,mask=batch;v=v.squeeze(1);a=a.squeeze(1);valid=content_mask(mask)
    t=core.LayerNorm_l(core.proj_l(core.model(ids,attention_mask=mask)[0]))*valid[...,None]
    def norm(x,n):return ((x-getattr(core,'v6_'+n+'_mean'))/getattr(core,'v6_'+n+'_std'))*getattr(core,'v6_'+n+'_active')*valid[...,None]
    z=[]
    for x,n in ((a,'a'),(v,'v')):
        h=getattr(core,'proj_'+n)(norm(x,{'a':'audio','v':'visual'}[n]).transpose(1,2)).permute(2,0,1)
        h=getattr(core,'LayerNorm_'+n)(encode_masked(getattr(core,'trans'+n),h,valid).transpose(0,1))*valid[...,None];z.append(h)
    return torch.stack([t]+z,1),valid

def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets);assert p['precheck_steps']==16 and p['storage_transport_qualified'] and p['health_v2_native_qualified'];assert p['task_loss']=='mse';assert p['epochs']==100 and p['updates']==4000
    a.out.mkdir();torch.set_num_threads(2);os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1';start=time.monotonic()
    with (a.assets/'assets/mosi.pkl').open('rb') as f:data=pickle.load(f)
    guard=FullTrainDevGuard(data['train'],data['dev']);del data
    assert {k:list(v) for k,v in guard.ids.items()}==p['official_train_dev_IDs']
    assert not set(s.split('[')[0] for s in guard.ids['train']) & set(s.split('[')[0] for s in guard.ids['dev'])
    author,_=load_author_components(a.bundle,a.assets);author.set_random_seed(128)
    train_inputs=author.get_appropriate_dataset(guard.inputs_only('train'));dev_inputs=author.get_appropriate_dataset(guard.inputs_only('dev'))
    model,tempopt,tempsched=author.prep_for_training(4000);matched=verify_public_encoder(model,a.assets/'assets/deberta-v3-base')
    for encoder in (model.dberta.transa,model.dberta.transv):encoder.embed_positions._float_tensor.zero_()
    stats=fit_statistics(train_inputs)
    for n in REMOVED_AUTHOR:delattr(model.dberta,n)
    model.dberta.own_flow=OfficialUpgrade();install(model.dberta,stats);model.dberta.forward=MethodType(forward_fixed,model.dberta)
    model.cuda();model.dberta.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True});del tempopt,tempsched
    gain=model.dberta.own_flow.gain;special=list(model.dberta.own_flow.message.parameters());specialids={id(v) for v in special}|{id(gain)}
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad];ordinary=[(n,v) for n,v in named if id(v) not in specialids]
    nd=('bias','LayerNorm.bias','LayerNorm.weight')
    groups=[dict(params=[v for n,v in ordinary if not any(k in n for k in nd)],lr=1e-5,weight_decay=.01),dict(params=[v for n,v in ordinary if any(k in n for k in nd)],lr=1e-5,weight_decay=0.),dict(params=[gain],lr=.001,weight_decay=0.),dict(params=[v for v in special if v.requires_grad],lr=.001,weight_decay=.01)]
    opt=torch.optim.AdamW(groups);scheduler=author.get_linear_schedule_with_warmup(opt,num_warmup_steps=400,num_training_steps=4000)
    optimized=[id(v) for g in opt.param_groups for v in g['params']];assert len(set(optimized))==len(optimized) and set(optimized)=={id(v) for _,v in named} and not opt.state
    train=author.get_appropriate_dataset(guard.train_supervision());ytrain=train.tensors[3].view(-1).cpu().numpy();assert len(ytrain)==1281 and np.isfinite(ytrain).all() and (abs(ytrain)<=3).all()
    order=np.load(a.bundle/p['orders_file'],allow_pickle=False);assert sha(a.bundle/p['orders_file'])==p['orders_SHA'] and order.shape==(100,1281)
    for r in order:train_batches([int(x) for x in r])
    info={n:dict(shape=list(v.shape),numel=v.numel(),dtype=str(v.dtype)) for n,v in named};indices={}
    for live,saved in zip(opt.param_groups,opt.state_dict()['param_groups']):
        for v,i in zip(live['params'],saved['params']):indices[str(i)]=next(n for n,q in named if q is v)
    initial=tensor_sha(model.state_dict());write(a.out/'clean_public_initialization.json',dict(actual_UTC=utc(),public_encoder_tensors_matched=matched,initial_state_SHA=initial,model_parameters=sum(v.numel() for v in model.parameters()),optimized_parameters=sum(v.numel() for _,v in named),official_IDs=p['official_train_dev_IDs'],TRAIN_only_normalization=True,old_task_state_loaded=False,optimizer_steps=0,model_source_SHA=p['source_sha256']['official_upgrade.py']))
    np.save(a.out/'TRAIN_targets.npy',ytrain);shutil.copy2(a.bundle/p['orders_file'],a.out/'original_common_orders.npy')
    def budget():
        alloc=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();assert max(alloc,reserved)<=p['GPU_peak_ceiling_bytes'];assert shutil.disk_usage(a.out).free>=p['remote_free_floor_bytes'];return dict(allocated_peak=alloc,reserved_peak=reserved)
    def unused_DEV_predict(dummy=0):
        model.eval();values=[]
        with torch.no_grad():
            for startrow in range(0,229,128):
                batch=[v[startrow:startrow+128].cuda() for v in dev_inputs.tensors];batch[3].fill_(dummy);values.append(forward_batch(model,batch).cpu().numpy())
        return np.concatenate(values).astype(np.float32)
    # A fixed TRAIN probe with no label use; same state/gain, functional message OFF.
    from torch.nn import functional as F
    steps=0; records=[];batches,omitted=train_batches([int(x) for x in order[0]])
    for rows in batches[:16]:
        model.train();opt.zero_grad(set_to_none=True);batch=[v[list(rows)].cuda() for v in train.tensors]
        torch.cuda.synchronize();tick=time.monotonic();pred=forward_batch(model,batch)
        task=F.mse_loss(pred,batch[3].view(-1));penalty=.01*model.dberta.own_flow.last_context.square().mean()
        pieces=component_gradients(model,task,penalty);loss=task+penalty
        assert torch.isfinite(loss);loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for _,v in named)
        record=dict(actual_UTC=utc(),step=steps+1,rows=list(rows),task_loss=float(task.detach()),context_penalty=float(penalty.detach()),component_gradients=pieces,total_gradients=gradient_summary(model),learning_rates=[g['lr'] for g in opt.param_groups],gain=float(model.dberta.own_flow.gain.detach().tanh()),context_RMS=float(model.dberta.own_flow.last_context.detach().square().mean().sqrt()),flow_minus_base_RMS=float((model.dberta.own_flow.last_flow.detach()-model.dberta.own_flow.last_base.detach()).square().mean().sqrt()))
        before=before_optimizer_step(model,opt);opt.step();scheduler.step();steps+=1
        record['actual_updates']=after_optimizer_step(model,before);del before
        torch.cuda.synchronize();record['seconds']=time.monotonic()-tick;records.append(record)
        with (a.out/'TRAIN_health.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        budget();assert time.monotonic()-start < p['precheck_budget_seconds']
    assert steps==16 and scheduler.last_epoch==16
    saved_rng=rng(); saved_sha=tensor_sha(model.state_dict());model.eval()
    probe=[v[list(batches[0])].cuda() for v in train_inputs.tensors];probe[3].fill_(0)
    flow_module=model.dberta.own_flow;message=flow_module.message
    with torch.no_grad():
        on=forward_batch(model,probe).cpu().numpy();base=flow_module.last_base.cpu().numpy();flow=flow_module.last_flow.cpu().numpy()
        forward_saved=message.forward
        message.forward=lambda slots: torch.zeros((len(slots),3,100),device=slots.device,dtype=slots.dtype)
        try: off=forward_batch(model,probe).cpu().numpy()
        finally: message.forward=forward_saved
        probe[3].fill_(7);dummy=forward_batch(model,probe).cpu().numpy()
    assert np.array_equal(on,dummy) and tensor_sha(model.state_dict())==saved_sha
    rr=rng();assert torch.equal(saved_rng['torch'],rr['torch']) and all(torch.equal(x,y) for x,y in zip(saved_rng['cuda'],rr['cuda']))
    np.savez(a.out/'fixed_TRAIN_probe.npz',ids=np.asarray(guard.ids['train'])[list(batches[0])],p=on,p0_same_flow=off,base=base,flow=flow,labels_not_used=True)
    metadata=dict(plan_SHA=a.plan_sha,task_loss='mse',steps=16,next_epoch=0,next_batch=16,scheduler_total_steps=4000,warmup_steps=400,orders_SHA=p['orders_SHA'],parameter_info=info,optimizer_index_to_name=indices,initial_public_state_SHA=initial,final_state_SHA=saved_sha,official_train_dev_IDs=p['official_train_dev_IDs'],normalization='TRAIN1281 only',validation_labels_used=False,TEST_entry_not_indexed=True,not_selected_checkpoint=True)
    state=dict(metadata=metadata,model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=cpu(scheduler.state_dict()),rng=saved_rng,prefix_records=records,omitted_rows=list(omitted))
    cp=a.out/'complete_resume_step16.pt';torch.save(state,cp);assert cp.stat().st_size<=p['checkpoint_bytes_ceiling']
    write(a.out/'guard_journal.json',guard.journal)
    result=dict(status='REAL_TRAIN_MSE_PREFIX16_COMPLETE_NOT_FULL_TRAIN',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,metadata=metadata,checkpoint_SHA=sha(cp),checkpoint_bytes=cp.stat().st_size,probe_dummy_state_RNG_pass=True,peak=budget(),wall_seconds=time.monotonic()-start,continuation_must_count_first16_in4000=True)
    write(a.out/'precheck_result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','out'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--plan-sha',required=True);run(parser.parse_args())
