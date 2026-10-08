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
from candidate_adapter import make_candidate,optimizer_groups,CandidateTail,candidate_objective as objective


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
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets);assert p['native_CPU_D_qualification'] and p['candidate_native_D_B_qualification'] and p['candidate_full_encoder_precheck_D_B_qualification'];assert p['candidate_mode'] in ('regression_aux','factorized_aux');assert p['epochs']==100 and p['updates']==4000
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
    model.dberta.own_flow=make_candidate(p['candidate_mode']);install(model.dberta,stats);model.dberta.forward=MethodType(forward_fixed,model.dberta)
    model.cuda();model.dberta.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True});del tempopt,tempsched
    groups,named=optimizer_groups(model)
    opt=torch.optim.AdamW(groups);scheduler=author.get_linear_schedule_with_warmup(opt,num_warmup_steps=400,num_training_steps=4000)
    optimized=[id(v) for g in opt.param_groups for v in g['params']];assert len(set(optimized))==len(optimized) and set(optimized)=={id(v) for _,v in named} and not opt.state
    train=author.get_appropriate_dataset(guard.train_supervision());ytrain=train.tensors[3].view(-1).cpu().numpy();assert len(ytrain)==1281 and np.isfinite(ytrain).all() and (abs(ytrain)<=3).all()
    order=np.load(a.bundle/p['orders_file'],allow_pickle=False);assert sha(a.bundle/p['orders_file'])==p['orders_SHA'] and order.shape==(100,1281)
    for r in order:train_batches([int(x) for x in r])
    info={n:dict(shape=list(v.shape),numel=v.numel(),dtype=str(v.dtype)) for n,v in named};indices={}
    for live,saved in zip(opt.param_groups,opt.state_dict()['param_groups']):
        for v,i in zip(live['params'],saved['params']):indices[str(i)]=next(n for n,q in named if q is v)
    initial=tensor_sha(model.state_dict());write(a.out/'clean_public_initialization.json',dict(actual_UTC=utc(),public_encoder_tensors_matched=matched,initial_state_SHA=initial,model_parameters=sum(v.numel() for v in model.parameters()),optimized_parameters=sum(v.numel() for _,v in named),official_IDs=p['official_train_dev_IDs'],TRAIN_only_normalization=True,old_task_state_loaded=False,optimizer_steps=0,model_source_SHA=p['source_sha256']['polarity_intensity_flow.py']))
    np.save(a.out/'TRAIN_targets.npy',ytrain);shutil.copy2(a.bundle/p['orders_file'],a.out/'original_common_orders.npy')
    def budget():
        alloc=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();assert max(alloc,reserved)<=p['GPU_peak_ceiling_bytes'];assert shutil.disk_usage(a.out).free>=p['remote_free_floor_bytes'];return dict(allocated_peak=alloc,reserved_peak=reserved)
    def predict(dummy=0):
        model.eval();values=[]
        with torch.no_grad():
            for startrow in range(0,229,128):
                batch=[v[startrow:startrow+128].cuda() for v in dev_inputs.tensors];batch[3].fill_(dummy);values.append(forward_batch(model,batch).cpu().numpy())
        return np.concatenate(values).astype(np.float32)
    history=[];best=float('inf');selected=None;bestepoch=0;bestpred=None;steps=0;timings=[];firstgradient=None
    for epoch in range(100):
        batches,omitted=train_batches([int(x) for x in order[epoch]]);losses=[];epochstart=time.monotonic()
        for j,rows in enumerate(batches):
            model.train();opt.zero_grad(set_to_none=True);batch=[v[list(rows)].cuda() for v in train.tensors];torch.cuda.synchronize();tick=time.monotonic()
            pred=forward_batch(model,batch);loss=objective(model,pred,batch[3]);assert torch.isfinite(loss);loss.backward()
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for _,v in named)
            if steps==0:
                firstgradient={n:float(v.grad.abs().sum()) for n,v in named};assert any(v>0 for n,v in firstgradient.items() if n.startswith('dberta.model.'))
                write(a.out/'first_real_TRAIN_gradient.json',firstgradient)
            opt.step();scheduler.step();steps+=1;torch.cuda.synchronize();elapsed=time.monotonic()-tick;losses.append(float(loss.detach()));timings.append(elapsed)
            with (a.out/'TRAIN_steps.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch+1,step=steps,rows=list(rows),loss=losses[-1],seconds=elapsed))+'\n')
            budget()
            if steps==2:
                projection=max(timings)*4000*1.5+100*10+900;assert projection<=p['training_budget_seconds'],projection
                write(a.out/'actual_two_update_budget.json',dict(actual_UTC=utc(),two_real_updates_counted_in_4000=True,projected_seconds=projection,budget_seconds=p['training_budget_seconds'],fresh_public_initialization=True,final_state_capture_required=True))
            assert time.monotonic()-start<p['training_budget_seconds']
        assert steps==(epoch+1)*40 and scheduler.last_epoch==steps
        prediction=predict();stateSHA=tensor_sha(model.state_dict());path=a.out/('DEV_epoch_%03d_prediction_only.npz'%(epoch+1));np.savez(path,row_ids=np.asarray(guard.ids['dev']),prediction=prediction,model_state_sha256=np.asarray(stateSHA));frozenSHA=sha(path)
        with (a.out/'DEV_prediction_frozen_before_labels.jsonl').open('a') as f:f.write(json.dumps(dict(actual_UTC=utc(),epoch=epoch+1,prediction_SHA=frozenSHA,state_SHA=stateSHA))+'\n')
        y=guard.dev_labels_after_frozen_prediction(path,frozenSHA,stateSHA)
        if epoch==0:np.save(a.out/'DEV_selection_targets.npy',y)
        else:assert np.array_equal(y,np.load(a.out/'DEV_selection_targets.npy',allow_pickle=False))
        score=author_dev_batch_mse([np.mean((prediction[i:i+128].astype(float)-y[i:i+128])**2) for i in (0,128)],(128,101))
        if score<best:best=score;bestepoch=epoch+1;selected=cpu(model.state_dict());bestpred=prediction.copy()
        record=dict(actual_UTC=utc(),epoch=epoch+1,steps=steps,TRAIN_objective=float(np.mean(losses)),DEV_batch_MSE=float(score),best_epoch=bestepoch,best_MSE=float(best),dropped_rows=list(omitted),prediction_SHA=frozenSHA,state_SHA=stateSHA,seconds=time.monotonic()-epochstart)
        history.append(record);write(a.out/'progress.json',record);write(a.out/'history.json',history);print(json.dumps(record),flush=True)
    final=cpu(model.state_dict());finalrng=rng();finalSHA=tensor_sha(final);bestSHA=tensor_sha(selected)
    metadata=dict(candidate_mode=p['candidate_mode'],fixed_loss='Huber1(p,y)+.1Huber1(m,abs(y))+.1weighted_signBCE+.01context2',plan_SHA=a.plan_sha,model_source_SHA=p['source_sha256']['polarity_intensity_flow.py'],epochs=100,steps=4000,best_epoch=bestepoch,best_MSE=float(best),final_state_SHA=finalSHA,selected_state_SHA=bestSHA,initial_public_state_SHA=initial,parameter_info=info,optimizer_index_to_name=indices,normalization='official TRAIN1281 only',official_train_dev_IDs=p['official_train_dev_IDs'],orders_SHA=p['orders_SHA'],checkpoint_selection='fixed DEV batch128/101 mean MSE strict earliest',whole_trusted_pickle_materialized=True,TEST_entry_not_indexed=True)
    state=dict(metadata=metadata,model=final,selected_model=selected,optimizer=cpu(opt.state_dict()),scheduler=cpu(scheduler.state_dict()),rng=finalrng,history=history)
    cp=a.out/'complete_final_and_selected.pt';torch.save(state,cp);assert cp.stat().st_size<=p['checkpoint_bytes_ceiling'];del state,final
    model.load_state_dict(selected,strict=True);model.eval();saved=tensor_sha(model.state_dict());rr=rng();v0=predict(0);v7=predict(7);assert np.array_equal(v0,bestpred) and np.array_equal(v0,v7) and tensor_sha(model.state_dict())==saved and torch.equal(rr['torch'],torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(rr['cuda'],torch.cuda.get_rng_state_all()))
    tail=CandidateTail(model.dberta);torch.save({'state':tail.state_dict(),'mode':p['candidate_mode']},a.out/'selected_original_tail.pt');torch.save({'state':cpu(model.dberta.own_flow.core.message.state_dict())},a.out/'selected_message.pt')
    sources=[];masks=[]
    with torch.no_grad():
        for i in range(0,229,16):
            batch=[v[i:i+16].cuda() for v in dev_inputs.tensors];source,mask=source_features(model.dberta,batch);sources.append(source.cpu().numpy());masks.append(mask.cpu().numpy())
    np.savez(a.out/'selected_DEV_original_source.npz',ids=np.asarray(guard.ids['dev']),source=np.concatenate(sources),mask=np.concatenate(masks),prediction=bestpred)
    np.savez(a.out/'selected_DEV_frozen_prediction.npz',ids=np.asarray(guard.ids['dev']),prediction=bestpred,state_SHA=np.asarray(bestSHA))
    write(a.out/'guard_journal.json',guard.journal)
    write(a.out/'training_result.json',dict(status='CLEAN_OFFICIAL_POLARITY_INTENSITY_TRAIN100_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,metadata=metadata,checkpoint_SHA=sha(cp),checkpoint_bytes=cp.stat().st_size,selected_DEV_prediction_SHA=sha(a.out/'selected_DEV_frozen_prediction.npz'),selected_source_SHA=sha(a.out/'selected_DEV_original_source.npz'),selected_tail_SHA=sha(a.out/'selected_original_tail.pt'),selected_message_SHA=sha(a.out/'selected_message.pt'),full_selected_DEV_replay_dummy0vs7_error=0,TEST_entry_not_indexed=True,peak=budget(),wall_seconds=time.monotonic()-start));print(read(a.out/'training_result.json'),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','out'):a.add_argument('--'+n,type=Path,required=True)
    a.add_argument('--plan-sha',required=True);run(a.parse_args())
