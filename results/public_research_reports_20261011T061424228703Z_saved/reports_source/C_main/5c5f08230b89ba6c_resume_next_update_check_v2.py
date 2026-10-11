"""Two isolated FIT updates at saved epoch10; neither enters formal trajectory."""
import copy, datetime, hashlib, json, random, struct, time
from pathlib import Path
import numpy as np
import torch
from minimal_fixed_runtime_v1 import construct_public_candidate, tensor_sha, fit_objective, require_real_gradients

def nested_sha(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            v=x.detach().cpu().contiguous();h.update(b'T');h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'D')
            for k in sorted(x,key=lambda z:(type(z).__name__,repr(z))):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode());h.update(str(len(x)).encode())
            for v in x:visit(v)
        elif isinstance(x,float):h.update(b'F'+struct.pack('!d',x))
        elif x is None:h.update(b'N')
        elif isinstance(x,(bool,int,str)):h.update(type(x).__name__.encode()+repr(x).encode())
        else:raise TypeError('UNSUPPORTED_STATE_HASH_TYPE '+str(type(x)))
    visit(value);return h.hexdigest()

def restore_rng(saved,extra):
    def tup(x):return tuple(tup(v) for v in x) if isinstance(x,list) else x
    torch.set_rng_state(saved['torch']);torch.cuda.set_rng_state_all(saved['cuda'])
    random.setstate(tup(extra['python_random']))
    n=extra['numpy_global_random'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),int(n[2]),int(n[3]),float(n[4])))

def rng_digest():
    n=np.random.get_state()
    extra={'python_random':random.getstate(),'numpy_global_random':[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])]}
    return nested_sha({'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'extra':extra})

def clear_graph(session):
    session.model.dberta.last_first_prediction=None;session.model.dberta.last_losses={};session.model.dberta.last_trace={}
    session.model.dberta.own_flow._observer_prediction=None
    session.optimizer.zero_grad(set_to_none=True)

def verify_next_update(session,path,rows,asset_base,bundle,output,budget):
    start=time.perf_counter();path=Path(path);snapshot=torch.load(path,map_location='cpu',weights_only=True)
    assert snapshot['metadata']['epoch']==10 and snapshot['metadata']['optimizer_steps']==220
    latest=snapshot['model'];saved_state_sha=tensor_sha(latest)
    # Same session remains last10 for continuous side branch. The caller may
    # have replayed selected best: restore exact last10 BEFORE branch comparison.
    session.model.load_state_dict(latest,strict=True)
    assert tensor_sha(session.model.state_dict())==saved_state_sha
    assert nested_sha(session.optimizer.state_dict())==nested_sha(snapshot['optimizer'])
    assert nested_sha(session.scheduler.state_dict())==nested_sha(snapshot['scheduler'])
    def step(current,label):
        local={int(r):i for i,r in enumerate(current.guard.ids['fit'])}
        indices=torch.tensor([local[int(r)] for r in rows],dtype=torch.long)
        batch=tuple(x[indices].to('cuda') for x in current.fit.tensors)
        current.model.train();current.optimizer.zero_grad(set_to_none=True)
        restore_rng(snapshot['rng'],snapshot['rng_extra'])
        before_rng=rng_digest();assert current.scheduler.last_epoch==220
        loss=fit_objective(current.model,batch);assert torch.isfinite(loss)
        loss_value=float(loss.detach());loss.backward();grads=require_real_gradients(current.model);assert len(grads)==364
        current.optimizer.step();current.scheduler.step();torch.cuda.synchronize()
        after={'model_state_sha256':tensor_sha(current.model.state_dict()),'optimizer_complete_sha256':nested_sha(current.optimizer.state_dict()),'scheduler_sha256':nested_sha(current.scheduler.state_dict()),'ending_rng_sha256':rng_digest(),'starting_rng_sha256':before_rng,'objective':loss_value,'scheduler_last_epoch':current.scheduler.last_epoch,'optimizer_parameter_tensors':len(current.optimizer.state)}
        assert current.scheduler.last_epoch==221
        for state in current.optimizer.state.values():assert float(state['step'])==221
        clear_graph(current);del batch,loss,grads
        budget('isolated_next_update_'+label)
        return after
    continuous=step(session,'continuous')
    # Move original full ownership to host before constructing a second GPU
    # instance. This is a diagnostic optimization, not formal parameter pruning.
    session.model.to('cpu')
    for state in session.optimizer.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor):state[key]=value.cpu()
    torch.cuda.empty_cache()
    reloaded=construct_public_candidate(asset_base,bundle)
    reloaded.model.dberta.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
    del reloaded.clean_initial
    reloaded.model.load_state_dict(latest,strict=True)
    reloaded.optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']));reloaded.scheduler.load_state_dict(snapshot['scheduler'])
    assert tensor_sha(reloaded.model.state_dict())==saved_state_sha
    assert nested_sha(reloaded.optimizer.state_dict())==nested_sha(snapshot['optimizer'])
    assert nested_sha(reloaded.scheduler.state_dict())==nested_sha(snapshot['scheduler'])
    restored=step(reloaded,'fresh_disk_restore')
    if continuous!=restored:raise RuntimeError('EXACT_NEXT_UPDATE_MODEL_ADAM_SCHEDULER_RNG_MISMATCH')
    del reloaded
    torch.cuda.empty_cache()
    # Return original session to EXACT saved boundary. Neither diagnostic step
    # enters model selection, formal count or future resumed training.
    session.model.to('cuda');session.model.load_state_dict(latest,strict=True)
    session.optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']));session.scheduler.load_state_dict(snapshot['scheduler'])
    clear_graph(session);restore_rng(snapshot['rng'],snapshot['rng_extra'])
    assert tensor_sha(session.model.state_dict())==saved_state_sha
    assert nested_sha(session.optimizer.state_dict())==nested_sha(snapshot['optimizer'])
    assert nested_sha(session.scheduler.state_dict())==nested_sha(snapshot['scheduler'])
    assert rng_digest()==continuous['starting_rng_sha256']
    result={'status':'ACTUAL_NEXT_UPDATE_CONTINUOUS_AND_FRESH_DISK_RESTORED_EXACT_MATCH','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.perf_counter()-start,'formal_steps_before_and_after':220,'isolated_diagnostic_steps':2,'formal_step221_not_committed':True,'fit_rows':rows.tolist(),'continuous':continuous,'restored':restored,'exact_complete_model_Adam_scheduler_RNG_compared':True,'original_boundary_restored':True,'new_selection_or_OUTER_labels_used':False}
    (Path(output)/'actual_next_update_continuation_proof.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['seconds']>240:raise RuntimeError('FOUR_MINUTE_NEXT_UPDATE_DIAGNOSTIC_BUDGET')
    return result
