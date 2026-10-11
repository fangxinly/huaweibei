from pathlib import Path
import ast,hashlib,json
p=Path(__file__).resolve().parent;out=p/'inflow_counterfactual_v5_checks_20261005T0502Z'
base=(p/'inflow_utility_v4_20261005T0346Z/run_inflow_utility_v4.py').read_text(encoding='utf-8')
s=base.split('    if cli.stage == "check":')[0]
s=s.replace('from utility_flow_model import CONFIG, WholeStateFlow','from counterfactual_flow_model import CONFIG, WholeStateFlow')
s=s.replace('choices=("check", "run")','choices=("check",)')
s=s.replace('91813','91814').replace('run_inflow_utility_v4.py','check_counterfactual_v5.py').replace('utility_flow_model.py','counterfactual_flow_model.py').replace('inflow_utility_v4','inflow_counterfactual_v5_check')
s=s.replace('Six tanh utility heads target detach(tanh(((p_m-y)^2-(p_mj-y)^2)/.5)); smoothL1 weight.01; pair MSE .025; every mode same auxiliary losses; labels absent from inference utility inputs',
    'TRAIN final-task marginal g=(p_without_i-y)^2-(p_ref-y)^2 with fixed reference .5; tanh(g/.05), detached target/input, balanced SmoothL1 .01; first10epochs shared fixed feedback; 20 mechanism updates only; 10 deterministic references training/1 inference; no label in u input')
s+='''    assert cli.stage=='check'
    core.own_flow.set_epoch(11)
    batch=tuple(x.to(author.DEVICE) for x in next(iter(train_loader)))
    model.eval()
    with torch.no_grad():
        neutral=(*batch[:3],torch.zeros_like(batch[3]),batch[4])
        changed=(*batch[:3],torch.full_like(batch[3],123),batch[4])
        baseline=forward_batch(model,neutral)[0]
        original_u=core.own_flow._stage1['utility'].clone()
        original_w=core.own_flow._stage1['weights'].clone()
        replay=forward_batch(model,changed)[0]
        assert torch.equal(baseline,replay)
        assert torch.equal(original_u,core.own_flow._stage1['utility']) and torch.equal(original_w,core.own_flow._stage1['weights'])
        reference_u=core.own_flow._stage1['reference_prediction'].clone()
        for override in ('none','fixed','predicted'):
            core.own_flow.context_override=override
            pred=forward_batch(model,neutral)[0]
            assert torch.allclose(pred,baseline,atol=1e-6,rtol=1e-6)
        core.own_flow.context_override=None
    assert tensor_state_sha(model)==initial_state_sha
    core.own_flow.set_epoch(1)
    model.train()
    flags={id(m):m.training for m in model.modules()}
    optimizer.zero_grad(set_to_none=True)
    pred,_,_=forward_batch(model,batch)
    assert {id(m):m.training for m in model.modules()}==flags
    stage=core.own_flow._stage1
    assert stage['effective_mode']=='fixed'
    assert torch.max(torch.abs(stage['raw_utility']))==0
    assert not stage['reference_prediction'].requires_grad and not stage['without_prediction'].requires_grad and not stage['target'].requires_grad
    initial_zero_target=True
    initial_gradient=torch.autograd.grad(core.last_losses['utility_calibration'],[core.own_flow.utility_heads[0][-1].weight,core.own_flow.donor_feedback[0][-1].weight,core.proj_a.weight,core.predictor.weight if hasattr(core.predictor,'weight') else next(core.predictor.parameters())],allow_unused=True,retain_graph=True)
    assert initial_gradient[0] is not None and initial_gradient[0].abs().sum()==0
    assert all(g is None for g in initial_gradient[1:])
    timings=[];seen_nonzero=False;seen_utility_gradient=False;first_feedback_gradient=None
    train_batches=list(train_loader)
    for step in range(20):
        batch=tuple(x.to(author.DEVICE) for x in train_batches[step])
        flags={id(m):m.training for m in model.modules()}
        torch.cuda.synchronize();started=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        pred,_,_=forward_batch(model,batch)
        assert {id(m):m.training for m in model.modules()}==flags
        stage=core.own_flow._stage1
        assert stage['effective_mode']=='fixed' and torch.equal(stage['weights'],torch.full_like(stage['weights'],.5))
        target_check=((stage['without_prediction']-batch[3].view(-1,1)).square()-(stage['reference_prediction'][:,None]-batch[3].view(-1,1)).square()).detach()
        assert torch.equal(target_check,stage['raw_utility'])
        assert not stage['reference_prediction'].requires_grad and not stage['without_prediction'].requires_grad
        assert not stage['joint_without_prediction'].requires_grad and not stage['target'].requires_grad
        assert torch.equal(stage['interaction_residual'],stage['joint_utility']-stage['raw_utility'].reshape(len(batch[3]),3,2).sum(2))
        utility_grads=torch.autograd.grad(core.last_losses['utility_calibration'],[core.own_flow.utility_heads[0][-1].weight,core.own_flow.donor_feedback[0][-1].weight,core.proj_a.weight,next(core.predictor.parameters())],allow_unused=True,retain_graph=True)
        assert all(g is None for g in utility_grads[1:])
        seen_nonzero|=bool(stage['raw_utility'].abs().sum()>0)
        seen_utility_gradient|=bool(utility_grads[0] is not None and utility_grads[0].abs().sum()>0)
        loss=objective(pred,batch[3],core.last_losses);assert torch.isfinite(loss);loss.backward()
        if step==0:
            first_feedback_gradient=float(core.own_flow.donor_feedback[0][-1].weight.grad.abs().sum())
            assert first_feedback_gradient>0
        for parameter in [core.model.embeddings.word_embeddings.weight,core.proj_a.weight,core.proj_v.weight,core.own_flow.forward_fields[0].net[-1].weight,core.own_flow.pair_heads[0][-1].weight]:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
        optimizer.step();scheduler.step();torch.cuda.synchronize();timings.append(time.monotonic()-started)
        print(json.dumps({'check_step':step+1,'seconds':timings[-1],'target_nonzero_fraction':float((stage['raw_utility']!=0).float().mean()),'utility_gradient_nonzero':bool(utility_grads[0].abs().sum()>0)}),flush=True)
    assert seen_nonzero and seen_utility_gradient
    hidden_gradient=core.own_flow.donor_feedback[0][1].weight.grad
    assert hidden_gradient is not None and torch.isfinite(hidden_gradient).all() and hidden_gradient.abs().sum()>0
    after_updates=tensor_state_sha(model);assert after_updates!=initial_state_sha
    core.own_flow.set_epoch(11);model.eval();state_before=tensor_state_sha(model)
    small=tuple(x[:4] for x in batch)
    with torch.no_grad():
        neutral=(*small[:3],torch.zeros_like(small[3]),small[4])
        changed=(*small[:3],torch.full_like(small[3],123),small[4])
        natural=forward_batch(model,neutral)[0];u=core.own_flow._stage1['utility'].clone();w=core.own_flow._stage1['weights'].clone()
        replay=forward_batch(model,changed)[0];assert torch.equal(natural,replay) and torch.equal(u,core.own_flow._stage1['utility']) and torch.equal(w,core.own_flow._stage1['weights'])
        assert core.own_flow._stage1['effective_mode']==cli.mode
        solo=torch.cat([forward_batch(model,tuple(x[i:i+1] for x in neutral))[0] for i in range(4)])
        assert torch.allclose(natural,solo,atol=2e-5,rtol=2e-5)
        perturb=[x.clone() for x in neutral];invalid=~perturb[4].bool();perturb[0][invalid]=1
        for i in (1,2):perturb[i][invalid]=1000
        assert torch.allclose(natural,forward_batch(model,tuple(perturb))[0],atol=2e-5,rtol=2e-5)
    assert tensor_state_sha(model)==state_before
    model.train();optimizer.zero_grad(set_to_none=True)
    pred,_,_=forward_batch(model,batch)
    task_grads=torch.autograd.grad((pred.view(-1)-batch[3].view(-1)).square().mean(),[core.own_flow.utility_heads[0][-1].weight,core.own_flow.donor_feedback[0][-1].weight],allow_unused=True)
    if cli.mode=='predicted':assert all(g is not None and g.abs().sum()>0 for g in task_grads)
    if cli.mode=='none':assert all(g is None or g.abs().sum()==0 for g in task_grads)
    if cli.mode=='fixed':assert task_grads[0] is None and task_grads[1] is not None and task_grads[1].abs().sum()>0
    output={'mode':cli.mode,'status':'COUNTERFACTUAL_MECHANISM_CHECK_COMPLETE','label_isolation_verified':True,'initial_zero_target_verified':initial_zero_target,
        'reference_and_utility_gradient_detach_verified':True,'module_training_flags_restored':True,'first_feedback_gradient':first_feedback_gradient,
        'nonzero_reference_targets_after_updates_verified':seen_nonzero,'nonzero_utility_gradient_after_updates_verified':seen_utility_gradient,
        'optimizer_updates':20,'shared_fixed_phase_verified':True,'post_phase_mode_gradient_verified':True,'batch_invariance_verified':True,'mask_invariance_verified':True,
        'initial_model_sha256':initial_state_sha,'initial_flow_sha256':initial_flow_sha,'after_shared_updates_model_sha256':after_updates,'batch_order_sha256':digest_bytes(orders.tobytes()),
        'steady_update_seconds_mean':float(np.mean(timings[5:])),'estimated_4000_update_seconds':float(np.mean(timings[5:])*4000),
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'reference_paths_train':10,'reference_paths_eval':1,'test_accessed':False,'performance_experiment_started':False}
    helpers.write(cli.out/'checks.json',output)
    print('COUNTERFACTUAL_MECHANISM_CHECK_COMPLETE',json.dumps(output),flush=True)

if __name__=='__main__':main()
'''
ast.parse(s);target=out/'check_counterfactual_v5.py';assert not target.exists();target.write_text(s,encoding='utf-8')
manifest={'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.py')}}
(out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(manifest))
