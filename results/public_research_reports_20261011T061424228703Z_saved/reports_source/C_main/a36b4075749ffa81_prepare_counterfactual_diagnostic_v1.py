from pathlib import Path
import ast,hashlib
w=Path(__file__).resolve().parent
s=(w/'diagnose_utility_v4_v1.py').read_text();header=s[:s.index('    observed=[]')]
header=header.replace('utility_flow_model','counterfactual_flow_model').replace('91813','91814').replace('INFLOW_UTILITY_RUN_COMPLETE','COUNTERFACTUAL_V5_RUN_COMPLETE')
header=header.replace('    original_update=flow.update_context','    flow.set_epoch(sel["best_epoch"])\n    original_update=flow.update_context\n    original_context=flow.context')
tail=r'''
    variant={'override':None,'disabled':(),'shift':()}
    cached={'pool':None,'context_calls':0}
    observed=[]
    def wrapped_update(self,old,relation):
        if self._calls==0:
            cached['pool']=relation['slots'].mean(2);cached['context_calls']=0
        return original_update(old,relation)
    def wrapped_context(self,old,feedback,weights):
        cached['context_calls']+=1
        if cached['context_calls']==1:return original_context(old,feedback,weights)
        assert cached['context_calls']==2
        changed=feedback.clone();effective=weights.clone();pooled=cached['pool']
        for i,(m,j) in enumerate(PAIRS):
            if i in variant['shift']:
                zm=pooled[:,m];zj=pooled[:,j].roll(1,0)
                features=torch.cat([zm,zj,.5*(zm-zj),.5*(zm+zj)],-1).detach()
                changed[:,i]=self.donor_feedback[i](features)
            if i in variant['disabled']:effective[:,i]=0
        self._stage1['weights']=effective;self._stage1['feedback_norm']=changed.detach().norm(dim=-1)
        result=original_context(old,changed,effective)
        observed.append({**{k:self._stage1[k].detach().cpu().numpy() for k in ['own','pair','utility','weights','predicted_weights','reference_prediction','feedback_norm']},'context_norm':result.detach().norm(dim=-1).cpu().numpy()})
        return result
    def collect():
        pv=[];yv=[]
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                batch=tuple(x.to(author.DEVICE) for x in batch);neutral=(*batch[:3],torch.zeros_like(batch[3]),batch[4])
                pv.append(forward_batch(model,neutral)[0].view(-1).cpu().numpy());yv.append(batch[3].view(-1).cpu().numpy())
                if flow.context!=original_context:assert cached['context_calls']==2
        return np.concatenate(pv),np.concatenate(yv)
    bases=[('default',None),('off','none'),('forced_fixed','fixed'),('forced_predicted','predicted')]
    references={}
    for name,override in bases:
        flow.context_override=override;references[name],y=collect()
    flow.context_override=None
    with np.load(run/'predictions.npz',allow_pickle=False) as saved:
        assert np.array_equal(y,saved['valid_y'])
        assert np.allclose(references['default'],saved['valid_pred'],atol=2e-5,rtol=2e-5)
        assert np.allclose(references['off'],saved['condition_off_pred'],atol=2e-5,rtol=2e-5)
        max_default=float(np.max(np.abs(references['default']-saved['valid_pred'])));max_off=float(np.max(np.abs(references['off']-saved['condition_off_pred'])))
    flow.update_context=MethodType(wrapped_update,flow);flow.context=MethodType(wrapped_context,flow)
    conditions=[(n,o,(),()) for n,o in bases]
    conditions += [(f'disable_{i}',None,(i,),()) for i in range(6)]
    conditions += [(f'joint_disable_{m}',None,(2*m,2*m+1),()) for m in range(3)]
    conditions += [(f'shift_donor_{i}',None,(),(i,)) for i in range(6)]
    conditions += [('shift_donor_all',None,(),tuple(range(6)))]
    conditions += [(f'fixed_disable_{i}','fixed',(i,),()) for i in range(6)]
    conditions += [(f'fixed_joint_disable_{m}','fixed',(2*m,2*m+1),()) for m in range(3)]
    assert len(conditions)==29
    arrays={'valid_y':y,'donor_permutation':np.concatenate([np.roll(np.arange(128),1),np.roll(np.arange(128,229),1)])}
    result_rows={};equivalence={};t=time.monotonic()
    for name,override,disabled,shift in conditions:
        variant.update(override=override,disabled=disabled,shift=shift);flow.context_override=override;observed.clear();pred,truth=collect()
        assert np.array_equal(y,truth) and pred.shape==(229,) and np.isfinite(pred).all() and len(observed)==2
        if name in references:
            difference=float(np.max(np.abs(pred-references[name])));assert np.allclose(pred,references[name],atol=2e-5,rtol=2e-5),name;equivalence[name]=difference
        arrays['pred_'+name]=pred
        for key in observed[0]:arrays[key+'_'+name]=np.concatenate([v[key] for v in observed])
        if name=='default':
            with np.load(run/'predictions.npz',allow_pickle=False) as saved:
                for key in ['own','pair','utility','predicted_weights','weights','reference_prediction']:assert np.allclose(arrays[key+'_default'],saved[key],atol=2e-5,rtol=2e-5),key
        if sel['selected_effective_mode']=='none' and override is None:assert np.allclose(pred,references['default'],atol=2e-5,rtol=2e-5)
        result_rows[name]=metrics(pred,y);print(json.dumps({'condition':name,'metrics':result_rows[name],'elapsed_seconds':time.monotonic()-t}),flush=True)
    assert np.allclose(arrays['reference_prediction_default'],arrays['pred_forced_fixed'],atol=2e-5,rtol=2e-5),'Fixed reference differs from actual fixed terminal path'
    for name in result_rows:assert np.array_equal(arrays['reference_prediction_'+name],arrays['reference_prediction_default'])
    flow.update_context=original_update;flow.context=original_context;flow.context_override=None
    after=state_sha(model);assert after==before and sha(run/'best.pt')==sel['checkpoint_sha256']
    np.savez_compressed(c.out/'predictions.npz',**arrays)
    result={'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'COUNTERFACTUAL_FROZEN_DEV_DIAGNOSTICS_COMPLETE','mode':c.mode,'selected_effective_mode':sel['selected_effective_mode'],'best_epoch':sel['best_epoch'],'gpu_uuid':c.expected_gpu_uuid,'checkpoint_sha256':sel['checkpoint_sha256'],'source_sha256':sha(__file__),'conditions':result_rows,'original_method_equivalence_maxabs':equivalence,'default_replay_maxabs':max_default,'off_replay_maxabs':max_off,'model_state_before_sha256':before,'model_state_after_sha256':after,'output_sha256':sha(c.out/'predictions.npz'),'protocol_sha256':sel['protocol_sha256'],'elapsed_seconds':time.monotonic()-t,'test_accessed':False,'optimizer_updates':0,'donor_permutation':'roll1 within fixed DEV128/101; receiver retained, donor replaced in feedback generation, natural weights/ref/u retained','limits':['Single exploratory seed and observed DEV.','Task marginal utility is not semantic truth.','Fixed reference risk and natural-policy risk are distinct.','Joint risk residual alone does not identify prediction nonadditivity.']}
    with (c.out/'results.json').open('x') as f:json.dump(result,f,indent=2)
    print('COUNTERFACTUAL_FROZEN_DEV_DIAGNOSTICS_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
'''
s=header+tail;ast.parse(s);p=w/'diagnose_counterfactual_v5_v1.py';assert not p.exists();p.write_text(s);print(hashlib.sha256(p.read_bytes()).hexdigest())
