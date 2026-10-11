from pathlib import Path
import ast,hashlib
p=Path(__file__).resolve().parent
s=(p/'diagnose_gated_conditions_v2.py').read_text(encoding='utf-8').split('    flow=core.own_flow')[0]
s=s.replace("choices=['none','state','task']","choices=['none','fixed','predicted']")
s=s.replace('INFLOW_CONDITIONS_RUN_COMPLETE','INFLOW_UTILITY_RUN_COMPLETE').replace('from conditional_flow_model import WholeStateFlow,CONFIG','from utility_flow_model import WholeStateFlow,CONFIG,PAIRS').replace("assert p['seed']==91812","assert p['seed']==91813")
s+='''    assert sel['best_epoch']==min(history,key=lambda r:r['valid_mse'])['epoch']
    flow=core.own_flow
    original_update=flow.update_context
    loader=torch.utils.data.DataLoader(dev,batch_size=128,shuffle=False)
    observed=[]
    variant={'override':None,'disabled':(),'shift':(),'full_path':False}
    def modified(self,old,relation):
        self._calls+=1
        if self._calls>1:return old
        pooled=relation['slots'].mean(2)
        own=torch.stack([self.unimodal_heads[m](pooled[:,m]).view(-1) for m in range(3)],1)
        pairs=[];utilities=[];feedback=[]
        for i,(m,j) in enumerate(PAIRS):
            zm=pooled[:,m];zj=pooled[:,j]
            shifted=zj.roll(1,0) if i in variant['shift'] else zj
            teacher_donor=shifted if variant['full_path'] else zj
            pj=self.unimodal_heads[j](teacher_donor).view(-1) if variant['full_path'] and i in variant['shift'] else own[:,j]
            pair=self.pair_heads[i](torch.cat([zm,teacher_donor],-1)).view(-1)
            ui=torch.cat([zm,teacher_donor,torch.tanh(torch.stack([own[:,m],pj,pair],-1)/3.)],-1).detach()
            utilities.append(torch.tanh(self.utility_heads[i](ui).view(-1)))
            fi=torch.cat([zm,shifted,.5*(zm-shifted),.5*(zm+shifted)],-1).detach()
            f=self.donor_feedback[i](fi)
            feedback.append(torch.zeros_like(f) if i in variant['disabled'] else f);pairs.append(pair)
        pairs=torch.stack(pairs,1);utilities=torch.stack(utilities,1);feedback=torch.stack(feedback,1)
        predicted_weights=torch.sigmoid(CONFIG['utility_gate_sharpness']*utilities)
        mode=self.mode if variant['override'] is None else variant['override']
        weights={'none':torch.zeros_like(predicted_weights),'fixed':torch.full_like(predicted_weights,.5),'predicted':predicted_weights}[mode]
        context=.5*(weights[:,:,None]*feedback).reshape(len(old),3,2,-1).sum(2)
        result=.5*old+.5*context
        self._stage1={'own':own,'pair':pairs,'utility':utilities,'weights':weights,'predicted_weights':predicted_weights}
        observed.append({**{k:v.detach().cpu().numpy() for k,v in self._stage1.items()},
            'feedback_norm':feedback.detach().norm(dim=-1).cpu().numpy(),
            'context_norm':result.detach().norm(dim=-1).cpu().numpy(),
            'state_norm':pooled.detach().norm(dim=-1).cpu().numpy()})
        return result
    def collect():
        pv=[];yv=[]
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                batch=tuple(x.to(author.DEVICE) for x in batch)
                neutral=(*batch[:3],torch.zeros_like(batch[3]),batch[4])
                pv.append(forward_batch(model,neutral)[0].view(-1).cpu().numpy())
                yv.append(batch[3].view(-1).cpu().numpy())
        return np.concatenate(pv),np.concatenate(yv)
    base_conditions=[('default',None),('off','none'),('forced_fixed','fixed'),('forced_predicted','predicted')]
    references={}
    for name,override in base_conditions:
        flow.context_override=override
        references[name],y=collect()
    flow.context_override=None
    with np.load(run/'predictions.npz',allow_pickle=False) as saved:
        assert np.array_equal(y,saved['valid_y'])
        assert np.allclose(references['default'],saved['valid_pred'],atol=2e-5,rtol=2e-5)
        assert np.allclose(references['off'],saved['condition_off_pred'],atol=2e-5,rtol=2e-5)
        max_default=float(np.max(np.abs(references['default']-saved['valid_pred'])))
        max_off=float(np.max(np.abs(references['off']-saved['condition_off_pred'])))
    flow.update_context=MethodType(modified,flow)
    conditions=[(n,o,(),(),False) for n,o in base_conditions]
    conditions += [(f'disable_{i}',None,(i,),(),False) for i in range(6)]
    conditions += [(f'shift_feedback_{i}',None,(),(i,),False) for i in range(6)]
    conditions += [('shift_feedback_all',None,(),tuple(range(6)),False)]
    conditions += [(f'shift_full_{i}',None,(),(i,),True) for i in range(6)]
    conditions += [('shift_full_all',None,(),tuple(range(6)),True)]
    assert len(conditions)==24
    arrays={'valid_y':y,'donor_permutation':np.concatenate([np.roll(np.arange(128),1),np.roll(np.arange(128,229),1)])}
    result_rows={};equivalence={};t=time.monotonic()
    for name,override,disabled,shift,full in conditions:
        variant.update(override=override,disabled=disabled,shift=shift,full_path=full)
        observed.clear();pred,truth=collect()
        assert np.array_equal(y,truth) and pred.shape==(229,) and np.isfinite(pred).all()
        assert len(observed)==2
        if name in references:
            difference=float(np.max(np.abs(pred-references[name])))
            assert np.allclose(pred,references[name],atol=2e-5,rtol=2e-5),name
            equivalence[name]=difference
        arrays['pred_'+name]=pred
        for key in observed[0]:arrays[key+'_'+name]=np.concatenate([v[key] for v in observed])
        if name=='default':
            with np.load(run/'predictions.npz',allow_pickle=False) as saved:
                for k in ['own','pair','utility','predicted_weights']:assert np.allclose(arrays[k+'_default'],saved[k],atol=2e-5,rtol=2e-5),k
        if c.mode=='none' and name not in ['forced_fixed','forced_predicted']:
            assert np.allclose(pred,references['default'],atol=2e-5,rtol=2e-5),('none intervention changed prediction',name)
        result_rows[name]=metrics(pred,y)
        print(json.dumps({'condition':name,'metrics':result_rows[name],'elapsed_seconds':time.monotonic()-t}),flush=True)
    after=state_sha(model)
    assert after==before and sha(run/'best.pt')==sel['checkpoint_sha256']
    np.savez_compressed(c.out/'predictions.npz',**arrays)
    result={'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'UTILITY_FROZEN_DEV_DIAGNOSTICS_COMPLETE',
        'mode':c.mode,'gpu_uuid':c.expected_gpu_uuid,'checkpoint_sha256':sel['checkpoint_sha256'],'source_sha256':sha(__file__),
        'conditions':result_rows,'original_method_equivalence_maxabs':equivalence,'default_replay_maxabs':max_default,'off_replay_maxabs':max_off,
        'model_state_before_sha256':before,'model_state_after_sha256':after,'output_sha256':sha(c.out/'predictions.npz'),
        'protocol_sha256':sel['protocol_sha256'],'elapsed_seconds':time.monotonic()-t,'test_accessed':False,'optimizer_updates':0,
        'donor_permutation':'roll1 within fixed official DEV128/101 batches, no label usage',
        'limits':['Inference intervention is not retraining.','Task-head utility does not establish semantic truth.','Exploratory single seed mode-node binding; DEV selected and observed.','Feedback-only shifts retain natural weights; full shifts recompute pair/utility with donor moved but receiver kept.']}
    with (c.out/'results.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print('UTILITY_FROZEN_DEV_DIAGNOSTICS_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
'''
ast.parse(s)
target=p/'diagnose_utility_v4_v1.py'
assert not target.exists()
target.write_text(s,encoding='utf-8')
print(hashlib.sha256(target.read_bytes()).hexdigest())
