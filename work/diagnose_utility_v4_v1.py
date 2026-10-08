"""Frozen-checkpoint DEV diagnosis; no training, TEST or label-fed forward."""
from pathlib import Path
from types import MethodType,SimpleNamespace
import argparse,datetime,hashlib,importlib,json,os,subprocess,sys,time
import numpy as np
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def state_sha(model):
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def metrics(pred,y):
    truth=y!=0
    return {'MAE':float(np.abs(pred-y).mean()),'MSE':float(np.square(pred-y).mean()),
        'author_batch_mse':float(np.mean([np.square(pred[i:i+128]-y[i:i+128]).mean() for i in range(0,len(y),128)])),
        'Non0_acc2':float(((pred[truth]>=0)==(y[truth]>=0)).mean()),
        'bias':float((pred-y).mean()),'Corr':float(np.corrcoef(pred,y)[0,1])}

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--mode',choices=['none','fixed','predicted'],required=True)
    a.add_argument('--expected-gpu-uuid',required=True)
    a.add_argument('--source-root',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    c=a.parse_args()
    assert not c.out.exists()
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits']).decode().strip()
    assert gpu.split(',')[0].strip()==c.expected_gpu_uuid
    assert int(gpu.split(',')[1])<=16
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
    c.out.mkdir(parents=True,exist_ok=False)
    run=c.source_root/('run_'+c.mode)
    p=json.loads((run/'protocol.json').read_text())
    sel=json.loads((run/'selection.json').read_text())
    assert p['mode']==c.mode and sel['epochs']==100
    history=json.loads((run/'history.json').read_text())
    assert len(history)==100 and 'INFLOW_UTILITY_RUN_COMPLETE' in (c.source_root/'training.log').read_text()
    assert sha(run/'protocol.json')==sel['protocol_sha256']
    assert sha(run/'best.pt')==sel['checkpoint_sha256']
    for name,digest in p['own_source_sha256'].items():assert sha(c.source_root/name)==digest
    B=Path('/data/coding/selective_flow/strong_baselines')
    for name,digest in p['helper_source_sha256'].items():assert sha(B/name)==digest
    for name,digest in p['author_source_sha256'].items():assert sha(B/'CaReFlow'/name)==digest
    for name,digest in p['backbone_sha256'].items():assert sha(B/'deberta-v3-base'/name)==digest
    assert sha(B/'mosi.pkl')==p['data_sha256']
    sys.path.insert(0,str(c.source_root));sys.path.insert(0,str(B))
    from utility_flow_model import WholeStateFlow,CONFIG,PAIRS
    from encoder_adapter import install,forward_v6,forward_batch
    helpers=importlib.import_module('run_careflow')
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(2)
    cli=SimpleNamespace(backbone=B/'deberta-v3-base',repo=B/'CaReFlow',epochs=100,seed=int(p['seed']))
    author,args=helpers.load_author(cli)
    assert p['seed']==91813
    author.set_random_seed(int(p['seed']))
    # Only DEV is requested from the container. Normalization is the saved TRAIN fit.
    dev=helpers.dataset(author,B/'mosi.pkl','dev')
    assert len(dev)==229
    model,opt,sched=author.prep_for_training(4000)
    del opt,sched
    core=model.dberta
    for name in ('reflow_a','reflow_v','reflow_a_b','reflow_v_b','rf_a','rf_v','rf_a_b','rf_v_b'):delattr(core,name)
    core.own_flow=WholeStateFlow(c.mode).to(author.DEVICE)
    statistics={m:{k:torch.tensor(v,dtype=torch.bool if k=='active' else torch.float32) for k,v in fields.items()} for m,fields in p['normalization_statistics'].items()}
    install(core,statistics)
    core.to(author.DEVICE)
    core.forward=MethodType(forward_v6,core)
    model.load_state_dict(torch.load(run/'best.pt',map_location='cpu'),strict=True)
    model.eval()
    before=state_sha(model)
    assert sel['best_epoch']==min(history,key=lambda r:r['valid_mse'])['epoch']
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
