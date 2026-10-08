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
    a.add_argument('--mode',choices=['none','state','task'],required=True)
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
    assert len(history)==100 and 'INFLOW_CONDITIONS_RUN_COMPLETE' in (run/'training.log').read_text()
    assert sha(run/'protocol.json')==sel['protocol_sha256']
    assert sha(run/'best.pt')==sel['checkpoint_sha256']
    for name,digest in p['own_source_sha256'].items():assert sha(c.source_root/name)==digest
    B=Path('/data/coding/selective_flow/strong_baselines')
    for name,digest in p['helper_source_sha256'].items():assert sha(B/name)==digest
    for name,digest in p['author_source_sha256'].items():assert sha(B/'CaReFlow'/name)==digest
    for name,digest in p['backbone_sha256'].items():assert sha(B/'deberta-v3-base'/name)==digest
    assert sha(B/'mosi.pkl')==p['data_sha256']
    sys.path.insert(0,str(c.source_root));sys.path.insert(0,str(B))
    from conditional_flow_model import WholeStateFlow,CONFIG
    from encoder_adapter import install,forward_v6,forward_batch
    helpers=importlib.import_module('run_careflow')
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(2)
    cli=SimpleNamespace(backbone=B/'deberta-v3-base',repo=B/'CaReFlow',epochs=100,seed=int(p['seed']))
    author,args=helpers.load_author(cli)
    assert p['seed']==91812
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
    flow=core.own_flow
    original_update=flow.update_context
    variant={'name':'default','scale':1.0,'donor_shift':False}
    observed=[]
    def modified(self,old,relation):
        result=original_update(old,relation)
        if len(self._observer_predictions)!=1:return result
        pooled=relation['slots'].mean(dim=2)
        preds=self._observer_predictions[0]
        if variant['donor_shift'] and self.mode!='none':
            scaled=torch.tanh(preds/3)
            features=[]
            for m,j,k in ((0,1,2),(1,0,2),(2,0,1)):
                if self.mode=='state':
                    donorj,donork=pooled[:,j].roll(1,0),pooled[:,k].roll(1,0)
                    f=torch.cat([pooled[:,m],donorj,donork,.5*(donorj-donork)],-1)
                else:
                    donorj,donork=scaled[:,j].roll(1,0),scaled[:,k].roll(1,0)
                    f=torch.stack([scaled[:,m],donorj,donork,.5*(donorj-donork)],-1).repeat_interleave(CONFIG['dimension'],-1)
                features.append(self.feedback[m](f.detach()))
            selected=torch.stack(features,1)*torch.tanh(self.feedback_gate)[None,:,None]
            result=.5*old+.5*selected
        result=result*variant['scale']
        observed.append({'predictions':preds.detach().cpu().numpy(),
            'context_norm':result.detach().norm(dim=-1).cpu().numpy(),
            'state_norm':pooled.detach().norm(dim=-1).cpu().numpy()})
        return result
    flow.update_context=MethodType(modified,flow)
    conditions=[('default',1.0,False),('off',0.0,False)]
    if c.mode!='none':conditions += [('scale025',.25,False),('scale050',.5,False),('scale200',2.0,False),('sign_flip',-1.0,False),('donor_shift1',1.0,True)]
    loader=torch.utils.data.DataLoader(dev,batch_size=128,shuffle=False)
    predictions={};head={};contexts={};state_norms={};y=None
    t=time.monotonic()
    for name,scale,shift in conditions:
        variant.update(name=name,scale=scale,donor_shift=shift)
        observed.clear();pv=[];ys=[]
        with torch.inference_mode():
            for batch in loader:
                batch=tuple(x.to(author.DEVICE) for x in batch)
                # Labels are replaced with zeros before forward, only real y is saved for scoring.
                neutral=(*batch[:3],torch.zeros_like(batch[3]),batch[4])
                pred=forward_batch(model,neutral)[0].view(-1)
                pv.append(pred.cpu().numpy());ys.append(batch[3].view(-1).cpu().numpy())
        pred=np.concatenate(pv);truth=np.concatenate(ys)
        if y is None:y=truth
        else:assert np.array_equal(y,truth)
        assert pred.shape==(229,) and np.isfinite(pred).all()
        predictions[name]=pred
        head[name]=np.concatenate([r['predictions'] for r in observed])
        contexts[name]=np.concatenate([r['context_norm'] for r in observed])
        state_norms[name]=np.concatenate([r['state_norm'] for r in observed])
        row={'condition':name,'metrics':metrics(pred,y),'context_norm_mean':contexts[name].mean(0).tolist(),'elapsed_seconds':time.monotonic()-t}
        print(json.dumps(row),flush=True)
    with np.load(run/'predictions.npz',allow_pickle=False) as saved:
        assert np.array_equal(y,saved['valid_y'])
        assert np.allclose(predictions['default'],saved['valid_pred'],atol=2e-5,rtol=2e-5)
        assert np.allclose(predictions['off'],saved['condition_off_pred'],atol=2e-5,rtol=2e-5)
        max_default=float(np.max(np.abs(predictions['default']-saved['valid_pred'])))
        max_off=float(np.max(np.abs(predictions['off']-saved['condition_off_pred'])))
    after=state_sha(model)
    assert after==before
    assert sha(run/'best.pt')==sel['checkpoint_sha256']
    arrays={'valid_y':y}
    for name in predictions:
        arrays['pred_'+name]=predictions[name];arrays['heads_'+name]=head[name]
        arrays['context_norm_'+name]=contexts[name];arrays['state_norm_'+name]=state_norms[name]
    np.savez_compressed(c.out/'predictions.npz',**arrays)
    result={'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FROZEN_DEV_DIAGNOSTICS_COMPLETE',
        'mode':c.mode,'gpu_uuid':c.expected_gpu_uuid,'checkpoint_sha256':sel['checkpoint_sha256'],'source_sha256':sha(__file__),
        'conditions':{n:metrics(v,y) for n,v in predictions.items()},
        'feedback_gates_raw':flow.feedback_gate.detach().cpu().tolist(),
        'feedback_gates_tanh':torch.tanh(flow.feedback_gate).detach().cpu().tolist(),
        'stage1_heads':[metrics(head['default'][:,m],y) for m in range(3)],
        'context_norm_mean':{n:v.mean(0).tolist() for n,v in contexts.items()},
        'state_norm_mean':{n:v.mean(0).tolist() for n,v in state_norms.items()},
        'default_replay_maxabs':max_default,'off_replay_maxabs':max_off,'model_state_before_sha256':before,
        'model_state_after_sha256':after,'output_sha256':sha(c.out/'predictions.npz'),
        'protocol_sha256':sel['protocol_sha256'],'elapsed_seconds':time.monotonic()-t,
        'test_accessed':False,'optimizer_updates':0,
        'limits':['Exploratory DEV already observed; scaling is frozen inference intervention, not a retrained model.',
                  'Donor shift is a deterministic cyclic permutation separately within fixed 128/101 batches; it is not a global dataset permutation or a semantic ground truth.',
                  'Ordinary unimodal predictions are not calibrated contribution/conflict truth.',
                  'Gated donor intervention retains each trained receiver gate; sign flip is frozen inference only, not a sign-retrained model.']}
    with (c.out/'results.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print('FROZEN_DEV_DIAGNOSTICS_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
