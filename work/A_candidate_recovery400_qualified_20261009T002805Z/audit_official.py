"""Other-node complete checkpoint/Adam/selection and actual flow CPU replay."""
import argparse,os,sys
from pathlib import Path
import numpy as np
import torch
from common import sha,read,write,utc,verify
from candidate_adapter import CandidateTail,make_candidate
from types import SimpleNamespace
from torch import nn
from fixed_flow_components_candidate import tensor_sha

def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets);a.out.mkdir();torch.set_num_threads(2)
    ref=p['training_original_reference'];root=a.input_root
    for n,h in ref['member_SHA'].items():assert sha(root/n)==h,n
    state=torch.load(root/'out/complete_final_and_selected.pt',map_location='cpu');m=state['metadata'];assert m['plan_SHA']==ref['training_plan_SHA'] and m['steps']==4000 and m['epochs']==100 and m['TEST_entry_not_indexed']
    assert tensor_sha(state['model'])==m['final_state_SHA'] and tensor_sha(state['selected_model'])==m['selected_state_SHA']
    info=m['parameter_info'];mapping=m['optimizer_index_to_name'];opt=state['optimizer'];assert len(opt['state'])==len(info)
    for i,q in opt['state'].items():
        n=mapping[str(i)];assert q['step'].item()==4000
        for key in ('exp_avg','exp_avg_sq'):assert list(q[key].shape)==info[n]['shape'] and torch.isfinite(q[key]).all()
    assert state['scheduler']['last_epoch']==4000 and state['rng']['torch'].numel()>0 and len(state['rng']['cuda'])==1
    hist=read(root/'out/history.json');assert hist==state['history'] and len(hist)==100 and m['prefix_updates_counted_in4000'] and len(state['prefix_records'])==16
    y=np.load(root/'out/DEV_selection_targets.npy',allow_pickle=False);assert y.shape==(229,)
    order=np.load(root/'out/original_common_orders.npy',allow_pickle=False);assert order.shape==(100,1281) and sha(root/'out/original_common_orders.npy')==p['orders_SHA']
    actual=[]
    for i,r in enumerate(hist):
        assert r['steps']==(i+1)*40 and r['dropped_rows']==[int(order[i,-1])]
        z=np.load(root/('out/DEV_epoch_%03d_prediction_only.npz'%(i+1)),allow_pickle=False)
        assert z['row_ids'].tolist()==p['official_train_dev_IDs']['dev'] and str(z['model_state_sha256'].item())==r['state_SHA']
        v=z['prediction'].astype(float);metric=np.mean([np.mean((v[s:s+128]-y[s:s+128])**2) for s in (0,128)]);assert abs(metric-r['DEV_batch_MSE'])<1e-12;actual.append(metric)
    best=int(np.argmin(actual))+1;assert best==m['best_epoch'] and abs(actual[best-1]-m['best_MSE'])<1e-12
    assert m['candidate_mode']==p['candidate_mode']
    cache=dict(np.load(root/'out/selected_DEV_original_source.npz',allow_pickle=False))
    core=SimpleNamespace(own_flow=make_candidate(m['candidate_mode']),fusion=nn.Sequential(nn.Linear(300,150),nn.ReLU(),nn.Linear(150,100)),predictor=nn.Sequential(nn.Linear(100,150),nn.ReLU(),nn.Linear(150,1)))
    tail=CandidateTail(core);export=torch.load(root/'out/selected_original_tail.pt',map_location='cpu');assert export['mode']==m['candidate_mode'];tail.load_state_dict(export['state'],strict=True)
    selected=state['selected_model']
    for n,v in tail.state_dict().items():
        key='dberta.own_flow.'+n[len('flow.'):] if n.startswith('flow.') else 'dberta.'+n
        assert torch.equal(v,selected[key]),n
    expected={k.removeprefix('dberta.own_flow.core.message.'):v for k,v in selected.items() if k.startswith('dberta.own_flow.core.message.')}
    assert tensor_sha(expected)==tensor_sha(torch.load(root/'out/selected_message.pt',map_location='cpu')['state'])
    pieces=[];before=tensor_sha(tail.state_dict());rng=torch.get_rng_state().clone()
    with torch.no_grad():
        for i in range(0,229,32):
            src=torch.from_numpy(cache['source'][i:i+32]);mask=torch.from_numpy(cache['mask'][i:i+32]);on,off=tail(src,mask);assert torch.isfinite(on).all() and torch.isfinite(off).all();pieces.append(on.numpy())
    err=float(np.max(abs(np.concatenate(pieces)-cache['prediction'])));assert err<=1e-4
    assert before==tensor_sha(tail.state_dict()) and torch.equal(rng,torch.get_rng_state())
    write(a.out/'audit_result.json',dict(status='OFFICIAL_UPGRADE_FULL_STATE_ADAM4000_SELECTION_AND_CPU_FLOW_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,training_archive_SHA=ref['archive_SHA'],checkpoint_SHA=sha(root/'out/complete_final_and_selected.pt'),selected_state_SHA=m['selected_state_SHA'],best_epoch=best,Adam_parameter_states=len(info),all_Adam_steps=4000,all100_prediction_then_label_selection_recomputed=True,CPU_actual_DEV_flow_maxerror=err,candidate_mode=m['candidate_mode'],CandidateTail_all_selected_parameters_exact=True,CPU_encoder_forward=False,parameters_buffers_RNG_unchanged=True,TEST_not_indexed=True))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','out','input-root'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())
