"""Independent full4000/selection audit and original candidate-tail CPU replay."""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
from common import sha, read, write, utc, verify
from candidate_audit_utils import load_verified_tail
from fixed_flow_components_candidate import tensor_sha

def run(a):
    p=read(a.plan); assert sha(a.plan)==a.plan_sha
    verify(p,a.bundle,a.assets); a.out.mkdir(); torch.set_num_threads(2)
    ref=p['training_original_reference']; root=a.input_root
    # All large and small member hashes are already verified by the linear ZIP
    # reader. Only explicitly extracted files are needed for this audit.
    byte_receipt=read(root/'verified_extraction.json')
    assert byte_receipt['archive_SHA']==ref['archive_SHA']
    assert byte_receipt['all_member_SHA_CRC_unique_verified']
    for name in byte_receipt['extracted_members']:
        assert sha(root/name)==ref['member_SHA'][name], name
    checkpoint=root/'out/complete_final_and_selected.pt'
    state=torch.load(checkpoint,map_location='cpu'); m=state['metadata']
    assert m['candidate_mode']==p['candidate_mode'] and m['steps']==4000 and m['epochs']==100
    assert m['plan_SHA']==ref['training_plan_SHA'] and m['TEST_entry_not_indexed']
    assert m['prefix_updates_counted_in4000'] and len(state['prefix_records'])==16
    assert m['recovery_replayed_updates_not_new_independent_replicate']
    assert tensor_sha(state['model'])==m['final_state_SHA']
    assert tensor_sha(state['selected_model'])==m['selected_state_SHA']
    assert m['selected_state_SHA']==p['selected_state_SHA']
    info=m['parameter_info']; mapping=m['optimizer_index_to_name']; opt=state['optimizer']
    assert len(opt['state'])==len(info)==345
    assert set(mapping.values())==set(info)
    for index, row in opt['state'].items():
        name=mapping[str(index)]; assert row['step'].item()==4000
        assert list(state['model'][name].shape)==info[name]['shape']
        for key in ('exp_avg','exp_avg_sq'):
            assert list(row[key].shape)==info[name]['shape'] and torch.isfinite(row[key]).all()
    assert state['scheduler']['last_epoch']==4000
    rr=state['rng']; assert rr['torch'].numel()>0 and len(rr['cuda'])==1
    assert len(rr['python'])==3 and rr['numpy'][0]=='MT19937' and len(rr['numpy'][1])==624
    hist=read(root/'out/history.json'); assert hist==state['history'] and len(hist)==100
    y=np.load(root/'out/DEV_selection_targets.npy',allow_pickle=False); assert y.shape==(229,)
    order=np.load(root/'out/original_common_orders.npy',allow_pickle=False)
    assert order.shape==(100,1281) and sha(root/'out/original_common_orders.npy')==p['orders_SHA']
    scores=[]
    for index, row in enumerate(hist):
        assert np.array_equal(np.sort(order[index]),np.arange(1281))
        assert row['steps']==(index+1)*40 and row['dropped_rows']==[int(order[index,-1])]
        path=root/('out/DEV_epoch_%03d_prediction_only.npz'%(index+1))
        assert sha(path)==row['prediction_SHA']
        with np.load(path,allow_pickle=False) as z:
            assert z['row_ids'].tolist()==p['official_train_dev_IDs']['dev']
            assert str(z['model_state_sha256'].item())==row['state_SHA']
            v=z['prediction'].astype(float)
        score=float(np.mean([np.mean((v[s:s+128]-y[s:s+128])**2) for s in (0,128)]))
        assert abs(score-row['DEV_batch_MSE'])<1e-12; scores.append(score)
    best=int(np.argmin(scores))+1
    assert best==m['best_epoch'] and abs(scores[best-1]-m['best_MSE'])<1e-12
    assert hist[best-1]['state_SHA']==m['selected_state_SHA']
    with np.load(root/'out/selected_DEV_frozen_prediction.npz',allow_pickle=False) as z:
        assert z['ids'].tolist()==p['official_train_dev_IDs']['dev']
        assert str(z['state_SHA'].item())==m['selected_state_SHA']
        selected_prediction=z['prediction'].copy()
    with np.load(root/('out/DEV_epoch_%03d_prediction_only.npz'%best),allow_pickle=False) as z:
        assert np.array_equal(selected_prediction,z['prediction'])
    cache=dict(np.load(root/'out/selected_DEV_original_source.npz',allow_pickle=False))
    assert cache['ids'].tolist()==p['official_train_dev_IDs']['dev']
    assert np.array_equal(cache['prediction'],selected_prediction)
    tail=load_verified_tail(torch.load(root/'out/selected_original_tail.pt',map_location='cpu'),p['candidate_mode'],state['selected_model'])
    msg=torch.load(root/'out/selected_message.pt',map_location='cpu')['state']
    expected={k.removeprefix('dberta.own_flow.core.message.'):v for k,v in state['selected_model'].items() if k.startswith('dberta.own_flow.core.message.')}
    assert tensor_sha(msg)==tensor_sha(expected)==tensor_sha(tail.flow.core.message.state_dict())
    before=tensor_sha(tail.state_dict()); rng=torch.get_rng_state().clone(); pieces=[]
    with torch.no_grad():
        for i in range(0,229,32):
            source=torch.from_numpy(cache['source'][i:i+32]); mask=torch.from_numpy(cache['mask'][i:i+32])
            prediction, off=tail(source,mask)
            assert torch.isfinite(prediction).all() and torch.isfinite(off).all()
            pieces.append(prediction.numpy())
    error=float(np.max(np.abs(np.concatenate(pieces)-selected_prediction)))
    assert error<=1e-4 and tensor_sha(tail.state_dict())==before and torch.equal(rng,torch.get_rng_state())
    write(a.out/'audit_result.json',dict(status='CANDIDATE_FULL4000_ADAM_SELECTION_AND_CPU_TAIL_PASSED',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,candidate_mode=p['candidate_mode'],training_archive_SHA=ref['archive_SHA'],checkpoint_SHA=sha(checkpoint),selected_state_SHA=m['selected_state_SHA'],best_epoch=best,Adam_parameter_states=345,all_Adam_steps=4000,all100_prediction_then_label_selection_recomputed=True,CPU_actual_DEV_flow_maxerror=error,CPU_encoder_forward=False,parameters_buffers_RNG_unchanged=True,TEST_not_indexed=True,extra_replayed_compute_disclosed=p['recovery_compute_disclosure']))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','bundle','assets','input-root','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha',required=True); run(p.parse_args())

