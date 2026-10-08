"""Independent original Torch CPU replay of saved second-step tail and new adapter."""
import argparse,os,sys
from pathlib import Path
import numpy as np
import torch
from common import sha,read,write,utc,verify
from incremental_message import OriginalTail,IncrementalMessage
from fixed_flow_components_candidate import tensor_sha
from train_increment import predict
def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets);a.out.mkdir();torch.set_num_threads(2)
    for path,key in ((a.cache,'cache_reference'),(a.tail,'tail_reference'),(a.state,'changed_state_reference'),(a.prediction,'prediction_reference')):assert sha(path)==p[key]['SHA'],key
    saved=torch.load(a.state,map_location='cpu');assert saved['updates']==940 and saved['plan_SHA']==p['training_plan_SHA'];assert len(saved['history'])==20 and saved['orders'].shape==(20,1494)
    assert all(np.array_equal(np.sort(r),np.arange(1494)) for r in saved['orders']);assert saved['parent']['full_SHA']==p['parent']['full_SHA'];assert saved['cache']['SHA']==p['cache_reference']['SHA'];assert saved['tail']['SHA']==p['tail_reference']['SHA']
    msg=IncrementalMessage();msg.load_state_dict(saved['adapter'],strict=True);msg.eval();assert tensor_sha(msg.state_dict())==saved['adapter_state_SHA'];assert len(saved['optimizer']['state'])==len([v for v in msg.parameters() if v.requires_grad]);assert all(int(v['step'])==940 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in saved['optimizer']['state'].values())
    parent_tail=torch.load(a.tail,map_location='cpu');tail=OriginalTail();tail.load_state_dict(parent_tail['state'],strict=True);c=dict(np.load(a.cache,allow_pickle=False));pred=dict(np.load(a.prediction,allow_pickle=False));before=tensor_sha(tail.state_dict());mbefore=tensor_sha(msg.state_dict());rng=torch.get_rng_state().clone();errors={}
    for role in ('fit','inner'):
        assert c[role+'_ids'].tolist()==pred[role+'_ids'].tolist();actual,off,context=predict(tail,msg,c,role,'cpu');errors[role]=dict(prediction=float(np.max(abs(actual-pred[role+'_prediction']))),p0=float(np.max(abs(off-pred[role+'_p0']))),context=float(np.max(abs(context-pred[role+'_context']))));assert max(errors[role].values())<=p['CPU_replay_tolerance'],errors
        # Actual trained direction must vanish for an absent donor.
        s=torch.from_numpy(c[role+'_slots'][:8])
        for i,(own,donor) in enumerate(__import__('anchored_flow').PAIRS):assert torch.equal(msg.channel(i,s[:,own],torch.zeros_like(s[:,donor])),torch.zeros_like(s[:,own]))
    assert before==tensor_sha(tail.state_dict()) and mbefore==tensor_sha(msg.state_dict()) and torch.equal(rng,torch.get_rng_state())
    write(a.out/'CPU_replay_result.json',dict(status='ORIGINAL_TAIL_NEW_MESSAGE_SAVED_STATE_CPU_REPLAY_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,errors=errors,optimizer_all_steps_940=True,complete_changed_parameters_and_two_moments=True,twenty_orders_complete=True,tail_and_adapter_state_rng_unchanged=True,trained_zero_donor_exact=True,encoder_CPU_model_forward=False,composite_parent_reference=p['parent']))
    print(read(a.out/'CPU_replay_result.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','cache','tail','state','prediction','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())
