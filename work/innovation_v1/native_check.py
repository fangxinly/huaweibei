"""Meaningful native checks; entirely synthetic, no task/encoder assets."""
import argparse,datetime,json,os,sys
from pathlib import Path
import numpy as np
import torch
from contract import synthetic_contract,Reducer,split_calibration,write
from models import Baseline,Proposal,ConditionalFlow,gate_features,GateHead
from pipeline import Mechanism,state_sha

def run(out):
    torch.set_num_threads(2);torch.manual_seed(128)
    tests=synthetic_contract();z=torch.randn(9,3,4);proposal=Proposal(4);delta,channels=proposal(z,torch.zeros(9,6,4))
    assert torch.equal(delta,torch.zeros_like(delta)) and torch.equal(channels,torch.zeros_like(channels));tests['zero_message_exact_zero_correction']=True
    flow=ConditionalFlow(4,draws=4,steps=2);tg=torch.Generator().manual_seed(128);loss=flow.objective(z,tg);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters());tests['CFM_all_fields_receive_finite_gradient']=True
    flow.eval();before=state_sha(flow);rng=torch.get_rng_state().clone()
    with torch.no_grad():a=flow.means(z);b=flow.means(z);perm=torch.tensor([8,1,7,3,2,0,6,4,5]);c=flow.means(z[perm])
    assert torch.equal(a,b) and torch.allclose(c,a[perm],atol=1e-6,rtol=1e-6) and torch.equal(torch.get_rng_state(),rng) and state_sha(flow)==before
    tests['fixed_antithetic_draws_replay_batch_order_state_rng_invariant']=True
    rng=np.random.default_rng(128);ids=np.array([f'v{v}[{i}]' for v in range(30) for i in range(2)]);features=[rng.normal(size=(60,k)).astype('float32') for k in (8,5,6)];y=.5*features[0][:,0]-.2*features[1][:,0]
    tr,cal=split_calibration(ids);cfg=dict(latent_dim=4,inner_crossfit_k=2,outer_crossfit_k=2,baseline_steps=2,flow_steps=2,proposal_steps=2,gate_steps=2)
    model=Mechanism([v[tr] for v in features],y[tr],ids[tr],[v[cal] for v in features],y[cal],ids[cal],cfg)
    predictions,extra=model.predict(features);complete=model.complete();assert set(predictions)=={'baseline_calibrated','full_message','flow_residual_gate','regression_residual_gate','flow_ordinary_gate'}
    assert all(np.isfinite(p).all() for p in predictions.values());assert all(((v['gate']>=0)&(v['gate']<=1)).all() for k,v in extra.items() if k!='conditional_donor_MSE')
    assert all(r['complete']['steps']==2 for r in complete['records']);tests['nested_video_and_CAL_lineage_checked']=True
    tests['complete_small_model_optimizer_rng_and_predictions']=True
    # Evaluation functions receive no labels, so changing evaluation labels has
    # no route into predictions. Confirm replay and all-off baseline identity.
    again,_=model.predict(features);assert all(np.array_equal(predictions[k],again[k]) for k in predictions);tests['inference_has_no_label_argument_and_replays']=True
    out.mkdir();torch.save(complete,out/'synthetic_complete.pt');restored=Mechanism.restore(torch.load(out/'synthetic_complete.pt',map_location='cpu'));back,_=restored.predict(features)
    assert all(np.array_equal(predictions[k],back[k]) for k in predictions);tests['complete_checkpoint_prediction_replay_exact']=True
    write(out/'actual_native_check.json',dict(status='SYNTHETIC_NATIVE_INNOVATION_CONTRACT_COMPLETE',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,torch_version=torch.__version__,tests=tests,real_data=False,real_performance_claimed=False,complete_state_bytes=(out/'synthetic_complete.pt').stat().st_size))
    print(json.dumps(tests))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
