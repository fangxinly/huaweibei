"""Synthetic original-Torch checks only; no dataset/weights/performance evaluation."""
import copy,datetime,json,os,sys
import numpy as np
import torch
from torch import nn
from official_upgrade import OfficialUpgrade
from controlled_flow import ControlledFlow
from oof_selector import ResidualSelector,check_lineage,PREDECESSORS,video

def rejected(fn):
    try:fn()
    except ValueError:return True
    raise AssertionError('Leakage/invalid-input fixture was accepted')

def run():
    torch.set_num_threads(2);torch.manual_seed(128)
    core=OfficialUpgrade().eval();decoder=nn.Sequential(nn.Linear(300,64),nn.GELU(),nn.Linear(64,1)).eval()
    # Deliberately nonzero, nonlinear SYNTHETIC parameters; never task checkpoints.
    with torch.no_grad():
        core.gain.fill_(.6)
        for h in core.message.heads:h[-1].weight.normal_(std=.05)
        for f in core.forward_fields:f.net[-1].weight.normal_(std=.1)
        core.role_head[-1].weight.normal_(std=.01)
    model=ControlledFlow(core).eval();source=torch.randn(7,3,6,100);mask=torch.ones(7,6,dtype=torch.bool);mask[0,1:]=False
    before={n:v.clone() for n,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    with torch.no_grad():
        packet=model.proposals(source,mask,decoder);full=core(source,mask,decoder)[0]
        original_error=float((full-packet['predictions'][:,-1]).abs().max());assert original_error<1e-6
        p,p0,_,trace=model(source,mask,decoder,labels=torch.zeros(7))
        assert torch.equal(p0,packet['predictions'][:,0]);assert torch.equal(p,packet['predictions'][:,-1])
        assert torch.equal(p,model(source,mask,decoder,labels=torch.full((7,),7.))[0])
        assert torch.equal(model(source,mask,decoder,gates=torch.zeros(6))[0],p0)
        permutation=torch.arange(6,-1,-1)
        perm=model.proposals(source[permutation],mask[permutation],decoder)
        assert torch.allclose(packet['predictions'][permutation],perm['predictions'],atol=1e-6,rtol=1e-6)
        assert torch.allclose(packet['features'][permutation],perm['features'],atol=1e-6,rtol=1e-6)
        padded=source.clone();padded[~mask[:,None,:,None].expand_as(source)]=999
        assert torch.equal(packet['predictions'],model.proposals(padded,mask,decoder)['predictions'])
        selector=ResidualSelector();out,choice,again=model.controlled(source,mask,decoder,selector)
        assert np.array_equal(choice['index'],np.zeros(7)) and torch.equal(out,p0)
        rho=np.linspace(-1.,1.,7);delta=packet['delta'].numpy().astype(float)
        identity=(rho[:,None]**2-(rho[:,None]-delta)**2)-(2*rho[:,None]*delta-delta**2)
        assert abs(identity).max()<1e-14
        nonadditivity=float((packet['delta'][:,-1]-packet['delta'][:,1:7].sum(1)).abs().max())
        assert nonadditivity>1e-7
        rejected(lambda:model.finish(model.prefix(source,mask,decoder),decoder,torch.full((7,6),2.)))
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,model.state_dict()[n]) for n,v in before.items())

    train_ids=[f'train_video_{v}[{j}]' for v in range(10) for j in range(4)]
    cal_ids=[f'cal_video_{v}[{j}]' for v in range(2) for j in range(4)]
    allowed=train_ids+cal_ids;lineage=[]
    for fold in range(5):
        held=[s for s in train_ids if video(s) in {f'train_video_{2*fold}',f'train_video_{2*fold+1}'}]
        trained=[s for s in train_ids if s not in held]
        lineage.append(dict(held_ids=held,predecessors={k:dict(train_ids=trained,state_SHA='a'*64) for k in PREDECESSORS}))
    gen=np.random.default_rng(128);x=gen.normal(size=(40,3));base=gen.normal(size=40);y=base+.4*x[:,0]-.2*x[:,1]
    est=ResidualSelector().fit(x,y,base,train_ids,allowed,cal_ids,lineage)
    cx=gen.normal(size=(8,3));cp=gen.normal(size=8);cy=cp+.4*cx[:,0]-.2*cx[:,1]
    final={k:dict(train_ids=train_ids,state_SHA='b'*64) for k in PREDECESSORS}
    slope=est.calibrate(cx,cy,cp,cal_ids,final);assert slope>0 and np.isfinite(slope)
    d=np.column_stack([np.zeros(8),gen.normal(size=(8,7))]);choice=est.choose(cx,d)
    expected=2*est.slope*est.residual(cx)[:,None]*d-d*d
    assert np.array_equal(choice['index'],expected.argmax(1)) and (choice['estimated_utility']>=0).all()
    est.slope=0.;assert np.array_equal(est.choose(cx,d)['index'],np.zeros(8))
    bad=copy.deepcopy(lineage);bad[0]['predecessors']['encoder']['train_ids'].append(bad[0]['held_ids'][0])
    rejected(lambda:check_lineage(train_ids,allowed,cal_ids,bad))
    bad=copy.deepcopy(lineage);bad[0]['predecessors']['normalization']['train_ids'].append(cal_ids[0])
    rejected(lambda:check_lineage(train_ids,allowed,cal_ids,bad))
    bad=copy.deepcopy(lineage);bad[0]['held_ids'].pop()
    rejected(lambda:check_lineage(train_ids,allowed,cal_ids,bad))
    rejected(lambda:check_lineage(train_ids+['VAL[0]'],allowed,cal_ids,lineage))
    badfinal=copy.deepcopy(final);badfinal['flow']['train_ids'].append(cal_ids[0])
    rejected(lambda:est.calibrate(cx,cy,cp,cal_ids,badfinal))
    return dict(status='SYNTHETIC_DIRECTIONAL_FLOW_AND_OOF_SELECTOR_CONTRACT_COMPLETE',
        actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,
        device='original server CPU',torch=torch.__version__,synthetic_only=True,real_dataset_or_pretrained_weights_used=False,
        actual_new_OOF_training=False,new_VAL_or_TEST_scoring=False,
        original_full_flow_maxerror=original_error,actual_nonlinear_channel_nonadditivity=nonadditivity,
        eight_masks_features=20,zero_gate_same_flow_p0=True,global_state_RNG_unchanged=True,
        dummy_labels_padding_permutation_passed=True,exact_utility_identity_maxerror=float(abs(identity).max()),
        low_capacity_ridge=ResidualSelector.RIDGE,fixture_calibration_slope=slope,
        learned_encoder_normalization_CAL_and_role_leakage_rejected=True,zero_slope_exact_OFF=True,
        risk_guarantee=False,full_crossfit_or_performance_improvement_proved=False)

if __name__=='__main__':print(json.dumps(run()),flush=True)
