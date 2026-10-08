"""Original PyTorch CPU contract to execute when a qualified runtime is available.

This script has not been run merely because it exists. Uses synthetic tensors,
never real data, pretrained weights, VAL/TEST labels or GPU.
"""
import copy,json,os,sys,datetime
from pathlib import Path
import torch
from torch import nn
from official_upgrade import OfficialUpgrade
from polarity_intensity_flow import PolarityIntensityFlow,train_objective

def run():
    torch.set_num_threads(2);torch.manual_seed(128)
    core=OfficialUpgrade();decoder=nn.Linear(300,1)
    rng=torch.get_rng_state().clone()
    factor=PolarityIntensityFlow(core,'factorized_aux')
    assert torch.equal(rng,torch.get_rng_state())
    regression=PolarityIntensityFlow(copy.deepcopy(core),'regression_aux')
    assert sum(p.numel() for p in factor.polarity_delta.parameters())+sum(p.numel() for p in factor.magnitude_delta.parameters())==2002
    x=torch.randn(7,3,8,100);mask=torch.ones(7,8,dtype=torch.bool);mask[0,1:]=False;mask[1,5:]=False
    factor.eval();regression.eval();core.eval()
    q=core(x,mask,decoder)[0]
    assert torch.equal(regression(x,mask,decoder)[0],q)
    pred=factor(x,mask,decoder)[0]
    assert torch.allclose(pred,q,atol=2e-6,rtol=2e-6)
    assert torch.all(factor.last['magnitude']>0)
    with torch.no_grad():
        # Ensure a genuine intervention after synthetic nonzero message weights.
        core.gain.fill_(.3)
        for h in core.message.heads:h[-1].weight.normal_(0,.03)
    values=factor.proposals(x,mask,decoder)
    p,p0,_,_=factor(x,mask,decoder)
    assert torch.equal(p,values['predictions'][:,-1]) and torch.equal(p0,values['predictions'][:,0])
    assert (p-p0).abs().max()>1e-7
    before={k:v.clone() for k,v in factor.state_dict().items()};rng=torch.get_rng_state().clone()
    with torch.no_grad():
        assert torch.equal(p,factor(x,mask,decoder,torch.zeros(7))[0])
        assert torch.equal(p,factor(x,mask,decoder,torch.ones(7)*7)[0])
        bad=x.clone();bad[~mask[:,None,:,None].expand_as(x)]=999
        assert torch.equal(p,factor(bad,mask,decoder)[0])
        order=torch.arange(6,-1,-1)
        assert torch.allclose(p[order],factor(x[order],mask[order],decoder)[0],atol=2e-6,rtol=2e-6)
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,factor.state_dict()[k]) for k,v in before.items())
    factor.train();y=torch.tensor([-2.,-1.,-.1,0.,.1,1.,2.])
    opt=torch.optim.AdamW(list(factor.parameters())+list(decoder.parameters()),lr=.001)
    for _ in range(3):
        opt.zero_grad();factor(x,mask,decoder);loss,parts=train_objective(factor,y,'TRAIN');loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in factor.parameters() if p.requires_grad)
        opt.step()
    for role in ('VAL','TEST','INNER'):
        try:train_objective(factor,y,role)
        except ValueError:pass
        else:raise AssertionError('Forbidden fitting role accepted')
    return dict(status='NATIVE_SYNTHETIC_POLARITY_INTENSITY_CONTRACT_COMPLETE',
        actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,
        torch=torch.__version__,device='CPU',real_dataset_or_weights=False,new_VAL_TEST_scores=False,
        equal_capacity_auxiliary_control=True,additional_parameters=2002,
        labels_padding_permutation_state_RNG_invariants=True,same_flow_OFF_ON_interventions=True,
        zero_initialized_product_identity_inside_anchor_range=True,TRAIN_only_update_guard=True,
        finite_gradients=True,OOF_utility_controller=False)

if __name__=='__main__':
    result=run()
    if len(sys.argv)!=2:raise ValueError('Explicit result JSON path required')
    Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
