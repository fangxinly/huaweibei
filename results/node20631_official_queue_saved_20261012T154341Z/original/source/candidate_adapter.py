"""Training/export glue for the original flow; no data access in this module."""
import copy
import torch
from torch import nn
from official_upgrade import OfficialUpgrade
from polarity_intensity_flow import PolarityIntensityFlow,train_objective

def make_candidate(mode):
    return PolarityIntensityFlow(OfficialUpgrade(),mode)

def candidate_objective(model,prediction,labels):
    flow=model.dberta.own_flow
    if prediction.shape!=flow.last['prediction'].shape:
        raise ValueError('Prediction and current candidate trace must align')
    loss,_=train_objective(flow,labels,'TRAIN')
    return loss

def optimizer_groups(model):
    flow=model.dberta.own_flow
    gain=flow.core.gain
    special=list(flow.core.message.parameters())+list(flow.polarity_delta.parameters())+list(flow.magnitude_delta.parameters())
    specialids={id(v) for v in special}|{id(gain)}
    named=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
    ordinary=[(n,v) for n,v in named if id(v) not in specialids]
    nd=('bias','LayerNorm.bias','LayerNorm.weight')
    groups=[dict(params=[v for n,v in ordinary if not any(k in n for k in nd)],lr=1e-5,weight_decay=.01),
            dict(params=[v for n,v in ordinary if any(k in n for k in nd)],lr=1e-5,weight_decay=0.),
            dict(params=[gain],lr=.001,weight_decay=0.),
            dict(params=[v for v in special if v.requires_grad],lr=.001,weight_decay=.01)]
    optimized=[id(v) for group in groups for v in group['params']]
    assert len(optimized)==len(set(optimized)) and set(optimized)=={id(v) for _,v in named}
    return groups,named

class CandidateTail(nn.Module):
    def __init__(self,core):
        super().__init__()
        if not isinstance(core.own_flow,PolarityIntensityFlow):
            raise ValueError('Candidate tail cannot silently use the old regression tail')
        # Rebuild modules rather than copying forward traces with autograd graphs.
        # Constructor must not consume the training model's CPU RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.flow=make_candidate(core.own_flow.mode).cpu()
        self.flow.load_state_dict({k:v.detach().cpu().clone() for k,v in core.own_flow.state_dict().items()},strict=True)
        self.fusion=copy.deepcopy(core.fusion).cpu()
        self.predictor=copy.deepcopy(core.predictor).cpu()
        self.eval().requires_grad_(False)

    def forward(self,source,mask):
        return self.flow(source,mask,lambda x:self.predictor(self.fusion(x)))[:2]
