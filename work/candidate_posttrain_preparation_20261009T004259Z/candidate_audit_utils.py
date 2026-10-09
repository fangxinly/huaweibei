"""Reconstruct only the saved small candidate tail, never the encoder."""
from types import SimpleNamespace
from torch import nn
import torch
from candidate_adapter import CandidateTail, make_candidate
from fixed_flow_components_candidate import tensor_sha

def make_tail(mode):
    core = SimpleNamespace(own_flow=make_candidate(mode),
        fusion=nn.Sequential(nn.Linear(300,150), nn.ReLU(), nn.Linear(150,100)),
        predictor=nn.Sequential(nn.Linear(100,150), nn.ReLU(), nn.Linear(150,1)))
    return CandidateTail(core)

def tail_model_key(name):
    prefix, rest = name.split('.', 1)
    return {'flow':'dberta.own_flow', 'fusion':'dberta.fusion',
            'predictor':'dberta.predictor'}[prefix] + '.' + rest

def load_verified_tail(saved, mode, selected):
    assert saved['mode'] == mode
    tail = make_tail(mode)
    tail.load_state_dict(saved['state'], strict=True)
    for name, value in tail.state_dict().items():
        assert torch.equal(value, selected[tail_model_key(name)]), name
    tail.eval().requires_grad_(False)
    return tail

