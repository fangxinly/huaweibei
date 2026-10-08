"""Linear task-free conditional prediction inside the original first Euler states.

This is an approximation to conditional predictability, not PID, an identified
information decomposition or a conditional flow-matching transport.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from anchored_flow import PAIRS
from incremental_message import IncrementalMessage
from fold_contract import video_id

def grouped_folds(ids,k=5):
    videos=np.asarray([video_id(str(s)) for s in ids]);counts={v:int((videos==v).sum()) for v in set(videos)};bins=[[] for _ in range(k)];sizes=[0]*k
    for v in sorted(counts,key=lambda v:(-counts[v],hashlib.sha256(('128:innovation:'+v).encode()).hexdigest())):
        j=min(range(k),key=lambda j:(sizes[j],len(bins[j]),j));bins[j].append(v);sizes[j]+=counts[v]
    folds=[np.flatnonzero(np.isin(videos,b)) for b in bins];assert len(set(videos))>=k and sorted(np.concatenate(folds).tolist())==list(range(len(ids)))
    for f in folds:assert not set(videos[f]) & set(videos[np.setdiff1d(np.arange(len(ids)),f)])
    return folds,bins

def fit_maps(slots):
    z=np.asarray(slots,dtype=np.float64);maps={k:[] for k in ('mean_x','std_x','mean_y','std_y','coefficient')}
    for receiver,donor in PAIRS:
        x,y=z[:,receiver],z[:,donor];mx=x.mean(0);my=y.mean(0);sx=np.maximum(x.std(0),1e-6);sy=np.maximum(y.std(0),1e-6);X=(x-mx)/sx;Y=(y-my)/sy
        B=np.linalg.solve(X.T@X+.01*len(X)*np.eye(100),X.T@Y)
        for k,v in zip(maps,(mx,sx,my,sy,B)):maps[k].append(v)
    return {k:np.stack(v) for k,v in maps.items()}

def innovations(slots,maps):
    z=np.asarray(slots,dtype=np.float64);result=[]
    for i,(receiver,donor) in enumerate(PAIRS):
        pred=((z[:,receiver]-maps['mean_x'][i])/maps['std_x'][i])@maps['coefficient'][i]*maps['std_y'][i]+maps['mean_y'][i]
        result.append(z[:,donor]-pred)
    out=np.stack(result,1).astype(np.float32);assert np.isfinite(out).all();return out

def crossfit(slots,ids):
    z=np.asarray(slots);folds,videos=grouped_folds(ids);oof=np.zeros((len(z),6,100),dtype=np.float32);partial=[]
    for f in folds:
        train=np.setdiff1d(np.arange(len(z)),f);m=fit_maps(z[train]);oof[f]=innovations(z[f],m);partial.append(m)
    full=fit_maps(z);summary=dict(rows=len(z),video_count=sum(map(len,videos)),fold_rows=[len(f) for f in folds],held_video_ids=videos,ridge=.01,label_access=False,OOF_only_for_conditional_predictor_not_encoder=True,full_innovation_variance=float(innovations(z,full).var()),oof_innovation_variance=float(oof.var()),donor_variance=float(z.var()))
    return full,oof,partial,summary

class InnovationMessage(IncrementalMessage):
    def __init__(self,maps=None):
        super().__init__()
        if maps is None:maps={k:(np.ones((6,100)) if k.startswith('std') else np.zeros((6,100,100)) if k=='coefficient' else np.zeros((6,100))) for k in ('mean_x','std_x','mean_y','std_y','coefficient')}
        for k,v in maps.items():self.register_buffer('conditional_'+k,torch.as_tensor(v,dtype=torch.float64).clone())
    def maps_numpy(self):return {k:getattr(self,'conditional_'+k).detach().cpu().numpy() for k in ('mean_x','std_x','mean_y','std_y','coefficient')}
    def new_information(self,slots):
        z=slots.double();out=[]
        for i,(receiver,donor) in enumerate(PAIRS):
            predicted=((z[:,receiver]-self.conditional_mean_x[i])/self.conditional_std_x[i])@self.conditional_coefficient[i]*self.conditional_std_y[i]+self.conditional_mean_y[i]
            out.append(z[:,donor]-predicted)
        return torch.stack(out,1).to(slots.dtype)
    def forward(self,slots,innovations=None,direction_mask=None):
        innovation=self.new_information(slots) if innovations is None else innovations;feedback=[]
        for i,(receiver,donor) in enumerate(PAIRS):feedback.append(self.channel(i,slots[:,receiver],innovation[:,i]))
        directional=torch.stack(feedback,1)
        if direction_mask is not None:directional=directional*direction_mask[...,None]
        return .125*directional.reshape(len(slots),3,2,100).sum(2)

def native_innovation_check(device):
    rng=np.random.default_rng(128);slots=rng.normal(size=(75,3,100)).astype(np.float32);ids=[f'v{i//3}[{i%3}]' for i in range(75)];maps,oof,partials,summary=crossfit(slots,ids);model=InnovationMessage(maps).to(device);s=torch.from_numpy(slots[:8]).to(device)
    with torch.no_grad():
        actual=model.new_information(s).cpu().numpy();assert np.max(abs(actual-innovations(slots[:8],maps)))<2e-6
    # The registered full-map transformation has no gradients and is label-free.
    assert all(not x.requires_grad for n,x in model.named_buffers());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    for _ in range(3):
        opt.zero_grad();c=model(s);loss=(c-.1).square().mean();loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad);opt.step()
    with torch.no_grad():
        assert torch.equal(model(s,innovations=torch.zeros((8,6,100),device=device)),torch.zeros((8,3,100),device=device));assert torch.equal(model(s,direction_mask=torch.zeros((8,6),device=device)),torch.zeros((8,3,100),device=device))
    return dict(device=device,video_conditional_predictor_folds_disjoint=True,torch_numpy_full_map_error=float(np.max(abs(actual-innovations(slots[:8],maps)))),zero_innovation_and_all_off_exact=True,conditional_buffers_not_trainable=True,finite_small_head_gradients=True,parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),map_fit_labels_read=False)
