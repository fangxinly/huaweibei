from pathlib import Path
import ast,hashlib,json,shutil,datetime
p=Path(__file__).resolve().parent
out=p/'inflow_counterfactual_v5_checks_20261005T0502Z';out.mkdir(exist_ok=False)
old=p/'inflow_utility_v4_20261005T0346Z'
legacy=(old/'legacy_flow_model.py').read_text(encoding='utf-8')
needle='                relation = self.reader(states, mask)\n                if self.variant != "single":'
assert legacy.count(needle)==1
legacy=legacy.replace(needle,'                relation = self.reader(states, mask)\n                relation["stage_states"] = states\n                if self.variant != "single":')
(out/'legacy_flow_model.py').write_text(legacy,encoding='utf-8')
adapter=(old/'encoder_adapter.py').read_text(encoding='utf-8')
needle='    prediction, first, losses, trace = self.own_flow(source, valid,'
assert adapter.count(needle)==1
adapter=adapter.replace(needle,'    self.own_flow.decoder_modules = (self.fusion, self.predictor)\n'+needle)
(out/'encoder_adapter.py').write_text(adapter,encoding='utf-8')
model='''"""TRAIN-only marginal final-task utility, with label-free deterministic reference."""
import torch
from torch import nn
import torch.nn.functional as F
from contextlib import contextmanager
from legacy_flow_model import CONFIG as BASE_CONFIG,WholeStateFlow as BaseFlow
CONFIG=dict(BASE_CONFIG,pair_weight=.025,utility_weight=.01,utility_target_scale=.05,utility_gate_sharpness=4.,shared_fixed_epochs=10)
PAIRS=((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))
@contextmanager
def deterministic_modules(modules):
    flags={m:m.training for root in modules for m in root.modules()}
    for module in flags:module.training=False
    try:yield
    finally:
        for module,flag in flags.items():module.training=flag
class WholeStateFlow(BaseFlow):
    def __init__(self,mode):
        assert mode in ('none','fixed','predicted')
        super().__init__('mid');del self.feedback
        self.mode=mode;self.context_override=None;self.epoch=1;self.decoder_modules=()
        d=CONFIG['dimension']
        self.pair_heads=nn.ModuleList([nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1)) for _ in PAIRS])
        self.utility_heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(3*d+1),nn.Linear(3*d+1,d),nn.GELU(),nn.Linear(d,1)) for _ in PAIRS])
        self.donor_feedback=nn.ModuleList([nn.Sequential(nn.LayerNorm(4*d),nn.Linear(4*d,d),nn.GELU(),nn.Linear(d,d)) for _ in PAIRS])
        for net in list(self.utility_heads)+list(self.donor_feedback):
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
        self._calls=0;self._stage1=None;self._decoder=None
    def set_epoch(self,epoch):
        assert 1<=epoch<=100;self.epoch=int(epoch)
    def context(self,old,feedback,weights):
        selected=.5*(weights[:,:,None]*feedback).reshape(len(old),3,2,-1).sum(2)
        return .5*old+.5*selected
    def update_context(self,old,relation):
        self._calls+=1
        if self._calls>1:return old
        pooled=relation['slots'].mean(2)
        own=torch.stack([self.unimodal_heads[m](pooled[:,m]).view(-1) for m in range(3)],1)
        pairs=[];feedback=[]
        for i,(m,j) in enumerate(PAIRS):
            pairs.append(self.pair_heads[i](torch.cat([pooled[:,m],pooled[:,j]],-1)).view(-1))
            fi=torch.cat([pooled[:,m],pooled[:,j],.5*(pooled[:,m]-pooled[:,j]),.5*(pooled[:,m]+pooled[:,j])],-1).detach()
            feedback.append(self.donor_feedback[i](fi))
        feedback=torch.stack(feedback,1);pairs=torch.stack(pairs,1)
        # References share exactly the first-stage state; no encoder re-run.
        was_training=self.training
        with torch.no_grad(),deterministic_modules((self,*self.decoder_modules)):
            state=relation['stage_states'].detach();valid=self.valid[:,None,:,None]
            fixed_weights=torch.full((len(old),6),.5,device=state.device,dtype=state.dtype)
            ref_context=self.context(old.detach(),feedback.detach(),fixed_weights)
            def terminal(context):
                velocity=torch.stack([self.forward_fields[m](state[:,m],.5,context[:,m]) for m in range(3)],1)
                terminal_state=(state+.5*velocity)*valid
                return self.read_prediction(terminal_state,self.reader(terminal_state,self.valid.bool()),self._decoder).view(-1)
            reference=terminal(ref_context)
            without=[];joint=[]
            if was_training:
                for i in range(6):
                    w=fixed_weights.clone();w[:,i]=0
                    without.append(terminal(self.context(old.detach(),feedback.detach(),w)))
                for m in range(3):
                    w=fixed_weights.clone();w[:,2*m:2*m+2]=0
                    joint.append(terminal(self.context(old.detach(),feedback.detach(),w)))
            without=torch.stack(without,1) if without else None
            joint=torch.stack(joint,1) if joint else None
        utility=[]
        for i,(m,j) in enumerate(PAIRS):
            features=torch.cat([pooled[:,m],pooled[:,j],feedback[:,i],torch.tanh(reference[:,None]/3)],-1).detach()
            utility.append(torch.tanh(self.utility_heads[i](features).view(-1)))
        utility=torch.stack(utility,1);predicted_weights=torch.sigmoid(CONFIG['utility_gate_sharpness']*utility)
        mode='fixed' if self.epoch<=CONFIG['shared_fixed_epochs'] else self.mode
        if self.context_override is not None:mode=self.context_override
        weights={'none':torch.zeros_like(predicted_weights),'fixed':torch.full_like(predicted_weights,.5),'predicted':predicted_weights}[mode]
        self._stage1={'own':own,'pair':pairs,'utility':utility,'weights':weights,'predicted_weights':predicted_weights,
            'reference_prediction':reference,'without_prediction':without,'joint_without_prediction':joint,'effective_mode':mode,
            'feedback_norm':feedback.detach().norm(dim=-1)}
        return self.context(old,feedback,weights)
    def forward(self,source,mask,decoder,labels=None):
        self._calls=0;self._stage1=None;self._decoder=decoder
        prediction,first,losses,trace=super().forward(source,mask,decoder,labels)
        assert self._calls==2 and self._stage1 is not None
        s=self._stage1
        if self.training:
            assert labels is not None
            y=labels.view(-1,1);reference=s['reference_prediction'][:,None]
            g=((s['without_prediction']-y).square()-(reference-y).square()).detach()
            target=torch.tanh(g/CONFIG['utility_target_scale']);assert not target.requires_grad
            per=F.smooth_l1_loss(s['utility'],target,reduction='none')
            balanced=[]
            for i in range(6):
                positive=g[:,i]>0;negative=g[:,i]<0
                groups=[per[mask,i].mean() for mask in (positive,negative) if mask.any()]
                balanced.append(torch.stack(groups).mean() if groups else per[:,i].mean())
            losses['utility_calibration']=torch.stack(balanced).mean()
            own_error=(s['own']-y).square()
            losses['source_unimodal_sentiment']=losses['unimodal_sentiment']
            losses['stage1_observer_sentiment']=own_error.mean()
            losses['unimodal_sentiment']=.5*(losses['source_unimodal_sentiment']+losses['stage1_observer_sentiment'])
            losses['pair_sentiment']=(s['pair']-y).square().mean()
            joint_g=((s['joint_without_prediction']-y).square()-(reference-y).square()).detach()
            s['raw_utility']=g;s['target']=target;s['joint_utility']=joint_g
            s['interaction_residual']=joint_g-g.reshape(len(g),3,2).sum(2)
        trace['utility_weights']=s['weights'].detach();trace['utility_prediction']=s['utility'].detach();trace['effective_mode']=s['effective_mode'];trace['epoch']=self.epoch
        return prediction,first,losses,trace
'''
(out/'counterfactual_flow_model.py').write_text(model,encoding='utf-8')
for f in out.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
plan={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'Mechanism checks only; no 100-epoch training launch authorized by this plan','seed':91814,'epochs_candidate':100,'shared_fixed_epochs':10,'reference_paths_training':10,'reference_paths_eval':1,'target_scale':.05,'utility_loss':'Per-direction SmoothL1 with positive/negative groups equally averaged when both exist, zero-only batch all mean','budget_check_updates':20,'test_accessed':False,'limits':'Space dependency unresolved. Ten references include three joint removals for interaction reporting. No semantic truth claim.'}
(out/'check_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'directory':str(out),'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('*.py')}}))
