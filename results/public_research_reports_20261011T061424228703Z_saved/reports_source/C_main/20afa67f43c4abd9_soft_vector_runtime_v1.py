"""Frozen-coordinate TRAIN learner and label-free full classifier inference."""
from types import MethodType, SimpleNamespace
import copy, importlib, sys
import numpy as np
import torch
from torch import nn
from task_gradient_vector_candidate_v3 import TaskGradientVectorFeedback

PAIRS=((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))

def configure_paths(root):
    sys.path[:0]=[str(root/'frozen_v5'),str(root/'assets')]

class FrozenCoordinateLearner(nn.Module):
    def __init__(self,root,mode,kappa=.1):
        super().__init__();configure_paths(root)
        from counterfactual_flow_model import WholeStateFlow
        saved=torch.load(root/'teacher_cache_v1/frozen_terminal.pt',map_location='cpu')
        self.flow=WholeStateFlow('none');self.flow.load_state_dict(saved['flow'])
        self.fusion=nn.Sequential(nn.Linear(300,150),nn.ReLU(),nn.Linear(150,100))
        self.predictor=nn.Sequential(nn.Linear(100,150),nn.ReLU(),nn.Linear(150,1))
        self.fusion.load_state_dict(saved['fusion']);self.predictor.load_state_dict(saved['predictor'])
        self.requires_grad_(False);self.eval()
        self.donor=copy.deepcopy(self.flow.donor_feedback);self.donor.requires_grad_(True)
        self.feedback=TaskGradientVectorFeedback(mode,relative_norm2=kappa)
        self.feedback.set_train_fitted_gradient_rms(np.load(root/'teacher_cache_v1/train_gradient_rms.npy'))
    def terminal(self,state,mask,context):
        flow=self.flow;flow.valid=mask.to(state.dtype)
        velocity=torch.stack([flow.forward_fields[m](state[:,m],.5,context[:,m]) for m in range(3)],1)
        final=(state+.5*velocity)*mask[:,None,:,None]
        return flow.read_prediction(final,flow.reader(final,mask.bool()),lambda x:self.predictor(self.fusion(x))).view(-1)
    def forward(self,batch,mode=None):
        pool=batch['pooled_state'];messages=[]
        for i,(m,j) in enumerate(PAIRS):
            features=torch.cat([pool[:,m],pool[:,j],.5*(pool[:,m]-pool[:,j]),.5*(pool[:,m]+pool[:,j])],-1).detach()
            messages.append(self.donor[i](features))
        messages=torch.stack(messages,1)
        if mode is not None:self.feedback.mode=mode
        context,normalized,obs=self.feedback(batch['old_context'],pool,batch['reference_prediction'],messages)
        prediction=self.terminal(batch['state'],batch['mask'],context)
        obs['message']=messages.detach();obs['context']=context.detach()
        return prediction,normalized,obs
    def addon(self):
        return {'donor':{k:v.detach().cpu() for k,v in self.donor.state_dict().items()},'feedback':{k:v.detach().cpu() for k,v in self.feedback.state_dict().items()}}
    def restore_addon(self,addon):
        self.donor.load_state_dict(addon['donor']);self.feedback.load_state_dict(addon['feedback'])

def make_full_reference(root,checkpoint):
    configure_paths(root)
    import json
    from counterfactual_flow_model import WholeStateFlow
    from encoder_adapter import install,forward_v6
    helpers=importlib.import_module('run_careflow')
    cli=SimpleNamespace(backbone=root/'assets/deberta-v3-base',repo=root/'assets/CaReFlow',epochs=100,seed=91814)
    author,args=helpers.load_author(cli);author.set_random_seed(91814)
    model,opt,sch=author.prep_for_training(4000);del opt,sch
    core=model.dberta
    for name in ('reflow_a','reflow_v','reflow_a_b','reflow_v_b','rf_a','rf_v','rf_a_b','rf_v_b'):delattr(core,name)
    core.own_flow=WholeStateFlow('none').to(author.DEVICE)
    protocol=json.loads((root/'teacher_a_metadata/protocol.json').read_text())
    stats={m:{k:torch.tensor(v,dtype=torch.bool if k=='active' else torch.float32) for k,v in fields.items()} for m,fields in protocol['normalization_statistics'].items()}
    install(core,stats);core.to(author.DEVICE);core.forward=MethodType(forward_v6,core)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu'),strict=True)
    model.eval();model.requires_grad_(False);core.own_flow.set_epoch(41)
    return model,author,helpers

def install_vector_inference(core,learner,mode):
    flow=core.own_flow
    flow.donor_feedback.load_state_dict(learner.donor.state_dict())
    flow.vector_feedback=copy.deepcopy(learner.feedback);flow.vector_feedback.mode=mode
    def update(self,old,relation):
        self._calls+=1
        if self._calls>1:return old
        pool=relation['slots'].mean(2);state=relation['stage_states'];mask=self.valid.bool()
        with torch.no_grad():
            velocity=torch.stack([self.forward_fields[m](state[:,m],.5,.5*old[:,m]) for m in range(3)],1)
            final=(state+.5*velocity)*self.valid[:,None,:,None]
            p0=self.read_prediction(final,self.reader(final,mask),self._decoder).view(-1)
        messages=[]
        for i,(m,j) in enumerate(PAIRS):
            features=torch.cat([pool[:,m],pool[:,j],.5*(pool[:,m]-pool[:,j]),.5*(pool[:,m]+pool[:,j])],-1).detach()
            messages.append(self.donor_feedback[i](features))
        context,normalized,obs=self.vector_feedback(old,pool,p0,torch.stack(messages,1))
        # Legacy eval trace fields only; zero legacy utility is explicitly unused.
        weights=obs['actual_weights'];self._stage1={'utility':torch.zeros_like(weights),'weights':weights,'effective_mode':mode}
        self.vector_observations=obs
        return context
    flow.update_context=MethodType(update,flow)
    core.eval();core.requires_grad_(False)
