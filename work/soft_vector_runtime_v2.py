"""Jacobian-aligned candidate; original v1 frozen runtime unchanged."""
import copy
from types import MethodType
import numpy as np
import torch
from soft_vector_runtime_v1 import FrozenCoordinateLearner as BaseLearner,make_full_reference,PAIRS
from soft_vector_jacobian_candidate_v1 import JacobianAlignedTaskGradientFeedback

class FrozenCoordinateLearner(BaseLearner):
    def __init__(self,root,mode,kappa=.1):
        super().__init__(root,mode,kappa)
        self.feedback=JacobianAlignedTaskGradientFeedback(mode,relative_norm2=kappa)
        self.feedback.set_train_fitted_gradient_rms(np.load(root/'teacher_cache_v1/train_gradient_rms.npy'))
    def forward(self,batch,mode=None):
        pool=batch['pooled_state'];messages=[]
        for i,(m,j) in enumerate(PAIRS):
            features=torch.cat([pool[:,m],pool[:,j],.5*(pool[:,m]-pool[:,j]),.5*(pool[:,m]+pool[:,j])],-1).detach()
            messages.append(self.donor[i](features))
        if mode is not None:self.feedback.mode=mode
        context,normalized,obs=self.feedback(batch['old_context'],pool,batch['reference_prediction'],torch.stack(messages,1),batch['reference_jacobian'])
        prediction=self.terminal(batch['state'],batch['mask'],context)
        obs['message']=torch.stack(messages,1).detach();obs['context']=context.detach()
        return prediction,normalized,obs

def install_vector_inference(core,learner,mode):
    flow=core.own_flow;flow.donor_feedback.load_state_dict(learner.donor.state_dict())
    flow.vector_feedback=copy.deepcopy(learner.feedback);flow.vector_feedback.mode=mode
    def update(self,old,relation):
        self._calls+=1
        if self._calls>1:return old
        pool=relation['slots'].mean(2);state=relation['stage_states'].detach();mask=self.valid.bool()
        # Even inside no_grad inference, compute only label-free context derivative.
        with torch.enable_grad():
            ctx=(.5*old).detach().requires_grad_()
            velocity=torch.stack([self.forward_fields[m](state[:,m],.5,ctx[:,m]) for m in range(3)],1)
            final=(state+.5*velocity)*self.valid[:,None,:,None]
            p0=self.read_prediction(final,self.reader(final,mask),self._decoder).view(-1)
            jacobian=torch.autograd.grad(p0.sum(),ctx,create_graph=False)[0].detach();p0=p0.detach()
        messages=[]
        for i,(m,j) in enumerate(PAIRS):
            features=torch.cat([pool[:,m],pool[:,j],.5*(pool[:,m]-pool[:,j]),.5*(pool[:,m]+pool[:,j])],-1).detach()
            messages.append(self.donor_feedback[i](features))
        context,normal,obs=self.vector_feedback(old,pool,p0,torch.stack(messages,1),jacobian)
        weights=obs['actual_weights'];self._stage1={'utility':torch.zeros_like(weights),'weights':weights,'effective_mode':mode};self.vector_observations=obs
        return context
    flow.update_context=MethodType(update,flow);core.eval();core.requires_grad_(False)
