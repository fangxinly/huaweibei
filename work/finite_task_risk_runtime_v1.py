"""New frozen-coordinate runtime; earlier scientific sources unchanged."""
import copy
from types import MethodType
import torch
from soft_vector_runtime_v1 import FrozenCoordinateLearner as Base,make_full_reference,PAIRS
from finite_task_risk_feedback_v1 import FiniteTaskRiskFeedback


class FrozenCoordinateLearner(Base):
    def __init__(self,root,mode,scales):
        super().__init__(root,'fixed',.1)
        self.feedback=FiniteTaskRiskFeedback(mode)
        self.feedback.set_train_scales(scales['residual_rms'],scales['message_rms'],scales['beta'])

    def forward(self,batch,mode=None):
        pool=batch['pooled_state'].detach();messages=[]
        for i,(m,j) in enumerate(PAIRS):
            features=torch.cat([pool[:,m],pool[:,j],.5*(pool[:,m]-pool[:,j]),.5*(pool[:,m]+pool[:,j])],-1)
            messages.append(self.donor[i](features))
        messages=torch.stack(messages,1)
        terminal=lambda context:self.terminal(batch['state'].detach(),batch['mask'],context)
        context,normalized,obs=self.feedback(batch['old_context'],pool,batch['reference_prediction'],messages,terminal,mode)
        prediction=terminal(context)
        return prediction,normalized,obs


def install_finite_inference(core,learner,mode):
    flow=core.own_flow;flow.donor_feedback.load_state_dict(learner.donor.state_dict())
    flow.vector_feedback=copy.deepcopy(learner.feedback);flow.vector_feedback.mode=mode
    def update(self,old,relation):
        self._calls+=1
        if self._calls>1:return old
        state=relation['stage_states'].detach();pool=relation['slots'].mean(2).detach();mask=self.valid.bool()
        def terminal(context):
            velocity=torch.stack([self.forward_fields[m](state[:,m],.5,context[:,m]) for m in range(3)],1)
            final=(state+.5*velocity)*self.valid[:,None,:,None]
            return self.read_prediction(final,self.reader(final,mask),self._decoder).view(-1)
        with torch.no_grad():p0=terminal(.5*old).detach()
        messages=[]
        for i,(m,j) in enumerate(PAIRS):
            features=torch.cat([pool[:,m],pool[:,j],.5*(pool[:,m]-pool[:,j]),.5*(pool[:,m]+pool[:,j])],-1).detach()
            messages.append(self.donor_feedback[i](features))
        context,_,obs=self.vector_feedback(old,pool,p0,torch.stack(messages,1),terminal,mode)
        self._stage1={'utility':torch.zeros_like(obs['actual_weights']),
                      'weights':obs['actual_weights'],'effective_mode':mode}
        self.vector_observations=obs
        return context
    flow.update_context=MethodType(update,flow);core.eval();core.requires_grad_(False)
