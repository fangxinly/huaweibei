"""C2 analytic single-token second-order numerical amendment; frozen v1 intact."""
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner as V1,make_full_reference,PAIRS,install_finite_inference as install_v1
from finite_single_token_reader_v1 import install_for_flow

class FrozenCoordinateLearner(V1):
 def __init__(self,root,mode,scales):
  super().__init__(root,mode,scales);install_for_flow(self.flow)

def install_finite_inference(core,learner,mode):
 install_for_flow(core.own_flow);install_v1(core,learner,mode)
