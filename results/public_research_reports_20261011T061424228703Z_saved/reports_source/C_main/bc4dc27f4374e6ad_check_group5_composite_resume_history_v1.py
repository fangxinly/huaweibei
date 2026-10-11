"""Counterexamples for prepared resume history; no scientific imports or inputs."""
import copy,json
from group5_composite_epoch_resume_v1 import history_state
def reject(method,h):
 try:history_state(method,h,2)
 except (ValueError,PermissionError):return
 raise AssertionError('Invalid history accepted')
m=[dict(epoch=1,updates=2,state_SHA='a'),dict(epoch=2,updates=4,state_SHA='b')]
a=[dict(epoch=1,updates=2,state_SHA='a',inner_MSE=1.,best_MSE=1.,best_epoch=1),dict(epoch=2,updates=4,state_SHA='b',inner_MSE=1.,best_MSE=1.,best_epoch=1)]
assert history_state('anchored_message20',m,2)['completed_epochs']==2
assert history_state('old_fixed_A',a,2)['best_epoch']==1
reject('anchored_message20',[])
x=copy.deepcopy(m);x[1]['epoch']=3;reject('anchored_message20',x)
x=copy.deepcopy(m);x[1]['updates']=3;reject('anchored_message20',x)
x=copy.deepcopy(m);x[0]['inner_MSE']=1.;reject('anchored_message20',x)
x=copy.deepcopy(a);x[1]['best_epoch']=2;reject('old_fixed_A',x)
x=copy.deepcopy(a);x[0]['inner_MSE']=float('nan');reject('old_fixed_A',x)
reject('unknown',m)
reject('anchored_message20',[dict(epoch=i,updates=2*i,state_SHA='x') for i in range(1,21)])
print(json.dumps(dict(status='SYNTHETIC_COMPOSITE_HISTORY_10_CASES_PASS',scientific_imports=False,original_arrays_labels_models_loaded=False,full_model_Adam_RNG_resume_executed=False,CUDA_resume_executed=False)))
