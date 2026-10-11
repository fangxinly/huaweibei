from pathlib import Path
import datetime,hashlib,json
import numpy as np

root=Path.cwd();source=root/'work/task_gradient_vector_candidate_v2.py'
eps=1e-12
def hard(f,a):
    n2=np.sum(a*a,axis=-1,keepdims=True)
    dot=np.sum(f*a,axis=-1,keepdims=True)
    return f-(n2>eps)*np.maximum(dot,0)/np.maximum(n2,eps)*a
def soft_reference(f,a):
    n2=np.sum(a*a,axis=-1,keepdims=True)
    dot=np.sum(f*a,axis=-1,keepdims=True)
    return f-np.maximum(dot,0)/(n2+eps)*a
f=np.array([1.,0.],dtype=np.float32)
below=np.array([1e-6*(1-1e-4),0.],dtype=np.float32)
above=np.array([1e-6*(1+1e-4),0.],dtype=np.float32)
bh,ah=hard(f,below),hard(f,above)
jump=float(np.linalg.norm(ah-bh)); perturb=float(np.linalg.norm(above-below))
assert jump>.99 and perturb<3e-10
bs,as_=soft_reference(f,below),soft_reference(f,above)
soft_jump=float(np.linalg.norm(as_-bs))
assert soft_jump<.001 and float(np.dot(above,as_))>0

# With the estimated gradient fixed, the active positive-dot branch has
# d(context)/d(F_i)=.125*(I-n*n.T), including the actual two v5 scalings.
a=np.array([1.,0.]); n=a/np.linalg.norm(a); projector=np.eye(2)-np.outer(n,n)
message=np.array([1.,.4]); upstream_normal=np.array([1.,0.]); upstream_tangent=np.array([0.,1.])
expected_normal=.125*projector@upstream_normal
expected_tangent=.125*projector@upstream_tangent
assert np.array_equal(expected_normal,np.zeros(2))
assert np.array_equal(expected_tangent,np.array([0.,.125]))
def context(f):return .125*hard(f,a)
h=1e-5
fd=np.column_stack([(context(message+h*np.eye(2)[i])-context(message-h*np.eye(2)[i]))/(2*h) for i in range(2)])
assert np.max(np.abs(fd-.125*projector))<1e-10

report={'status':'NUMPY_THRESHOLD_COUNTEREXAMPLE_AND_MESSAGE_JACOBIAN_REFERENCE_ONLY',
 'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'epsilon':eps,'boundary_float32':{'gradient_below':below.tolist(),'gradient_above':above.tolist(),
 'message_below':bh.tolist(),'message_above':ah.tolist(),'gradient_perturbation_norm':perturb,
 'message_jump_norm':jump,'jump_to_gradient_change_ratio':jump/perturb},
 'unimplemented_soft_denominator_reference':{'message_change_norm':soft_jump,
 'positive_estimated_dot_after_transform':float(np.dot(above,as_)),
 'limits':'A continuous reference only; positive residual forfeits the exact halfspace guarantee. Not implemented or chosen.'},
 'fixed_estimated_gradient_jacobian':{'analytic':(.125*projector).tolist(),
 'finite_difference_max_error':float(np.max(np.abs(fd-.125*projector))),
 'normal_upstream_message_gradient':expected_normal.tolist(),'tangent_upstream_message_gradient':expected_tangent.tolist(),
 'limits':'Positive-dot active branch only. Piecewise boundaries need separate checks. Zero normal gradient can be intended projection behavior.'},
 'torch_forward_or_backward_executed':False,'gpu_verified':False,'candidate_source_modified':False,
 'new_training_started':False,'train_dev_test_arrays_accessed':False}
(root/'outputs/向量反馈阈值连续性与消息梯度边界核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
