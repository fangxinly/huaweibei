"""Synthetic NumPy proximal math and AST only, not Torch/GPU evidence."""
from pathlib import Path
import ast,datetime,hashlib,json
import numpy as np
root=Path(__file__).resolve().parents[1]
source=root/'work/task_gradient_vector_candidate_v3.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
forward=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward']
assert len(forward)==1
assert [v.arg for v in forward[0].args.args]==['self','old_context','pooled_state','reference_prediction','donor_messages']
def prox(f,a,lam):
    n2=np.sum(a*a,axis=-1,keepdims=True)
    dot=np.sum(f*a,axis=-1,keepdims=True)
    return f-np.maximum(dot,0)/(n2+lam)*a
rng=np.random.default_rng(20261005)
f=rng.normal(size=(256,6,100));a=rng.normal(size=f.shape);lam=np.ones((1,6,1))
z=prox(f,a,lam);dot=(a*f).sum(-1,keepdims=True);after=(a*z).sum(-1,keepdims=True)
expected=np.where(dot>0,dot*lam/((a*a).sum(-1,keepdims=True)+lam),dot)
residual=float(np.max(np.abs(after-expected)));assert residual<1e-12
assert np.array_equal(z[dot[...,0]<=0],f[dot[...,0]<=0])
stationarity=z-f+np.maximum(after,0)/lam*a
kkt=float(np.max(np.abs(stationarity)));assert kkt<1e-12
other=rng.normal(size=f.shape);z2=prox(other,a,lam)
nonexpansion=float(np.max(np.linalg.norm(z-z2,axis=-1)-np.linalg.norm(f-other,axis=-1)))
assert nonexpansion<1e-12
orth_delta=z-f;parallel=orth_delta-(orth_delta*a).sum(-1,keepdims=True)/(a*a).sum(-1,keepdims=True)*a
orth_error=float(np.max(np.abs(parallel)));assert orth_error<1e-12
assert np.array_equal(prox(f,np.zeros_like(a),lam),f)
scaled=float(np.max(np.abs(prox(f,5*a,25*lam)-z)));assert scaled<1e-12
old=rng.normal(size=(256,3,100));fixed=.5*old+.125*f.reshape(256,3,2,100).sum(2)
zero_prox=.5*old+.125*prox(f,np.zeros_like(a),lam).reshape(256,3,2,100).sum(2)
assert np.array_equal(fixed,zero_prox)
small_a=np.array([1e-6*(1-1e-4),0],dtype=np.float32)
large_a=np.array([1e-6*(1+1e-4),0],dtype=np.float32)
tiny_lam=np.float32(1e-12);unit=np.array([1.,0.],dtype=np.float32)
boundary_change=float(np.linalg.norm(prox(unit,large_a,tiny_lam)-prox(unit,small_a,tiny_lam)))
assert boundary_change<.001
aa=np.array([1.,2.,0.]);ff=2*aa+np.array([0.,0.,.3]);ll=.7
jac=.125*(np.eye(3)-np.outer(aa,aa)/(np.dot(aa,aa)+ll))
h=1e-5
fd=np.column_stack([.125*(prox(ff+h*np.eye(3)[i],aa,ll)-prox(ff-h*np.eye(3)[i],aa,ll))/(2*h) for i in range(3)])
jac_error=float(np.max(np.abs(jac-fd)));assert jac_error<1e-10
parallel_eigen=float(.125*ll/(np.dot(aa,aa)+ll));assert parallel_eigen>0
# Same nonlinear task counterexample remains: smoothing cannot certify benefit.
curved=prox(np.array([1.,2.]),np.array([1.,0.]),1.)
curvature_increase=float(curved[0]+curved[1]**2);assert curvature_increase==4.5
report={'status':'SOFT_PROXIMAL_NUMPY_KKT_AND_GEOMETRY_VERIFIED_AST_ONLY',
'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
'synthetic_shape':[256,6,100],'reference_lambda':1.,
'reference_lambda_limits':'Synthetic identity, not a selected TRAIN/dev calibration value.',
'positive_dot_residual_identity_error':residual,'convex_objective_stationarity_error':kkt,
'fixed_gradient_nonexpansion_max_residual':nonexpansion,'orthogonal_preservation_error':orth_error,
'gradient_and_lambda_rescaling_equivariance_error':scaled,'zero_gradient_fixed_context_exact':True,
'float32_old_threshold_crossing_output_change':boundary_change,
'active_branch_context_jacobian_max_error':jac_error,'active_branch_parallel_eigen_reference':parallel_eigen,
'nonlinear_counterexample_risk_increase':curvature_increase,
'exact_halfspace_guarantee':False,'torch_executed':False,'gpu_verified':False,
'train_scale_fitted':False,'full_model_integrated':False,'formal_hyperparameter_selected':False,
'new_training_started':False,'train_dev_test_accessed':False}
(root/'outputs/连续向量近端候选数学与静态核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
