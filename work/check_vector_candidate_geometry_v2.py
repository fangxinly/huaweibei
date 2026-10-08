"""NumPy reference checks only; AST is not a PyTorch gradient/GPU verification."""
from pathlib import Path
import ast,datetime,hashlib,json
import numpy as np
root=Path(__file__).resolve().parents[1];source=root/'work/task_gradient_vector_candidate_v2.py';tree=ast.parse(source.read_text(encoding='utf-8'))
forward=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward'];assert len(forward)==1
assert [a.arg for a in forward[0].args.args]==['self','old_context','pooled_state','reference_prediction','donor_messages']
def project(f,a,eps=1e-12):
 n=(a*a).sum(-1,keepdims=True);dot=(f*a).sum(-1,keepdims=True);return f-(n>eps)*np.maximum(dot,0)/np.maximum(n,eps)*a
rng=np.random.default_rng(20261005);f=rng.normal(size=(256,6,100));a=rng.normal(size=f.shape);filtered=project(f,a)
dot=(f*a).sum(-1);after=(filtered*a).sum(-1);assert after.max()<1e-10
assert np.array_equal(filtered[dot<=0],f[dot<=0])
before_orth=f-dot[...,None]/(a*a).sum(-1,keepdims=True)*a
after_orth=filtered-after[...,None]/(a*a).sum(-1,keepdims=True)*a
orth_error=float(np.max(np.abs(before_orth-after_orth)));assert orth_error<1e-12
zero=np.zeros_like(a);assert np.array_equal(project(f,zero),f)
small=np.full_like(a,1e-10);assert np.array_equal(project(f,small),f)
old=rng.normal(size=(256,3,100));fixed=.5*old+.25*(.5*f).reshape(256,3,2,100).sum(2)
scalar=.5*old+.25*(np.full((256,6,1),.5)*f).reshape(256,3,2,100).sum(2)
proj=.5*old+.25*(.5*project(f,zero)).reshape(256,3,2,100).sum(2)
assert np.array_equal(fixed,scalar) and np.array_equal(fixed,proj)
truth=a+.1*rng.normal(size=a.shape);true_dot=(filtered*truth).sum(-1);bound=np.linalg.norm(truth-a,axis=-1)*np.linalg.norm(filtered,axis=-1);assert np.max(true_dot-bound)<1e-10
# Nonlinear counterexample: projection may leave a curvature-harmful orthogonal
# displacement. L(c)=c[0]+c[1]^2, grad at zero=(1,0), F=(1,2).
fc=np.array([[1.,2.]]);ac=np.array([[1.,0.]]);pc=project(fc,ac);assert np.array_equal(pc,np.array([[0.,2.]]));nonlinear_increase=float(pc[0,0]+pc[0,1]**2);assert nonlinear_increase==4
k=601;d=100;symbolic_parameters=3*(2*k+k*d+d+d*d+d);assert symbolic_parameters==214506
report={'status':'VECTOR_CANDIDATE_NUMPY_GEOMETRY_VERIFIED_TORCH_AST_ONLY','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'synthetic_reference_shape':[256,6,100],'positive_dot_projection_maxabs':float(np.maximum(after,0).max()),'orthogonal_component_maxabs_error':orth_error,'zero_and_small_gradient_identity_verified':True,'zero_head_three_control_contexts_equal':True,'gradient_error_bound_max_residual':float(np.max(true_dot-bound)),'curvature_counterexample_risk_increase':nonlinear_increase,'gradient_head_symbolic_parameter_count':symbolic_parameters,'inference_signature_has_no_labels_or_teacher_gradient':True,'torch_forward_executed':False,'actual_gpu_verified':False,'torch_gradient_isolation_verified':False,'full_model_integrated':False,'new_training_started':False,'test_accessed':False,'limits':'Synthetic NumPy geometry, not TRAIN/dev results. AST and reference arithmetic do not validate PyTorch runtime, gradient isolation, batch independence, performance or semantic decomposition.'}
with (root/'outputs/向量反馈候选几何与静态接口核验_v2.json').open('x',encoding='utf-8') as file:json.dump(report,file,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'status':report['status'],'symbolic_parameters':symbolic_parameters,'curvature_counterexample':nonlinear_increase}))
