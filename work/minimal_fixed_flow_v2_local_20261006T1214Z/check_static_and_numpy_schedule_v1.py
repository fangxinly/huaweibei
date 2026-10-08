"""No Torch import, model execution, data labels, or GPU claims."""
import ast, datetime, hashlib, json
from pathlib import Path
import numpy as np

root=Path(__file__).parent
source=(root/'minimal_fixed_flow_v2.py').read_text(encoding='utf-8');tree=ast.parse(source)
compile(tree,str(root/'minimal_fixed_flow_v2.py'),'exec')
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MinimalFixedFlow')
methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
init=methods['__init__']; attrs={n.targets[0].attr for n in ast.walk(init) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Attribute) and isinstance(n.targets[0].value,ast.Name) and n.targets[0].value.id=='self'}
assert {'forward_fields','backward_fields','reader','role_head','unimodal_heads','donor_feedback','variant'}<=attrs
assert not attrs&{'feedback','utility_heads','pair_heads','utility_calibration','decoder_modules'}
calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in {'load','load_state_dict','cuda','step'} for n in calls)
assert not any(isinstance(n.func,ast.Name) and n.func.id in {'deterministic_modules','terminal'} for n in calls)
assert sum(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='LegacyFlow' and n.func.attr=='forward' for n in calls)==1
donor_detach=next(n for n in ast.walk(methods['update_context']) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='features')
assert isinstance(donor_detach.value,ast.Call) and donor_detach.value.func.attr=='detach'

rng=np.random.default_rng(91819);old=rng.normal(size=(7,3,5));feedback=rng.normal(size=(7,6,5))
original=.5*old+.5*(.5*(np.full((7,6,1),.5)*feedback).reshape(7,3,2,5).sum(2))
reduced=.5*old+.125*feedback.reshape(7,3,2,5).sum(2)
error=float(np.max(abs(original-reduced)));assert error<1e-15
assert np.allclose(.5*0+.125*np.ones((1,3,2,1)).sum(2),.25)
mask=np.array([[1,1,0],[1,0,0]],dtype=bool);state=rng.normal(size=(2,3,3,5))*mask[:,None,:,None]
context=np.zeros((2,3,5));seen=[]
for step in range(2):
    seen.append(context.copy());velocity=np.tanh(state+context[:,:,None,:]+step*.5)
    state=(state+.5*velocity)*mask[:,None,:,None]
    if step==0:
        donated=np.ones((2,6,5));context=.5*context+.125*donated.reshape(2,3,2,5).sum(2)
assert np.max(abs(seen[0]))==0 and np.all(seen[1]==.25) and np.max(abs(state[~np.broadcast_to(mask[:,None,:,None],state.shape)]))==0
assert len(seen)==2
for name in ['legacy_flow_model.py','finite_single_token_reader_v1.py']:
    text=(root/name).read_text(encoding='utf-8');compile(ast.parse(text),str(root/name),'exec')
legacy=ast.parse((root/'legacy_flow_model.py').read_text(encoding='utf-8'))
base=next(n for n in legacy.body if isinstance(n,ast.ClassDef) and n.name=='WholeStateFlow')
forward=next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
source_calls=ast.unparse(forward)
assert 'reconstructed = terminal' in source_calls and 'final_context[:, m]' in source_calls and 'source.detach()' in source_calls
assert 'final_context.detach()' not in source_calls and 'terminal.detach()' not in source_calls
proof=dict(status='LOCAL_CANDIDATE_AST_AND_SYNTHETIC_NUMPY_SCHEDULE_ONLY_PASSED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_actual_Torch_model_execution=True,no_real_labels_read=True,no_GPU_precheck=True,no_parameter_gradient_or_HVP_claim=True,context_algebra_max_error=error,masked_synthetic_singleton_schedule=True,declared_retained_modules=sorted(attrs),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')},euler_steps=2,passes=1,mandatory_real_remaining=['full_model_public_pretrain_and_task_initialization_audit','FIT_only_statistics_and_label_guard','all_retained_gradient_optimizer_coverage_and_singleton_second_order_if_needed','whole_process_GPU_memory_and_epoch_timing','strict_full_INNER_original_input_disk_reload','fresh_space_and_two_hour_save_reserve','full_new_weight_D_and_other_node_CPU_preservation'])
(root/'local_static_numpy_receipt.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':proof['status'],'context_algebra_error':error,'real_GPU':False}))
