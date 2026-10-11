from pathlib import Path
import ast,json,hashlib
F=Path('outputs/parallel_gpu_20261003_1800');dep=F/'dual_source_deployment_20261004T0044Z'
s=(F/'source_train_shuffle_deployment_20261004T0016Z/frozen_source_train_shuffle_view_audit_v1_20261004T0018Z.py').read_text(encoding='utf-8')
for case in ['full_null','text_one']:
 t=s.replace("V=B/'source_train_shuffle_deployment_20261004T0016Z'","V=B/'dual_source_deployment_20261004T0044Z/"+case+"'")
 t=t.replace('Aligned sequence lengths and encoders trained under seven route losses with a train-only same-valid-mask AV relationship permutation control retain shared alignment context.','Encoded masks and learned encoders retain alignment cues. This new checkpoint diagnostic is fixed train-only and makes no source causal claim.')
 ast.parse(t);(dep/case/'frozen_views.py').write_text(t,encoding='utf-8',newline='\n')
for case in ['full_null','text_one']:
 p=dep/case;flow=(p/'source_view_flow.py').read_text();r=(p/'run_source_validation.py').read_text()
 ref=F/('source_train_shuffle_deployment_20261004T0016Z' if case=='full_null' else 'source_view_deployment_20261003T2246Z')/'source_view_flow.py'
 def fn(s,n):return ast.dump(next(x for x in ast.walk(ast.parse(s)) if isinstance(x,ast.FunctionDef) and x.name==n),include_attributes=False)
 assert fn(flow,'__init__')==fn(ref.read_text(encoding='utf-8'),'__init__') and fn(flow,'routes')==fn(ref.read_text(encoding='utf-8'),'routes')
 assert ('final[:,6]-labels.reshape(-1)' if case=='full_null' else 'final[:,0]-labels.reshape(-1)') in flow
 assert hashlib.sha256((p/'native_flow.py').read_bytes()).hexdigest()=='51df1c42a41271bb7497cc0dab2919bfce908ab334c5b324cf46181540b0c7e7'
 ast.parse(r);ast.parse((dep/'launch_dual_source_v1_20261004T0044Z.py').read_text())
with (F/'dual_source_source_audit_20261004T0044Z.json').open('x') as f:json.dump({'status':'AST_IDENTICAL_FIELD_ROUTES_INITIALIZATION_NATIVE_AND_INTENDED_LOSS_VERIFIED','cases':['full_null','text_one'],'unchanged_capacity_and_route_compute':True,'concurrent_gpu_authorization_explicit':True,'runtime_gpu_check_pending':True,'candidate_promoted':False},f,indent=2)

