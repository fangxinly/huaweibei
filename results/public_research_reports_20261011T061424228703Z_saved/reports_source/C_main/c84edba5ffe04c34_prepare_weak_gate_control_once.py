import ast,hashlib,json,shutil,sys,zipfile
from pathlib import Path
clock=sys.argv[1];base=Path('D:/CodexBackups/selective_flow_20261003_1105');parent=base/'selected20_requested_VAL_TEST_20261007T111903Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent/'A_complete_capture.zip')==json.loads((parent/'A_capture_receipt.json').read_text(encoding='utf8'))['sha256']
with zipfile.ZipFile(parent/'A_complete_capture.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
stamp=clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z';d=base/('weak_prediction_gate_control_'+stamp);d.mkdir();bundle=d/'source';bundle.mkdir()
source=Path('work/weak_prediction_gate_control_once.py');metric=Path('work/sentiment_metrics_careflow_v1.py')
for p in [source,metric]:ast.parse(p.read_text(encoding='utf8'));shutil.copy2(p,bundle/p.name)
manifest=json.loads((parent/'A_original_capsule/manifest.json').read_text(encoding='utf8'))['member_sha256'];inputs={}
for role in ('fit','inner'):
 for suffix in ('_bfp.npz','_evaluation_targets.npy'):
  n=role+suffix;p=parent/'A_original_capsule/original'/n;assert sha(p)==manifest['original/'+n];shutil.copy2(p,bundle/n);inputs[n]=sha(p)
plan=dict(status='FIXED_PREDICTION_ONLY_GATE_DEVELOPMENT_PROTOCOL_FROZEN',actualclock_freeze_UTC=clock,human_request='接下来继续优化；为什么不直接针对弱情绪误差问题',formula='q=b+(b^2/(1+b^2))*(p-b)',scale_squared=1,no_fit=True,no_hyperparameter_search=True,no_new_GPU_forward=True,no_true_label_at_inference=True,source_sha256={p.name:sha(p) for p in [bundle/source.name,bundle/metric.name]},input_sha256=inputs,history_FIT_INNER_explored=True,official_VAL_TEST_access_forbidden=True,outer_evaluation_forbidden=True,parent_checkpoint_SHA='5a58f50c554890baa4324d49437155649a92c68032dcbc6d88138a6269e6ed31',selected_model_SHA='35fa4c4eae77b1f505f5c717669d533e05a6810fde5cbdb39fffd36b8450209e',parent_receipt_SHA='534929dc0cbb7cf82c9b9c9b6255f526f0efd0a15bcd2f2f61e27af78540dc3d',decision='Single fixed development mechanism test; report all5 and weak/strong. No INNER selection between alternatives, no deployment or training benefit claim. Stop this fixed gate if tradeoff violates five-metric goal.',affine_OLS_proposal_unexecuted=True)
(bundle/'protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
with zipfile.ZipFile(d/'frozen_source_inputs.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in bundle.iterdir():z.write(p,p.name)
record=dict(D=str(d),bundle=str(bundle),plan_SHA=sha(bundle/'protocol.json'),source_zip_SHA=sha(d/'frozen_source_inputs.zip'),actualclock=clock)
(d/'freeze_receipt.json').write_text(json.dumps(record,indent=2),encoding='utf8');Path('work/weak_gate_current.json').write_text(json.dumps(record,indent=2),encoding='utf8');print(json.dumps(record))
