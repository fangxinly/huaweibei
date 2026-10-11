from pathlib import Path
import zipfile,json,hashlib,datetime,shutil
w=Path(__file__).resolve().parent;r=w/'group_teacher_plan_20261005T1650Z';d=Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_precheck_20261006T0210Z');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={}
for f in sorted(r.iterdir()):
 if f.suffix=='.npy' or f.name=='group_teacher_plan_v3.json':files['plan/'+f.name]=f
for n in ['group_teacher_runtime_v2.py','check_group_teacher_v2.py','audit_group_teacher_mechanism_receipts_v2.py','verify_group_teacher_precheck_cpu_v2.py']:
 files[n]=w/n
for node in ['a','b','c']:
 pre=json.loads((d/node/'precheck_v2/receipt.json').read_text());assert pre['all_retained_parameter_gradients_finite'] and pre['original_v1_initial_prediction_error']==0
 for n in ['precheck_v2/receipt.json','precheck_v2/normalization_witness.npz','precheck_v2/initial_replay_predictions.npz','precheck_v2_exit.json','precheck_v2_execution.log','all_gradient_receipt.json','all_gradient_exit.json','all_gradient_execution.log']:
  files['original_receipts/'+node+'/'+n]=d/node/n
manifest={n:{'bytes':p.stat().st_size,'sha256':sha(p)} for n,p in files.items()}
bundle=w/'group_teacher_precheck_cpu_v2.zip';assert not bundle.exists()
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
 for n,f in files.items():z.write(f,n)
 z.writestr('preservation_manifest.json',json.dumps(manifest,indent=2))
proof=dict(status='TEACHER_V2_PRECHECK_ORIGINAL_ARRAYS_CPU_TRANSPORT_BUNDLE_FROZEN',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=sha(bundle),bytes=bundle.stat().st_size,files=manifest,full_initial_weights_included=False)
target=w.parent/'outputs/视频隔离教师v2预检CPU传输包冻结.json';target.write_text(json.dumps(proof,indent=2),encoding='utf-8')
for f in [bundle,target,w/'capture_soft_vector_v18.py',w/'audit_soft_snapshot_v18.py',w/'group_teacher_runtime_v2.py',w/'check_group_teacher_v2.py',r/'group_teacher_plan_v3.json',w/'audit_group_teacher_mechanism_receipts_v2.py',w/'verify_group_teacher_precheck_cpu_v2.py',w/'train_group_teacher_v1.py']:
 t=d/f.name;assert not t.exists();shutil.copy2(f,t);assert sha(t)==sha(f)
print(json.dumps({k:proof[k] for k in ['status','sha256','bytes']}))
