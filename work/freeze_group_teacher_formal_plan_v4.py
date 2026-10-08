from pathlib import Path
import hashlib,json,datetime,zipfile,shutil
w=Path(__file__).resolve().parent;p=w/'group_teacher_plan_20261005T1650Z';out=w/'group_teacher_formal_authorization_v4';assert not out.exists();out.mkdir();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();now=datetime.datetime.now(datetime.timezone.utc)
auditfile=w.parent/'outputs/视频隔离教师v2三GPU机制回执独立核验.json';audit=json.loads(auditfile.read_text(encoding='utf-8'))
assert audit['status']=='THREE_ORIGINAL_GPU_MECHANISM_RECEIPTS_INDEPENDENT_CPU_AUDIT_PASSED' and audit['plan_sha256']==sha(p/'group_teacher_plan_v3.json')
d=Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_precheck_20261006T0210Z')
cpu=json.loads((d/'cpu_preservation_receipt_v2.json').read_text());assert cpu['status']=='TEACHER_V2_ORIGINAL_THREE_GPU_PRECHECK_ARRAYS_AND_RETAINED_V1_FAILURES_CPU_SHA_AUDITED'
assert cpu['source_sha256']==sha(w/'verify_group_teacher_precheck_cpu_v2.py')
second=json.loads((d/'cpu_preservation_receipt_v2_c.json').read_text());assert second['status']==cpu['status'] and second['source_sha256']==cpu['source_sha256']
assert cpu['independent_from_original_gpu_host']['a'] and cpu['independent_from_original_gpu_host']['c'] and second['independent_from_original_gpu_host']['b']
free=shutil.disk_usage(d).free;budget=8*1024**3;assert free>budget
uuids=['GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'];inventories={}
for fold,node in enumerate(['a','b','c']):
 invfile=d/node/'fresh_formal_inventory.json';inv=json.loads(invfile.read_text());assert inv['gpu'].split(',')[0].strip()==uuids[fold] and not inv['compute'] and not inv['matching_teacher_processes']
 assert 0<=(now-datetime.datetime.fromisoformat(inv['utc'])).total_seconds()<600
 assert inv['runtime_sha256']==sha(w/'group_teacher_runtime_v2.py') and inv['plan_sha256']==sha(p/'group_teacher_plan_v3.json')
 r=json.loads((d/node/'precheck_v2/receipt.json').read_text());assert inv['data_disk_free']>4*r['initial_checkpoint_bytes']+1024**3
 assert json.loads((d/node/'precheck_v2_exit.json').read_text())['exit_code']==0
 inventories[node]={'original_inventory_sha256':sha(invfile),**inv}
space=dict(status='FRESH_PERMANENT_LOCAL_AND_THREE_REMOTE_SPACES_VERIFIED',utc=now.isoformat(),permanent_local_free_bytes=free,permanent_local_path=str(d),all_three_weights_and_preservation_budget_bytes=budget,deletion_required=False,nodes=inventories)
(out/'formal_space_authorization.json').write_text(json.dumps(space,indent=2),encoding='utf-8');shutil.copy2(auditfile,out/'three_fold_gpu_mechanism_audit.json')
formal=dict(status='FORMAL_TRAINING_PLAN_FROZEN_AFTER_THREE_GPU_MECHANISM_AUDITS',utc=now.isoformat(),version=4,parent_plan_sha256=sha(p/'group_teacher_plan_v3.json'),trainer_sha256=sha(w/'train_group_teacher_v1.py'),runtime_sha256=sha(w/'group_teacher_runtime_v2.py'),mechanism_audit_sha256=sha(out/'three_fold_gpu_mechanism_audit.json'),space_authorization_sha256=sha(out/'formal_space_authorization.json'),independent_original_CPU_receipt_sha256=sha(d/'cpu_preservation_receipt_v2.json'),estimated_lease_end_utc='2026-10-06T12:08:17+00:00',platform_deadline_verified=False,preservation_margin_seconds=7200,epochs=100,seed=91818,teacher_trainable_parameters=184749003,retained_parameter_tensors=291,teacher_full_parameter_finetuning=True,unused_pooler_removed=True,prior_failed_mechanism_retained=True,all_other_initial_tensors_and_forward_identical_to_v1=True,outer_labels_read_before_fit_or_selection=False,dev_requested=False,test_requested=False,selection='Earliest strict minimum sample-weighted INNER MSE of all100; one zero-label OUTER scalar inference only after selected full disk reload.',budget='Conservative twice measured pilot update/evaluation plus setup; estimate, not guarantee. Local8GiB reserve for3selectedweights/new materials and preservation. Initial weights fresh references only, not called downloaded.',whole_pipeline_crossfit=False)
lease=datetime.datetime.fromisoformat(formal['estimated_lease_end_utc']);assert (lease-now).total_seconds()>max(v['seconds_conservative_pilot_estimate'] for v in audit['folds'])+7200
(out/'group_teacher_formal_plan_v4.json').write_text(json.dumps(formal,indent=2),encoding='utf-8');shutil.copy2(w/'train_group_teacher_v1.py',out/'train_group_teacher_v1.py')
manifest={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in out.iterdir() if f.is_file()}
with zipfile.ZipFile(w/'group_teacher_formal_authorization_v4.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in manifest:z.write(out/n,n)
 z.writestr('formal_deployment_manifest.json',json.dumps(manifest,indent=2))
proof=dict(status='FORMAL_V4_SOURCE_AND_BUDGET_FROZEN_AFTER_ACTUAL_DEPENDENCIES_NOT_LAUNCHED_BY_THIS_SCRIPT',utc=now.isoformat(),formal=sha(out/'group_teacher_formal_plan_v4.json'),payload_sha256=sha(w/'group_teacher_formal_authorization_v4.zip'),space=space,measured_pilot=audit['folds'])
(w.parent/'outputs/视频隔离教师正式100冻结核验.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
backup=d/'formal_authorization_v4';assert not backup.exists();shutil.copytree(out,backup)
for n in manifest:assert sha(backup/n)==manifest[n]['sha256']
print(json.dumps({k:proof[k] for k in ['status','formal','payload_sha256']}))
