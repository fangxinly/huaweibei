from pathlib import Path
import base64,datetime,hashlib,json
root=Path.cwd();w=root/'work/new_p4_checks';out=root/'outputs'
nodes=[('a','REDACTED_SERVER_HOST.invalid',53449,'GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7'),('b','REDACTED_SERVER_HOST.invalid',53416,'GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'),('c','REDACTED_SERVER_HOST.invalid',53458,'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa')]
decode=lambda p:json.loads(base64.b64decode(p.read_text().strip()))
source_sha=hashlib.sha256((root/'work/task_gradient_vector_candidate_v3.py').read_bytes()).hexdigest()
checker_sha=hashlib.sha256((root/'work/check_soft_vector_cuda_v2.py').read_bytes()).hexdigest()
rows=[]
for node,host,port,uuid in nodes:
 r=decode(w/(node+'_cuda_report.b64'));inv=decode(w/(node+'_inventory.b64'))
 assert r['status']=='CUDA_SYNTHETIC_MODULE_FORWARD_BACKWARD_VERIFIED_NOT_FULL_MODEL'
 assert r['gpu_uuid']==uuid and inv['gpu'].split(',')[0].strip()==uuid
 assert r['source_sha256']==source_sha and r['checker_sha256']==checker_sha
 assert r['actual_parameters']==214506 and r['pure_message_cuda_gradcheck'] and r['zero_head_initial_controls_equal']
 assert inv['compute']=='' and inv['disk_data_free']>3_000_000_000
 assert not r['full_model_integrated'] and not r['formal_training_started'] and not r['train_dev_test_accessed']
 for mode,record in r['mode_checks'].items():
  assert record['task_message_gradient_max_error']<1e-6
  assert record['auxiliary_teacher_grad_none'] and record['task_gradient_heads_grad_none'] and record['feature_grad_detached']
 (w/(node+'_cuda_report.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
 (w/(node+'_inventory.json')).write_text(json.dumps(inv,indent=2),encoding='utf-8')
 rows.append({'node':node,'host':host,'port':port,'gpu_uuid':uuid,'inventory':inv,'module_check':r})
assert len({x['module_check']['initial_state_sha'] for x in rows})==1
result={'status':'THREE_DISTINCT_NEW_P4_CUDA_SYNTHETIC_MODULE_CHECKS_AUDITED','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'limits':'Synthetic .01 regularization and unit scales only, not selected task parameters or TRAIN fit. No full-model integration, replay, data collection or formal training. Host GPU PID differs from container PID; equality not claimed. Initial-empty/own allocation/single matching UUID/post-exit-empty observations retained.','first_v1_failure':'All three v1 runs reached final PID-equality guard and failed; not counted as completed checks. v1 source retained; v2 removes false equality requirement and records namespace and driver PIDs separately.','lease':{'human_duration_hours':24,'anchor_first_local_read_utc':'2026-10-05T12:08:17+00:00','estimated_expiry_utc':'2026-10-06T12:08:17+00:00','provider_expiry_verified':False},'C_old_final_recovered':False}
(out/'新三P4真实GPU与连续向量模块核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'initial_state_sha':rows[0]['module_check']['initial_state_sha'],'actual_parameters':214506,'max_error':max(v['task_message_gradient_max_error'] for x in rows for v in x['module_check']['mode_checks'].values())}))
