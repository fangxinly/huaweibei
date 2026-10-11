import datetime,json
from pathlib import Path
F=Path('outputs/parallel_gpu_20261003_1800');h=json.loads((F/'heartbeat_progress_20261004T0048Z.json').read_text(encoding='utf-8'));ref=json.loads(Path('D:/CodexBackups/selective_flow_20261003_1105/completed_refresh_20261003T2313Z/source_supervised_seed128_4060Ti/attempt01/protocol.json').read_text(encoding='utf-8'))
jobs=h['nodes']['4060ti']['jobs'];rows={}
for case,j in jobs.items():
 p=j['protocol'];allow={'name','source_sha256','parallel_authorization','task_loss','training_source_permutation','limits'}
 assert set(p)|allow==set(ref)|allow
 for k in set(p)&set(ref)-allow:assert p[k]==ref[k],(case,k)
 assert p['parallel_authorization']['gpu_uuid']==ref['parallel_authorization']['gpu_uuid'] and p['parallel_authorization']['frozen_inputs_verified']
 assert p['parallel_authorization']['shared_gpu_pair_authorized'] and p['parallel_authorization']['maximum_training_processes']==2
 assert p['limits']==(ref['limits'] if case=='full_null' else json.loads(Path('D:/CodexBackups/selective_flow_20261003_1105/completed_refresh_20261003T2338Z/source_full_control_seed128_4060Ti/attempt01/protocol.json').read_text(encoding='utf-8'))['limits'])
 if case=='full_null':
  null=json.loads(Path('D:/CodexBackups/selective_flow_20261003_1105/completed_refresh_20261004T0037Z/source_train_shuffle_seed128_4060Ti/attempt01/protocol.json').read_text(encoding='utf-8'))
  assert p['training_source_permutation']==null['training_source_permutation']
 else:assert 'training_source_permutation' not in p
 rows[case]={'status':'RUNTIME_PROTOCOL_ALL_FROZEN_KEYS_AND_GPU_IDENTITY_VERIFIED','name':p['name'],'task_loss':p['task_loss'],'history_epochs':j['epoch'],'actual_child_pid':j['child_pid'],'shared_gpu_authorized':True}
with (F/'dual_source_protocol_audit_20261004T0049Z.json').open('x') as f:json.dump({'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':rows,'limits':'Two concurrent jobs on same physical GPU; does not establish identical trajectories or hardware runtime to earlier serial comparisons. No test/five-seed promotion.'},f,indent=2)
print(json.dumps(rows))



