import datetime,json,hashlib,shutil
from pathlib import Path
F=Path('outputs/parallel_gpu_20261003_1800').resolve();D=Path('D:/CodexBackups/selective_flow_20261003_1105/completed_refresh_20261004T0037Z')
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
now=datetime.datetime.now(datetime.timezone.utc).isoformat();new={};proofs={}
for p in D.iterdir():
 r=load(p/'local_checkpoint_verification.json');targetfile=F/(p.name+'_destination_verification_20261004T0049Z.json');q=load(targetfile)
 assert r['selection']==q['selection'] and r['files_sha256']==q['files_sha256'] and r['bytes']==q['checkpoint_bytes']
 proofs[p.name]={'selection':q['selection'],'files_sha256':q['files_sha256'],'local_and_target_all_hashes_match':True,'local_checkpoint_file':r['local_file'],'target_directory':q['directory'],'destination_proof':str(targetfile),'target_verified_at':q['verified_at']}
 new[p.name]={'checkpoint_file':r['local_file'],'checkpoint_sha256':r['selection']['checkpoint_sha256'],'bytes':r['bytes'],'local_full_checkpoint_sha_verified':True,'audit_file':str(p/'local_checkpoint_verification.json'),'files_sha256':r['files_sha256'],'independent_remote_copy':{'node':'memory','directory':q['directory'],'status':'COMPLETE_TARGET_FULL_SHA_VERIFIED','proof_file':str(targetfile)},'source_retained':True}
with (F/'local_and_independent_copy_audit_20261004T0050Z.json').open('x',encoding='utf-8') as f:json.dump({'verified_at':now,'status':'ALL_THREE_LOCAL_AND_INDEPENDENT_COPY_FILES_SELECTION_BYTES_MATCH','jobs':proofs},f,indent=2)
with (F/'current_checkpoint_locations_20261004T0050Z.json').open('x',encoding='utf-8') as f:json.dump({'verified_at':now,'previous_registry':str(F/'current_checkpoint_locations_20261004T0019Z.json'),'completed_weights_local_full_sha':62,'new_completions':new,'overall_research_complete':False,'permanent_backup_complete':False,'source_and_all_copies_retained':True},f,indent=2)
with (F/'backup_gap_20261004T0050Z.json').open('x',encoding='utf-8') as f:json.dump({'created_at':now,'completed_weights_local_full_sha':62,'completed_checkpoint_gaps':[],'permanent_backup_complete':False,'pending':['Three P4 coupled unguarded/memory guarded pairing','Two parallel 4060 full-null/text-one pilots and their frozen diagnostic','Research decisions and justified successors','All lease final dynamic preservation']},f,indent=2)
with (F/'dual_protocol_utf8_failure_20261004T0050Z.json').open('x',encoding='utf-8') as f:json.dump({'error':'Local default GBK decoding failed reading prior UTF8 protocol; script changed to explicit UTF8 and passed. No training/source/protocol changes.','corrected':True},f)
folder=F/'supervision_20261004T0050Z';folder.mkdir(exist_ok=False);sources=[]
for name in ['local_and_independent_copy_audit_20261004T0050Z.json','current_checkpoint_locations_20261004T0050Z.json','backup_gap_20261004T0050Z.json','heartbeat_progress_20261004T0040Z.json','heartbeat_progress_20261004T0048Z.json','source_train_shuffle_frozen_view_local_audit_20261004T0040Z.json','new_completed_candidate_decision_20261004T0045Z.json','completion_auditor_entry_key_failure_20261004T0045Z.json','dual_source_candidate_plan_20261004T0044Z.json','dual_source_source_audit_20261004T0044Z.json','dual_source_protocol_audit_20261004T0049Z.json','dual_protocol_utf8_failure_20261004T0050Z.json','capture_live_supervision_snapshot_v8_20261004T0045Z.py','verify_live_supervision_snapshot_local_v13_20261004T0046Z.py','verify_mixed_snapshot_local_v4_20261004T0040Z.py','audit_source_train_shuffle_frozen_views_local_v1_20261004T0040Z.py','audit_dual_full_null_local_v1_20261004T0048Z.py','audit_dual_text_one_local_v1_20261004T0048Z.py','supervision_destination_verification_20261004T0020Z.json','supervision_source_target_audit_20261004T0020Z.json']:
 sources.append((F/name,name))
for p in D.iterdir():
 for x in p.rglob('*'):
  if x.is_file() and x.name!='best.pt':sources.append((x,p.name+'__'+str(x.relative_to(p)).replace('\\','__')))
 sources.append((F/(p.name+'_destination_verification_20261004T0049Z.json'),p.name+'_destination_verification_20261004T0049Z.json'))
for dirname in ['live_snapshots_20261004T0037Z','live_snapshots_20261004T0048Z','source_train_shuffle_diagnostic_20261004T0037Z','dual_source_deployment_20261004T0044Z']:
 for x in (F/dirname).rglob('*'):
  if x.is_file():sources.append((x,dirname+'__'+str(x.relative_to(F/dirname)).replace('\\','__')))
manifest={}
for x,name in sources:
 assert name not in manifest and x.exists(),name
 b=x.read_bytes();(folder/name).write_bytes(b);manifest[name]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'source_local':str(x)}
with (folder/'manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
print(json.dumps({'files':len(manifest),'bytes':sum(e['bytes'] for e in manifest.values()),'registry':62}))

