"""Freeze original fixed candidate continuation only after preserved native/peer400 proofs."""
import argparse,ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
def read(f):return json.loads(f.read_text(encoding='utf8'))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def verify_original(stem):
 c=read(ev/(stem+'_capture.json'));f=ev/(stem+'_original.zip');assert c['natural_exit']==0 and sha(f)==c['archive_SHA'] and f.stat().st_size==c['archive_bytes']
 with zipfile.ZipFile(f) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for n,h in json.loads(z.read('member_SHA.json')).items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
 return c
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args();source=base/'work/candidate_recovery400_preparation_20261009T000900Z'
clock=datetime.datetime.strptime(a.stamp,'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc)
for node in ('A','B'):
 c=verify_original(node+'_recovery400_native');r=read(ev/(node+'_recovery400_native_result.json'))
 assert r['candidate_SHA']==sha(source/'resume_official_mse100_v1.py') and r['source_SHA']==sha(source/'qualify_resume_official_v1.py') and r['final_steps']==440 and r['qualified_modes']==['regression_aux','factorized_aux'] and r['prediction_parameter_Adam_scheduler_python_numpy_torch_RNG_identical']
 assert datetime.datetime.fromisoformat(c['actual_UTC'])<=clock
for node,peer,mode in [('A','B','factorized_aux'),('B','A','regression_aux')]:
 cap=read(ev/(node+'_interrupted_capture.json'));pub=read(ev/(node+'_interrupted_publication.json'));client=read(ev/(node+'_interrupted_publication_client_exit.json'));audit=read(ev/(peer+'_interrupted_peer400_seekfix_result.json'));peer_cap=verify_original(peer+'_interrupted_peer400_seekfix')
 assert cap['capture_process_natural_exit']==client['natural_exit']==0 and pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED'
 assert cap['prior_training_natural_exit_unknown'] and not cap['prior_complete100'] and (audit['public_Release_download'] or audit.get('retained_public_Release_origin_archive_reuse')) and audit['all_member_SHA_CRC_unique']
 assert audit['status']=='INTERRUPTED_ORIGINAL_PEER_COMPLETE_400_MODEL_ADAM_SCHED_RNG_ORDER_SELECTION_CPU_PASSED' and audit['mode']==mode and audit['peer']==node and audit['all_Adam_steps']==400 and audit['Adam_parameter_states']==345
 assert audit['archive_SHA']==cap['archive_SHA'] and audit['warm400_checkpoint_SHA']==cap['warm400_checkpoint_SHA']
 assert all(datetime.datetime.fromisoformat(r['actual_UTC'])<=clock for r in (cap,pub,audit,peer_cap))
 oldroot=base/'work'/('polarity_intensity_'+mode+'_resume100_qualified_20261008T205936Z');q=read(oldroot/'qualified_resume_plan.json')
 cost=read(ev/(node+'_interrupted_compute_cost.json'));assert cost['original_completed_extra_replayed_updates']==cost['original_last_recorded_step']-400 and cost['original_last_recorded_step']>=cap['last_epoch']*40
 root=base/'work'/(node+'_candidate_recovery400_qualified_'+a.tag);root.mkdir()
 for f in source.iterdir():
  if f.suffix in ('.py','.npy'):shutil.copy2(f,root/f.name)
 for n in q['qualification_sha256']:shutil.copy2(oldroot/n,root/n)
 for host in ('A','B'):
  for suffix in ('recovery400_native_result.json','recovery400_native_exit.json','recovery400_native_capture.json'):shutil.copy2(ev/(host+'_'+suffix),root/(host+'_'+suffix))
 for suffix in ('interrupted_capture.json','interrupted_publication.json','interrupted_publication_client_exit.json'):shutil.copy2(ev/(node+'_'+suffix),root/('actual_'+suffix))
 shutil.copy2(ev/(node+'_interrupted_compute_cost.json'),root/'actual_interrupted_compute_cost.json')
 (root/'recovery400_othernode_CPU_result.json').write_text(json.dumps(audit,indent=2),encoding='utf8');shutil.copy2(ev/(peer+'_interrupted_peer400_seekfix_capture.json'),root/'recovery400_othernode_CPU_capture.json')
 wrapper=root/'run_resumed100_capture_v1.py';s=wrapper.read_text(encoding='utf8');marker=" uuid=subprocess.check_output(['nvidia-smi'"
 pos=s.index(marker)
 extra=" assert p['recovery400_CPU_qualified'] and p['recovery_native_qualified']\n peer=json.loads((bundle/'recovery400_othernode_CPU_result.json').read_text());cap=json.loads((bundle/'recovery400_othernode_CPU_capture.json').read_text())\n assert peer['status']=='INTERRUPTED_ORIGINAL_PEER_COMPLETE_400_MODEL_ADAM_SCHED_RNG_ORDER_SELECTION_CPU_PASSED' and cap['natural_exit']==0 and peer['warm400_checkpoint_SHA']==p['resume_checkpoint_SHA'] and peer['all_Adam_steps']==400 and peer['mode']==p['candidate_mode']\n for node in ('A','B'):\n  native=json.loads((bundle/(node+'_recovery400_native_result.json')).read_text());nexit=json.loads((bundle/(node+'_recovery400_native_exit.json')).read_text())\n  assert nexit['natural_exit']==0 and native['candidate_SHA']==p['source_sha256']['resume_official_mse100_v1.py'] and native['final_steps']==440 and native['prediction_parameter_Adam_scheduler_python_numpy_torch_RNG_identical']\n"
 s=s[:pos]+extra+s[pos:];wrapper.write_text(s,encoding='utf8');ast.parse(s)
 q.update(status='ACTUAL_FIXED_CANDIDATE_RECOVERY400_QUALIFIED',actualclock_qualification_UTC=a.stamp,recovery400_CPU_qualified=True,recovery_native_qualified=True,resume_checkpoint_SHA=cap['warm400_checkpoint_SHA'],interrupted_prior_root='/data/coding/'+node+'_polarity_intensity_'+mode+'_resume100_actual_20261008T205936Z',interrupted_original_archive_SHA=cap['archive_SHA'],replayed_interrupted_recorded_updates=cost['original_completed_extra_replayed_updates'],original_last_recorded_step=cost['original_last_recorded_step'],unlogged_last_update_not_excluded=True,extra_replay_compute_must_be_reported=True,not_new_independent_replicate=True,detached_process_required=True,mutable_checkpoint_interval_updates=400,old_frozen_files_untouched=True,source_sha256={f.name:sha(f) for f in root.iterdir() if f.suffix in ('.py','.npy')},qualification_sha256={f.name:sha(f) for f in root.glob('*.json')})
 (root/'qualified_resume_plan.json').write_text(json.dumps(q,indent=2),encoding='utf8')
 with zipfile.ZipFile(root/'qualified_resume_source.zip','x',zipfile.ZIP_DEFLATED) as z:
  for f in root.iterdir():
   if f.suffix!='.zip':z.write(f,f.name)
 print(json.dumps(dict(node=node,mode=mode,root=str(root),plan_SHA=sha(root/'qualified_resume_plan.json'),source_SHA=sha(root/'qualified_resume_source.zip'),real_recovery_started=False)))
