"""Qualify fixed100 continuation from actual prefix/Release/peer/native evidence."""
import argparse,ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_original(ev,stem):
 c=read(ev/(stem+'_capture.json'));f=ev/(stem+'_original.zip');assert c['natural_exit']==0 and sha(f)==c['archive_SHA'] and f.stat().st_size==c['archive_bytes']
 with zipfile.ZipFile(f) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for n,h in json.loads(z.read('member_SHA.json')).items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 return c

def run(source,tag,stamp):
 base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z';clock=datetime.datetime.strptime(stamp,'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc)
 for node in ('A','B'):
  c=verify_original(ev,node+'_resume_native_v2');r=read(ev/(node+'_resume_native_v2_result.json'));e=read(ev/(node+'_resume_native_v2_exit.json'))
  assert e['natural_exit']==0 and r['candidate_SHA']==sha(source/'resume_official_mse100_v1.py') and r['source_SHA']==sha(source/'qualify_resume_official_v1.py') and r['qualified_modes']==['regression_aux','factorized_aux'] and r['prefix_counted_once'] and r['prediction_parameter_Adam_scheduler_python_numpy_torch_RNG_identical']
  assert datetime.datetime.fromisoformat(c['actual_UTC'])<=clock
 for node,peer,mode in [('A','B','factorized_aux'),('B','A','regression_aux')]:
  c=verify_original(ev,peer+'_peer_audit');audit=read(ev/(peer+'_peer_audit_result.json'));cap=read(ev/(node+'_prefix16_capture.json'));tr=read(ev/(node+'_prefix16_result.json'));pub=read(ev/(node+'_prefix16_publication.json'));client=read(ev/(node+'_prefix16_publication_client_exit.json'))
  assert cap['natural_exit']==client['natural_exit']==0 and pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED' and audit['status']=='CANDIDATE_OTHER_NODE_PREFIX16_COMPLETE_MODEL_ADAM_RNG_ORDER_CPU_ON_OFF_PASSED'
  assert audit['archive_SHA']==cap['archive_SHA'] and audit['checkpoint_SHA']==tr['checkpoint_SHA'] and audit['mode']==mode and audit['peer']==node and audit['all_Adam_steps']==16
  assert all(datetime.datetime.fromisoformat(r['actual_UTC'])<=clock for r in (cap,pub,audit,c))
  root=base/'work'/('polarity_intensity_'+mode+'_resume100_qualified_'+tag);root.mkdir()
  for f in source.iterdir():
   if f.suffix in ('.py','.npy'):shutil.copy2(f,root/f.name)
  for host in ('A','B'):
   for suffix in ('native_result.json','native_exit.json'):shutil.copy2(ev/(host+'_'+suffix),root/(host+'_'+suffix))
   for suffix in ('resume_native_v2_result.json','resume_native_v2_exit.json'):shutil.copy2(ev/(host+'_'+suffix),root/(host+'_'+suffix))
  prior=base/'work/autonomous_mse16_healthfix_20261008T181431Z'
  for n in ('B_release_transport_result.json','public_release_metadata.json'):shutil.copy2(prior/n,root/n)
  (root/'prefix16_original_othernode_CPU_result.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
  for suffix in ('prefix16_capture.json','prefix16_exit.json','prefix16_publication.json','prefix16_publication_client_exit.json','prefix16_result.json'):
   shutil.copy2(ev/(node+'_'+suffix),root/('actual_'+suffix))
  p=read(source/'resume_native_preparation_plan.json');p.update(status='ACTUAL_POLARITY_INTENSITY_RESUME100_QUALIFIED',actualclock_qualification_UTC=stamp,execution_node=node,candidate_mode=mode,health_v2_native_qualified=True,prefix16_Release_othernode_CPU_qualified=True,resume_native_trajectory_qualified=True,prefix16_checkpoint_SHA=tr['checkpoint_SHA'],prefix16_original_archive_SHA=cap['archive_SHA'],prefix16_updates_in_total4000=True,training_budget_seconds=10800,wall_budget_basis='Observed 16-step medians ~1.12s with instrumentation; conservative 3h ceiling plus >=2h preservation fits human fourth-lease24h remaining; both arms same4000 updates, no extra training',source_sha256={f.name:sha(f) for f in root.iterdir() if f.suffix in ('.py','.npy')},qualification_sha256={f.name:sha(f) for f in root.glob('*.json')},continuation_runner_qualified=True)
  f=root/'qualified_resume_plan.json';f.write_text(json.dumps(p,indent=2),encoding='utf8')
  for n in root.glob('*.py'):ast.parse(n.read_text(encoding='utf8'))
  with zipfile.ZipFile(root/'qualified_resume_source.zip','x',zipfile.ZIP_DEFLATED) as z:
   for n in root.iterdir():
    if n.suffix!='.zip':z.write(n,n.name)
  print(json.dumps(dict(node=node,mode=mode,root=str(root),plan_SHA=sha(f),bundle_SHA=sha(root/'qualified_resume_source.zip'),real100_started=False)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args();run(a.source,a.tag,a.stamp)
