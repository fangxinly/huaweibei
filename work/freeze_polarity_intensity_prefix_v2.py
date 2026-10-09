"""Freeze matched complete-prefix executions only from two actual native receipts."""
import argparse,ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def run(tag,stamp):
 base=Path(__file__).resolve().parents[1];source=base/'work/polarity_intensity_prefix_v2_20261008T203225Z';ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z';p=read(source/'native_preparation_plan.json');clock=datetime.datetime.strptime(stamp,'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc)
 for node in ('A','B'):
  r=read(ev/(node+'_native_result.json'));e=read(ev/(node+'_native_exit.json'));c=read(ev/(node+'_native_capture.json'));f=ev/(node+'_native_original.zip')
  assert c['natural_exit']==e['natural_exit']==0 and r['UUID']==p['GPU_UUID'][node] and r['qualified_modes']==['regression_aux','factorized_aux']
  assert r['first16_same_parameter_trajectory'] and r['rounded_pure_decay_control_passed'] and r['helper_SHA']==sha(source/'training_health_components_v2.py') and r['source_SHA']==sha(source/'qualify_health_components_v2.py')
  assert datetime.datetime.fromisoformat(c['actual_UTC'])<=clock and sha(f)==c['archive_SHA'] and f.stat().st_size==c['archive_bytes']
  with zipfile.ZipFile(f) as z:
   assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
   for n,h in json.loads(z.read('member_SHA.json')).items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 assert shutil.disk_usage('C:/').free>=240_000_000
 for node,mode in [('A','factorized_aux'),('B','regression_aux')]:
  root=base/'work'/('polarity_intensity_'+mode+'_qualified_'+tag);root.mkdir()
  for f in source.iterdir():
   if f.suffix in ('.py','.npy'):shutil.copy2(f,root/f.name)
  for host in ('A','B'):
   for suffix in ('native_result.json','native_exit.json','native_capture.json'):shutil.copy2(ev/(host+'_'+suffix),root/(host+'_'+suffix))
  prior=base/'work/autonomous_mse16_healthfix_20261008T181431Z'
  for n in ('B_release_transport_result.json','public_release_metadata.json'):shutil.copy2(prior/n,root/n)
  plan=dict(p);plan.update(status='ACTUAL_POLARITY_INTENSITY_PREFIX16_QUALIFIED',actualclock_qualification_UTC=stamp,health_v2_native_qualified=True,execution_node=node,candidate_mode=mode,source_sha256={f.name:sha(f) for f in root.iterdir() if f.suffix in ('.py','.npy')},qualification_sha256={f.name:sha(f) for f in root.glob('*.json')},continuation_runner_qualified=False,real_TRAIN_started=False)
  target=root/'qualified_precheck_plan.json';target.write_text(json.dumps(plan,indent=2),encoding='utf8')
  for f in root.glob('*.py'):ast.parse(f.read_text(encoding='utf8'))
  with zipfile.ZipFile(root/'qualified_source.zip','x',zipfile.ZIP_DEFLATED) as z:
   for f in root.iterdir():
    if f.suffix!='.zip':z.write(f,f.name)
  print(json.dumps(dict(node=node,mode=mode,root=str(root),plan_SHA=sha(target),bundle_SHA=sha(root/'qualified_source.zip'),real_TRAIN_started=False)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args();run(a.tag,a.stamp)
