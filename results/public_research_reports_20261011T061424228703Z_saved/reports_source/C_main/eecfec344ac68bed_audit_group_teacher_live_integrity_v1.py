from pathlib import Path
import json,zipfile,hashlib,datetime,shutil
w=Path(__file__).resolve().parent;o=w.parent/'outputs'
d=Path('D:/CodexBackups/selective_flow_20261003_1105')
new=d/'group_teacher_live_snapshots_20261006T022505Z'
old=d/'group_teacher_precheck_snapshots_20261006T022030Z'
sha=lambda b:hashlib.sha256(b).hexdigest();rows={}
live=json.loads((o/'视频隔离教师正式三折运行动态核验.json').read_text(encoding='utf-8'))
for n in ['a','b','c']:
 with zipfile.ZipFile(new/n/'snapshot.zip') as z,zipfile.ZipFile(old/n/'snapshot.zip') as prev:
  m=json.loads(z.read('member_manifest.json'));prior=json.loads(prev.read('member_manifest.json'));count=0
  for name,meta in prior.items():
   if name.endswith('inventory.json') or name=='large_file_manifest.json' or name.startswith('source/capture_soft_vector_'):continue
   assert m.get(name)==meta,('Prior completed or precheck evidence changed',n,name);count+=1
  large=json.loads(z.read('large_file_manifest.json'))
  for name,meta in json.loads(prev.read('large_file_manifest.json')).items():
   assert large[name]['sha256']==meta['sha256'] and large[name]['bytes']==meta['bytes'],(n,name)
  assert sha(z.read('source/capture_soft_vector_v19.py'))==sha((w/'capture_soft_vector_v19.py').read_bytes())
  rows[n]={'prior_immutable_small_members_unchanged':count,'capture_utc':live['nodes'][n]['capture_utc'],'epochs_at_capture':live['nodes'][n]['epochs'],'state':live['nodes'][n]['state']}
result={'status':'TEACHER_FORMAL_LIVE_AND_PRIOR_COMPLETED_AND_PRECHECK_EVIDENCE_UNCHANGED_AUDITED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'nodes':rows,'new_fullweights_downloaded':False,'new_completed_teacher_CPU_save':False}
out=o/'视频隔离教师正式启动联合动态保存核验.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for f in [out,o/'视频隔离教师正式三折运行动态核验.json',o/'视频隔离教师训练启动后既有根独立核验.json',w/'audit_group_teacher_live_integrity_v1.py',w/'audit_group_teacher_formal_snapshot_v1.py',w/'audit_soft_snapshot_v19.py',w/'capture_soft_vector_v19.py']:
 dest=new/f.name;assert not dest.exists();shutil.copy2(f,dest);assert sha(dest.read_bytes())==sha(f.read_bytes())
print(result['status'])
