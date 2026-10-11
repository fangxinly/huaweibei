"""Compare actual completed evidence while allowing explicitly live teacher files."""
from pathlib import Path
import argparse,json,zipfile,hashlib,datetime
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--formal-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
w=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest();report=json.loads(a.formal_audit.read_text(encoding='utf-8'));nodes={}
assert report['status']=='FORMAL_THREE_FOLD_TEACHER_SOURCE_ORDERS_INITIALIZATION_AND_ACTUAL_SNAPSHOT_STATES_AUDITED'
for node in ['a','b','c']:
 receipt=json.loads((a.directory/node/'receipt.json').read_text());assert receipt['utc']==report['nodes'][node]['capture_utc'] and sha((a.directory/node/'snapshot.zip').read_bytes())==receipt['sha256']
 assert (a.directory/node/'exit_code.txt').read_text().strip()=='0'
 with zipfile.ZipFile(a.directory/node/'snapshot.zip') as z,zipfile.ZipFile(a.previous/node/'snapshot.zip') as old:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()));members=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(members)|{'member_manifest.json'}
  for name,meta in members.items():assert len(z.read(name))==meta['bytes'] and sha(z.read(name))==meta['sha256']
  count=0
  for name,meta in json.loads(old.read('member_manifest.json')).items():
   if name.endswith('inventory.json') or name=='large_file_manifest.json' or name.startswith('source/capture_soft_vector_'):continue
   if name.startswith('group_teacher/run_v1/') or name=='group_teacher/formal_training.log':continue
   assert members.get(name)==meta,('Prior immutable evidence changed',node,name);count+=1
  large=json.loads(z.read('large_file_manifest.json'))
  for name,meta in json.loads(old.read('large_file_manifest.json')).items():
   if name=='group_teacher/run_v1/selected_full_checkpoint.pt':continue
   assert large[name]['sha256']==meta['sha256'] and large[name]['bytes']==meta['bytes'],(node,name)
  assert sha(z.read('source/capture_soft_vector_v19.py'))==sha((w/'capture_soft_vector_v19.py').read_bytes())
  nodes[node]={'capture_utc':receipt['utc'],'snapshot_sha256':receipt['sha256'],'immutable_prior_small_members_checked':count,'teacher_state':report['nodes'][node]['state'],'teacher_epochs':report['nodes'][node]['epochs']}
result={'status':'CURRENT_TEACHER_STATES_AND_UNCHANGED_COMPLETED_RESEARCH_PRECHECK_CPU_EVIDENCE_AUDITED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'nodes':nodes,'whole_new_teacher_weight_local_or_CPU_save_proven':False,'scope':'Atomic snapshots contain original arrays/metadata/source plus stable-inode SHA references for large weights. Earlier completed-stage local/CPU full-weight proofs are separate; new teacher final full save must be obtained when actually complete.'}
assert not a.out.exists();a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(result['status'])
