from pathlib import Path
import argparse,json,hashlib,zipfile,datetime
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
w=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest();reports={}
uuids=['GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa']
plan=json.loads((w/'group_teacher_plan_20261005T1650Z/group_teacher_plan_v3.json').read_text())
for fold,node in enumerate(['a','b','c']):
 d=a.directory/node;receipt=json.loads((d/'receipt.json').read_text());raw=(d/'snapshot.zip').read_bytes();assert len(raw)==receipt['bytes'] and sha(raw)==receipt['sha256']
 with zipfile.ZipFile(d/'snapshot.zip') as z,zipfile.ZipFile(a.previous/node/'snapshot.zip') as old:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  m=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
  for name,meta in m.items():
   b=z.read(name);assert len(b)==meta['bytes'] and sha(b)==meta['sha256']
  inv=json.loads(z.read('inventory.json'));assert inv['gpu'].split(',')[0]==uuids[fold]
  assert sha(z.read('source/capture_soft_vector_v18.py'))==inv['capture_source_sha256']==sha((w/'capture_soft_vector_v18.py').read_bytes())
  unchanged=0
  for name,meta in json.loads(old.read('member_manifest.json')).items():
   if name.endswith('inventory.json') or name=='large_file_manifest.json' or name.startswith('source/capture_soft_vector_'):continue
   assert m.get(name)==meta,('Prior evidence changed',node,name);unchanged+=1
  large=json.loads(z.read('large_file_manifest.json'))
  for name,meta in json.loads(old.read('large_file_manifest.json')).items():assert large[name]['sha256']==meta['sha256'] and large[name]['bytes']==meta['bytes']
  prefix='group_teacher/';r=json.loads(z.read(prefix+'precheck_v2/receipt.json'))
  assert r['fold']==fold and r['status'].endswith('FOUR_REAL_UPDATES_PASSED')
  assert r['plan_sha256']==sha(z.read(prefix+'group_teacher_plan_v3.json'))
  for name,h in plan['source_sha256'].items():assert sha(z.read(prefix+name))==h
  for name,meta in plan['folds'][fold]['files'].items():assert m[prefix+name]==meta
  assert json.loads(z.read(prefix+'precheck_v2_exit.json'))['exit_code']==0
  gi=json.loads(z.read('group_teacher_inventory.json'));assert gi['precheck_receipt_present'] and not gi['matching_processes'] and not gi['completion_present']
  cp=large[prefix+'precheck_v2/initial_full_checkpoint.pt'];assert cp['sha256']==r['initial_checkpoint_sha256'] and cp['bytes']==r['initial_checkpoint_bytes']
  reports[node]=dict(capture_utc=receipt['utc'],snapshot_sha256=receipt['sha256'],prior_small_members_unchanged=unchanged,initial_weight_fresh_sha_reference=cp,precheck_passed=True,formal_100_complete=False)
result=dict(status='THREE_ATOMIC_TEACHER_GPU_PRECHECK_SNAPSHOTS_AND_PRIOR_COMPLETED_EVIDENCE_AUDITED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),nodes=reports,initial_fullweights_scope='Fresh full SHA references and GPU strict reload evidence; this snapshot does not download full initial weights.',whole_pipeline_crossfit=False)
assert not a.out.exists();a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(result['status'])
