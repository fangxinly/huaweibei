"""Audit complete atomic snapshots, frozen prior evidence, and new original CPU copy."""
from pathlib import Path
import datetime,hashlib,json,zipfile
work=Path(__file__).resolve().parent;outputs=work.parent/'outputs'
base=Path('D:/CodexBackups/selective_flow_20261003_1105')
shots=base/'train_oracle_completed_snapshots_20261006T020034Z'
previous=base/'finite_c2_completed_snapshots_20261005T1627Z'
done=base/'train_oracle_completed_20261006T0158Z/c'
sha=lambda b:hashlib.sha256(b).hexdigest();reports={}
uuids=dict(a='GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7',b='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067',c='GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa')
for node in ['a','b','c']:
 d=shots/node;receipt=json.loads((d/'receipt.json').read_text());raw=(d/'snapshot.zip').read_bytes()
 assert len(raw)==receipt['bytes'] and sha(raw)==receipt['sha256']
 with zipfile.ZipFile(d/'snapshot.zip') as z,zipfile.ZipFile(previous/node/'snapshot.zip') as old:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  m=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
  for name,info in m.items():
   data=z.read(name);assert len(data)==info['bytes'] and sha(data)==info['sha256'],name
  inv=json.loads(z.read('inventory.json'))
  assert inv['gpu'].split(',')[0]==uuids[node] and inv['compute']==''
  assert inv['capture_source_sha256']==sha(z.read('source/capture_soft_vector_v16.py'))==sha((work/'capture_soft_vector_v16.py').read_bytes())
  oldm=json.loads(old.read('member_manifest.json'));unchanged=0
  for name,info in oldm.items():
   if name.endswith('inventory.json') or name in ['large_file_manifest.json'] or name.startswith('source/capture_soft_vector_'):continue
   assert name in m and m[name]==info,('Old scientific evidence changed/missing',node,name)
   unchanged+=1
  prior_large=json.loads(old.read('large_file_manifest.json'));large=json.loads(z.read('large_file_manifest.json'))
  for name,meta in prior_large.items():
   assert large[name]['sha256']==meta['sha256'] and large[name]['bytes']==meta['bytes']
  role='no new diagnostic copy'
  if node=='c':
   prefix='train_oracle/'
   for filename in ['diagnose_train_oracle_v2.py','train_oracle_plan_v2.json','audit_train_oracle_v1.py']:
    assert z.read(prefix+filename)==(work/filename).read_bytes()
   assert z.read(prefix+'out/receipt.json')==(done/'out/receipt.json').read_bytes()
   assert z.read(prefix+'out/diagnostics.npz')==(done/'out/diagnostics.npz').read_bytes()
   assert z.read(prefix+'exit_code.txt').strip()==b'0'
   assert b'TRAIN_ORACLE_COMPLETE' in z.read(prefix+'execution.log')
   oi=json.loads(z.read('train_oracle_inventory.json'));assert oi['receipt_present'] and not oi['matching_processes']
   role='original C complete diagnostic and retired process'
  elif node=='b':
   prefix='train_oracle/cpu_preservation/c/'
   cpu=json.loads(z.read(prefix+'cpu_preservation_receipt.json'))
   assert cpu==json.loads((done/'cpu_preservation_receipt.json').read_text())
   for name,info in cpu['files'].items():assert m[prefix+name]==info
   assert sha(z.read(prefix+'independent_cpu_array_audit.json'))==cpu['array_audit_sha256']
   assert z.read(prefix+'exit_code.txt').strip()==b'0'
   assert b'ORACLE_CPU_PRESERVATION_COMPLETE' in z.read(prefix+'cpu_verify.log')
   role='independent CPU B original nine files and downloaded original receipt'
  reports[node]=dict(capture_utc=receipt['utc'],snapshot_sha256=receipt['sha256'],members=receipt['members'],
      frozen_prior_small_members_unchanged=unchanged,prior_large_weights_fresh_sha_refs_unchanged=len(prior_large),
      compute=inv['compute'],data_disk_free=inv['data_disk_free'],role=role)
out=dict(status='THREE_ATOMIC_ORACLE_COMPLETE_SNAPSHOTS_PRIOR_EVIDENCE_AND_ORIGINAL_CPU_COPY_AUDITED',
 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),nodes=reports,
 large_fullweight_scope='Fresh whole SHA references only; no duplicate completed fullweight download.',
 result_scope='C TRAIN fixed-weight diagnostic complete, not a new trained model or generalization claim.')
(outputs/'TRAIN_Oracle完成后三节点动态保存联合核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(reports,ensure_ascii=True))
