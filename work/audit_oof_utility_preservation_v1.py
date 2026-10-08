from pathlib import Path
import argparse,datetime,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument('--snapshots',type=Path,required=True);p.add_argument('--completed',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
cpu=json.loads((a.completed/'original_B_CPU_preservation_receipt.json').read_text())
ca=json.loads((a.completed/'original_B_CPU_array_audit.json').read_text())
assert cpu['origin']=='c' and cpu['target']=='b' and cpu['cpu_array_audit_exit_code']==0 and cpu['member_hash_all_passed'] and cpu['zip_crc_passed']
assert cpu['cpu_array_audit_sha256']==sha((a.completed/'original_B_CPU_array_audit.json').read_bytes())
assert ca['independent_cpu']['gpu_inventory'].split(',')[0]=='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'
assert not ca['independent_cpu']['torch_imported'] and not ca['independent_cpu']['cuda_forward_executed']
for name,meta in ca['files'].items():
 f=a.completed/'original_C'/name;assert f.stat().st_size==meta['bytes'] and sha(f.read_bytes())==meta['sha256']
report={}
previous={
 'b':Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_OOF_CPU_B_snapshot_20261006T031844Z_OOF_CPU_B'),
 'c':Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_completed_snapshots_20261006T030758Z_teacher100_cpu_completed/c')}
uuids={'b':'GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
prefixes={'c':'group_teacher/oof_utility_diagnostic_v1_20261006T033054Z/','b':'group_teacher/cpu_preservation/oof_utility_C_20261006T033054Z/files/'}
for node in ['b','c']:
 d=a.snapshots/node;r=json.loads((d/'receipt.json').read_text());raw=(d/'snapshot.zip').read_bytes()
 assert sha(raw)==r['sha256'] and len(raw)==r['bytes'] and (d/'exit_code.txt').read_text().strip()=='0'
 with zipfile.ZipFile(d/'snapshot.zip') as z,zipfile.ZipFile(previous[node]/'snapshot.zip') as old:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  members=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(members)|{'member_manifest.json'}
  for name,meta in members.items():
   b=z.read(name);assert len(b)==meta['bytes'] and sha(b)==meta['sha256']
  inv=json.loads(z.read('inventory.json'));assert inv['gpu'].split(',')[0]==uuids[node] and inv['compute']==''
  assert sha(z.read('source/capture_soft_vector_v19.py'))=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
  unchanged=0
  for name,meta in json.loads(old.read('member_manifest.json')).items():
   if name.endswith('inventory.json') or name=='large_file_manifest.json':continue
   assert members[name]==meta,('Completed old evidence changed',node,name);unchanged+=1
  large=json.loads(z.read('large_file_manifest.json'))
  for name,meta in json.loads(old.read('large_file_manifest.json')).items():
   assert large[name]['sha256']==meta['sha256'] and large[name]['bytes']==meta['bytes']
  for name,meta in ca['files'].items():assert members[prefixes[node]+name]==meta,(node,name)
  if node=='b':
   prefix='group_teacher/cpu_preservation/oof_utility_C_20261006T033054Z/'
   for remote,local in [('original_cpu_array_audit.json','original_B_CPU_array_audit.json'),('original_cpu_preservation_receipt.json','original_B_CPU_preservation_receipt.json'),('audit.log','original_B_CPU_audit.log')]:
    assert z.read(prefix+remote)==(a.completed/local).read_bytes()
   assert sha(z.read('group_teacher/audit_oof_utility_v2.py'))==cpu['audit_source_sha256']
  report[node]=dict(actual_capture_utc=r['utc'],sha256=r['sha256'],members=len(members),new_original_files=len(ca['files']),old_unchanged_members=unchanged,whole_zip_crc_and_member_hash=True,fullweight_SHA_reference_not_new_download=True)
out=dict(status='NEW_OOF_UTILITY_C_COMPLETE_D_ORIGINALS_B_CPU_ORIGINAL_RECEIPTS_AND_B_C_ATOMIC_SNAPSHOTS_JOINED_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),nodes=report,CPU_original_receipt_sha256=sha((a.completed/'original_B_CPU_preservation_receipt.json').read_bytes()),preserved_original_predictions_sha256=sha((a.completed/'original_C/execute/predictions_frozen.npz').read_bytes()),scope='New short diagnostic and CPU array copies only; old complete training/source/weight evidence unchanged; no fullweight retransmission.')
assert not a.out.exists();a.out.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
