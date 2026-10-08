"""Audit actual late-lease ZIPs and retain missing historical windows honestly."""
from pathlib import Path
import argparse, datetime, hashlib, json, zipfile

p=argparse.ArgumentParser()
p.add_argument('--directory',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda f:json.loads(Path(f).read_text(encoding='utf-8'))
backup=Path('D:/CodexBackups/selective_flow_20261003_1105')
prior={
 'a':backup/'group_teacher_completed_snapshots_20261006T030758Z_teacher100_cpu_completed/a',
 'b':backup/'native_radius_cal_mechanism_actual_20261006T072031Z/snapshot_B',
 'c':backup/'native_radius_cal_mechanism_actual_20261006T072031Z/snapshot_C'}
uuids=['GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa']
capture_sha='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
dynamic={'inventory.json','finite_inventory.json','finite_c2_inventory.json','train_oracle_inventory.json','group_teacher_inventory.json','large_file_manifest.json'}
rows={}
for node,uuid in zip('abc',uuids):
 d=a.directory/node
 r=read(d/'receipt.json');ex=read(d/'actual_capture_exit.json')
 assert ex['exit_code']==0 and (d/'exit_code.txt').read_text().strip()=='0'
 raw=(d/'snapshot.zip').read_bytes();assert sha(raw)==r['sha256'] and len(raw)==r['bytes']
 assert ex['capture_source_sha256']==capture_sha
 assert ex['argv'][1].endswith('/capture_soft_vector_v19.py')
 assert ex['argv'][-1] in str(d/'unused') or ex['argv'][-1].endswith('_late_lease') or ex['argv'][-1].endswith('_final_window')
 with zipfile.ZipFile(d/'snapshot.zip') as z:
  assert z.testzip() is None
  m=json.loads(z.read('member_manifest.json'));large=json.loads(z.read('large_file_manifest.json'))
  assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(m)|{'member_manifest.json'}
  for n,it in m.items():
   data=z.read(n);assert len(data)==it['bytes'] and sha(data)==it['sha256'],n
  inv=json.loads(z.read('inventory.json'))
  assert inv['gpu'].split(',')[0].strip()==uuid and not inv['compute']
  assert inv['capture_source_sha256']==sha(z.read('source/capture_soft_vector_v19.py'))==capture_sha
  assert inv['actual_process_argv'] is None
  for n in ['group_teacher_inventory.json','finite_inventory.json','finite_c2_inventory.json','train_oracle_inventory.json']:
   if n in m:
    info=json.loads(z.read(n));assert not info.get('compute','')
    assert not info.get('matching_processes',[])
  lease_prefix='group_teacher/lease_capture_'+ex['argv'][-1].removesuffix('_late_lease').removesuffix('_final_window')+'/'
  assert lease_prefix+'lease_inventory.json' in m
  li=json.loads(z.read(lease_prefix+'lease_inventory.json'))
  assert li['gpu_uuid']==uuid and li['node']==node and not li['compute']
  assert sha(z.read(lease_prefix+'run_same_teacher_capture_v1.py'))==ex['wrapper_source_sha256']
  with zipfile.ZipFile(prior[node]/'snapshot.zip') as old:
   om=json.loads(old.read('member_manifest.json'));ol=json.loads(old.read('large_file_manifest.json'))
   same=0;closed=[]
   for n,it in om.items():
    if n in dynamic:continue
    if m.get(n)==it:same+=1;continue
    assert n.endswith('.log') and it['bytes']==0 and n.split('/')[-1].startswith('capture_'), 'Old evidence changed: '+n
    data=z.read(n).decode();assert data.startswith('NEW_RESEARCH_CAPTURE_COMPLETE ') and len(data.splitlines())==1
    closed.append(n)
   for n,it in ol.items():
    assert large[n]['sha256']==it['sha256'] and large[n]['bytes']==it['bytes'],n
  newlarge={n:it for n,it in large.items() if n not in ol}
  # New arrays referenced on A must match already downloaded original CPU/collection evidence;
  # this snapshot does not itself assert another large-file transfer.
  if newlarge:
   known={}
   for f in backup.rglob('*.npz'):
    if f.stat().st_size>8*1024**2:
     digest=hashlib.sha256()
     with f.open('rb') as h:
      for b in iter(lambda:h.read(1024*1024),b''):digest.update(b)
     known[(f.stat().st_size,digest.hexdigest())]=str(f)
   for n,it in newlarge.items():
    assert (it['bytes'],it['sha256']) in known,'New large reference not linked to permanent original: '+n
    it['permanent_original']=known[(it['bytes'],it['sha256'])]
  rows[node]={'actual_capture_started_utc':ex['started_utc'],'actual_capture_finished_utc':ex['finished_utc'],
   'inventory_utc':inv['utc'],'gpu':inv['gpu'],'compute':inv['compute'],'data_free':inv['data_disk_free'],
   'zip_bytes':r['bytes'],'zip_sha256':r['sha256'],'members':len(m),'large_references':len(large),
   'prior_scientific_members_unchanged':same,'prior_large_references_unchanged':len(ol),
   'prior_natural_stdout_closures':closed,'new_large_references_linked':newlarge,
   'new_complete_weight_download':False}
out={'status':'THREE_ACTUAL_LATE_LEASE_CAPTURES_SHA_CRC_MEMBERS_AND_PRIOR_EVIDENCE_VERIFIED',
 'audit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,
 'missing_historical_windows':['2026-10-06T08:08Z','2026-10-06T10:08Z'],
 'scope':'Actual current captures only; no retrospective capture, no new GPU experiment or CPU model forward. Large weights were previously preserved independently.'}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
