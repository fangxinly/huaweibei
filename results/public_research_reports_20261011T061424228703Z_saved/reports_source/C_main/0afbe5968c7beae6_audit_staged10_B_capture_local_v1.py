"""Audit original completed other-node CPU capture and full reference association."""
import datetime,hashlib,json,zipfile
from pathlib import Path
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference_first10_actual_20261006T151954Z')
B=BASE/'b'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(B/'capture_receipt.json');ex=read(B/'actual_capture_exit.json');cpu=read(BASE/'B_full_state_CPU_original_receipt.json')
assert ex['exit_code']==0 and ex['natural_wait_verified'] and ex['child_pid']==r['pid'] and ex['child_full_argv'][1:]==r['argv']
assert ex['capture_source_sha256']==r['capture_source_sha256']==sha(Path(__file__).with_name('capture_staged_CPU_audit_v1.py'))
assert sha(B/'snapshot.zip')==r['snapshot_sha256'] and not r['CPU_model_forward'] and not r['GPU_used']
raw=B/'original_small_files';raw.mkdir(exist_ok=True)
with zipfile.ZipFile(B/'snapshot.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==r['members']
 mf=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(mf['small_members'])|{'member_manifest.json'}
 for n,m in mf['small_members'].items():
  assert not n.startswith('/') and '..' not in Path(n).parts
  data=z.read(n);assert hashlib.sha256(data).hexdigest()==m['sha256'] and len(data)==m['bytes']
  p=raw/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 assert z.read('audit_root/cpu_original_receipt.json')==(BASE/'B_full_state_CPU_original_receipt.json').read_bytes()
 assert z.read('audit_root/cpu_original_receipt.actual_exit.json')==(BASE/'B_full_state_CPU_original_exit.json').read_bytes()
 assert mf['original_CPU_receipt_sha256']==sha(BASE/'B_full_state_CPU_original_receipt.json')
 assert mf['original_CPU_natural_exit_sha256']==sha(BASE/'B_full_state_CPU_original_exit.json')
local=read(BASE/'local_D_joint_audit_v2.json')
for n in ('selected_best_full.pt','complete_resume_full.pt'):
 record=mf['large_references_not_downloads']['audit_root/run/out/'+n]
 assert record['sha256']==local['full_files'][n]['sha256']==cpu['states'][n]['file_sha256']
 assert record['bytes']==local['full_files'][n]['bytes'] and record['stable_before_after']
proof={'status':'ACTUAL_B_FULL_CPU_ROOT_CAPTURE_SHA_CRC_UNIQUE_AND_D_WHOLE_FILE_ASSOCIATION_PASSED',
 'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'capture_actual_utc':r['actual_utc'],
 'capture_receipt_sha256':sha(B/'capture_receipt.json'),'capture_exit_sha256':sha(B/'actual_capture_exit.json'),
 'snapshot_sha256':r['snapshot_sha256'],'members':r['members'],'large_original_reference_count':len(mf['large_references_not_downloads']),
 'CPU_model_forward':False,'new_GPU_science':False,'source_sha256':sha(__file__)}
(BASE/'B_capture_local_joint_audit.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
