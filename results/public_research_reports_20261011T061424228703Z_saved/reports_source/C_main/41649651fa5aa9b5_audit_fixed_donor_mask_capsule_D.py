import argparse,datetime,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--node',choices=['A','B'],required=True);a=p.parse_args()
d=Path('D:/CodexBackups/selective_flow_20261003_1105/fixed_weak_donor_mask_20261007T161226Z')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cap=d/(a.node+'_complete_capture.zip');cr=json.loads((d/(a.node+'_capture_receipt.json')).read_text(encoding='utf8'))
assert sha(cap)==cr['sha256'] and cap.stat().st_size==cr['bytes']
with zipfile.ZipFile(cap) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 manifest=json.loads(z.read('manifest.json'))
 assert set(manifest['member_sha256'])==set(z.namelist())-{'manifest.json'}
 for k,h in manifest['member_sha256'].items():assert hashlib.sha256(z.read(k)).hexdigest()==h
 plan=json.loads(z.read('protocol.json'));r=json.loads(z.read('original/actual_stage_receipt.json'));child=json.loads(z.read('execution/actual_child.json'));natural=json.loads(z.read('execution/natural_exit.json'))
 assert manifest['plan_sha256']==hashlib.sha256(z.read('protocol.json')).hexdigest()==r['plan_sha256']==child['plan_sha256']==natural['plan_sha256']
 assert natural['natural_exit']==0 and natural['pid']==child['pid']==r['pid']
 assert natural['fullargv']==child['fullargv']==r['fullargv']
 assert manifest['original_receipt_sha256']==hashlib.sha256(z.read('original/actual_stage_receipt.json')).hexdigest()
 script=next(k for k in z.namelist() if k.startswith('fixed_weak_pilot_donor_mask_diagnostic_'))
 assert hashlib.sha256(z.read(script)).hexdigest()==plan['diagnostic_source_sha256']
 for name in z.namelist():
  path=(d/(a.node+'_original_capsule')/name).resolve();assert path.is_relative_to((d/(a.node+'_original_capsule')).resolve());data=z.read(name)
  if path.exists():assert path.read_bytes()==data
  else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
 if a.node=='B':
  ar=json.loads((d/'A_original_capsule/original/actual_stage_receipt.json').read_text(encoding='utf8'))
  assert r['original_receipt_sha256']==sha(d/'A_original_capsule/original/actual_stage_receipt.json') and r['saved_metrics_exact_equal']
  assert r['predictions']==ar['prediction_sha256']
record=dict(status='ACTUAL_'+a.node+'_DONOR_MASK_CAPTURE_D_SHA_CRC_UNIQUE_ALL_ORIGINAL_MEMBER_SOURCE_ARGV_PASSED',actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),capsule_SHA=cr['sha256'],member_count=len(manifest['member_sha256'])+1,pid=r['pid'],natural_exit=0,plan_SHA=r['plan_sha256'],parent_checkpoint_SHA_ref=plan['checkpoint_sha256'],CPU_model_forward=False if a.node=='B' else None)
(d/(a.node+'_actual_D_audit.json')).write_text(json.dumps(record,indent=2),encoding='utf8');print(json.dumps(record))
