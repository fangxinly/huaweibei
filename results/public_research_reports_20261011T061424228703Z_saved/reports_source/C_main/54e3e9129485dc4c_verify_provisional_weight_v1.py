from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,zipfile
a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True);a.add_argument('--receipt-sha',required=True);a.add_argument('--target-uuid');a.add_argument('--out',type=Path,required=True);c=a.parse_args();d=c.directory
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
receipt=d/'provisional_receipt.json';assert sha(receipt)==c.receipt_sha;r=json.loads(receipt.read_text());assert not r['training_finished'] and r['files']['best.pt']['bytes']>700_000_000
for name,e in r['files'].items():assert (d/name).stat().st_size==e['bytes'] and sha(d/name)==e['sha256']
with zipfile.ZipFile(d/'best.pt') as z:assert z.testzip() is None and len(z.namelist())>100
h=json.loads((d/'source_history.json').read_text());best=min(h,key=lambda x:x['valid_mse']);assert len(h)==r['epochs_observed'] and best['epoch']==r['best_epoch_so_far'] and best['valid_mse']==r['best_dev_batch_mse_so_far']
uuid=None
if c.target_uuid:
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().strip();assert uuid==c.target_uuid and uuid!=r['gpu_uuid']
report={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'INDEPENDENT_CPU_PROVISIONAL_FULL_WEIGHT_ALL_SHA_AND_ZIP_CRC_VERIFIED' if uuid else 'PERMANENT_LOCAL_PROVISIONAL_FULL_WEIGHT_ALL_SHA_AND_ZIP_CRC_VERIFIED','source_captured_at':r['captured_at'],'directory':str(d),'source_gpu_uuid':r['gpu_uuid'],'target_gpu_uuid':uuid,'receipt_sha256':c.receipt_sha,'epochs_observed':r['epochs_observed'],'best_epoch_so_far':r['best_epoch_so_far'],'files':r['files'],'training_complete':False,'limits':'Provisional fullweight backup, not100-epoch completed selection. Source receipt asserts stable read; no inference/optimizer/TEST.'}
assert not c.out.exists();c.out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
