"""CPU-only target SHA verification after seven new run files fully arrive."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 a=argparse.ArgumentParser()
 a.add_argument('--directory',type=Path,required=True)
 a.add_argument('--manifest',type=Path,required=True)
 a.add_argument('--expected-manifest-sha',required=True)
 a.add_argument('--target-gpu-uuid',required=True)
 a.add_argument('--out',type=Path,required=True)
 cli=a.parse_args()
 assert not cli.out.exists()
 assert sha(cli.manifest)==cli.expected_manifest_sha
 expected=json.loads(cli.manifest.read_text(encoding='utf-8'))
 names={'best.pt','protocol.json','batch_orders.npy','history.json','selection.json','results.json','predictions.npz'}
 assert set(expected['files'])==names
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().strip()
 assert uuid==cli.target_gpu_uuid
 assert expected['source_gpu_uuid']!=uuid, 'Independent node required'
 measured={}
 for name in sorted(names):
  p=cli.directory/name
  measured[name]={'sha256':sha(p),'bytes':p.stat().st_size}
  assert measured[name]==expected['files'][name],name
 selection=json.loads((cli.directory/'selection.json').read_text())
 assert selection==expected['selection']
 assert measured['best.pt']['sha256']==selection['checkpoint_sha256']
 assert measured['protocol.json']['sha256']==selection['protocol_sha256']
 r={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'NEW_INFLOW_INDEPENDENT_LEASED_COPY_ALL_SEVEN_FILES_SHA_VERIFIED','target_gpu_uuid':uuid,'source_gpu_uuid':expected['source_gpu_uuid'],'mode':expected['mode'],'directory':str(cli.directory.resolve()),'manifest_sha256':cli.expected_manifest_sha,'files':measured,'selection':selection,'local_audit_sha256':expected['local_audit_sha256'],'verification':'CPU streaming SHA only; no model import, CUDA inference or optimizer','limits':'Lease copy is temporary; permanent local checkpoint separately required.'}
 with cli.out.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
 print(json.dumps(r))

if __name__=='__main__':main()
