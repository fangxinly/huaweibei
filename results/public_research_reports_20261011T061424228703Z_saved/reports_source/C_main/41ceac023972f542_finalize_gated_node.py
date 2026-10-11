from pathlib import Path
import argparse,hashlib,json,zipfile
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
a=argparse.ArgumentParser();a.add_argument('--audit',type=Path,required=True);a.add_argument('--snapshot',type=Path,required=True);a.add_argument('--run',type=Path,required=True);c=a.parse_args()
audit=json.loads(c.audit.read_text(encoding='utf-8'))
assert audit['status']=='GATED_SELECTED_NODES_COMPLETED_WHOLE_SHA_100_SELECTION_OFFICIAL_DEV_VERIFIED'
assert len(audit['rows'])==1
r=audit['rows'][0];assert r['full_checkpoint_verified'] and r['epochs']==100
cp=c.run/'best.pt';assert cp.resolve()==Path(r['checkpoint']).resolve() and cp.stat().st_size==r['checkpoint_bytes']
assert sha(c.snapshot)==r['archive_sha256']
with zipfile.ZipFile(c.snapshot) as z:
 for name,expected in r['files'].items():
  b=z.read('inflow_gated_v3_deployment_20261005T0241Z/run_'+r['mode']+'/'+name)
  assert len(b)==expected['bytes'] and hashlib.sha256(b).hexdigest()==expected['sha256']
  p=c.run/name
  with p.open('xb') as f:f.write(b)
  assert sha(p)==expected['sha256']
files=dict(r['files']);files['best.pt']={'bytes':r['checkpoint_bytes'],'sha256':r['checkpoint_sha256']}
uuid={'a':'GPU-5902bbd4-2328-0822-777c-1311d53ee5a3','b':'GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
m={'mode':r['mode'],'source_gpu_uuid':uuid[r['node']],'files':files,'selection':r['selection'],'local_audit_sha256':sha(c.audit)}
p=c.run.parent/'independent_manifest.json'
with p.open('x',encoding='utf-8') as f:json.dump(m,f,indent=2)
print(json.dumps({'node':r['node'],'manifest':str(p),'manifest_sha256':sha(p),'run':str(c.run),'status':'SIX_METADATA_EXTRACTED_FROM_VERIFIED_CAPTURE_MANIFEST_PREPARED'}))
