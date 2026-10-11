from pathlib import Path
import argparse,hashlib,json,zipfile
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
a=argparse.ArgumentParser();a.add_argument('--audit',type=Path,required=True);a.add_argument('--snapshot',type=Path,required=True);a.add_argument('--run',type=Path,required=True);c=a.parse_args()
audit=json.loads(c.audit.read_text(encoding='utf-8'));assert audit['status']=='COUNTERFACTUAL_SNAPSHOT_AUDITED' and audit['stage']=='complete' and len(audit['rows'])==1
r=audit['rows'][0];assert r['full_checkpoint_verified'] and r['epochs']==100
cp=c.run/'best.pt';assert cp.resolve()==Path(r['checkpoint']).resolve() and cp.stat().st_size==r['checkpoint_bytes'] and sha(cp)==r['checkpoint_sha256'];assert sha(c.snapshot)==r['archive_sha256']
with zipfile.ZipFile(c.snapshot) as z:
 for name,e in {**r['files'],**r['extra_files']}.items():
  b=z.read('inflow_counterfactual_v5_deployment_20261005T0520Z/run_'+r['mode']+'/'+name);assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
  with (c.run/name).open('xb') as f:f.write(b)
  assert sha(c.run/name)==e['sha256']
files={**r['files'],'best.pt':{'bytes':r['checkpoint_bytes'],'sha256':r['checkpoint_sha256']}}
uuid={'a':'GPU-5902bbd4-2328-0822-777c-1311d53ee5a3','b':'GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
m={'mode':r['mode'],'source_gpu_uuid':uuid[r['node']],'files':files,'selection':r['selection'],'local_audit_sha256':sha(c.audit),'extra_metadata_saved':r['extra_files']}
p=c.run.parent/'independent_manifest.json'
with p.open('x',encoding='utf-8') as f:json.dump(m,f,indent=2)
print(json.dumps({'node':r['node'],'manifest':str(p),'manifest_sha256':sha(p),'run':str(c.run),'status':'SIX_CORE_AND_SHARED_PHASE_METADATA_EXTRACTED_FROM_AUDITED_CAPTURE'}))
