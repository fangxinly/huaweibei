"""A real Release publication must precede the peer audit protocol."""
import argparse,datetime,hashlib,json,shutil,zipfile
from pathlib import Path

def run(tag,stamp):
 base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z';clock=datetime.datetime.strptime(stamp,'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc)
 for node,peer,mode in [('A','B','regression_aux'),('B','A','factorized_aux')]:
  pub=json.loads((ev/(peer+'_prefix16_publication.json')).read_text(encoding='utf8'));cap=json.loads((ev/(peer+'_prefix16_capture.json')).read_text(encoding='utf8'));assert pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED' and cap['natural_exit']==0 and datetime.datetime.fromisoformat(pub['actual_UTC'])<=clock
  assert sum(a['bytes'] for a in pub['assets'] if a['source_sha256']==cap['archive_SHA'])==cap['archive_bytes']
  root=base/'work'/('candidate_peer_prefix_audit_'+node+'_'+tag);root.mkdir()
  for n in ('audit_public_prefix16_candidate_v2.py','run_candidate_peer_audit_capture_v2.py'):shutil.copy2(base/'work'/n,root/n)
  (root/'peer_publication.json').write_text(json.dumps(pub,indent=2),encoding='utf8');(root/'peer_capture.json').write_text(json.dumps(cap,indent=2),encoding='utf8')
  sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();p=json.loads((base/'work/polarity_intensity_prefix_v2_20261008T203225Z/native_preparation_plan.json').read_text(encoding='utf8'))
  p.update(status='ACTUAL_PEER_PREFIX16_PUBLICATION_BEFORE_CPU_AUDIT_FROZEN',actualclock_freeze_UTC=stamp,execution_node=node,candidate_mode=mode,source_sha256={f.name:sha(f) for f in root.glob('*.py')},proof_sha256={f.name:sha(f) for f in root.glob('*.json')})
  f=root/'audit_plan.json';f.write_text(json.dumps(p,indent=2),encoding='utf8')
  with zipfile.ZipFile(root/'audit_source.zip','x',zipfile.ZIP_DEFLATED) as z:
   for n in root.iterdir():
    if n.suffix!='.zip':z.write(n,n.name)
  print(json.dumps(dict(node=node,peer=peer,mode=mode,root=str(root),plan_SHA=sha(f),bundle_SHA=sha(root/'audit_source.zip'))))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args();run(a.tag,a.stamp)
