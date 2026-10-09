import argparse,ast,hashlib,json,shutil,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def read(f):return json.loads(f.read_text(encoding='utf8'))
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args()
old=read(base/'work/polarity_intensity_factorized_aux_resume100_qualified_20261008T205936Z/qualified_resume_plan.json')
for host,peer in [('A','B'),('B','A')]:
 pub=read(ev/(peer+'_interrupted_publication.json'));cap=read(ev/(peer+'_interrupted_capture.json'));client=read(ev/(peer+'_interrupted_publication_client_exit.json'))
 assert cap['capture_process_natural_exit']==client['natural_exit']==0 and cap['prior_training_natural_exit_unknown'] and not cap['prior_complete100']
 assert pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED' and all(r['source_sha256']==cap['archive_SHA'] for r in pub['assets']) and sum(r['bytes'] for r in pub['assets'])==cap['archive_bytes']
 root=base/'work'/(host+'_interrupted_peer400_qualified_'+a.tag);root.mkdir()
 for n in ('audit_interrupted_candidate400_v1.py','run_interrupted_peer400_capture_v1.py'):
  shutil.copy2(base/'work'/n,root/n);ast.parse((root/n).read_text())
 (root/'publication.json').write_text(json.dumps(pub,indent=2),encoding='utf8');(root/'interrupted_capture.json').write_text(json.dumps(cap,indent=2),encoding='utf8')
 q={n:old[n] for n in ('GPU_UUID','runtime_versions','asset_sha256','conservative_lease_end_UTC','remote_free_floor_bytes')}
 q.update(status='INTERRUPTED_PEER400_ORIGINAL_CPU_QUALIFIED',execution_node=host,source_peer=peer,actualclock_before_qualification_UTC=a.stamp,source_sha256={f.name:sha(f) for f in root.glob('*.py')},qualification_sha256={f.name:sha(f) for f in root.glob('*.json')})
 (root/'plan.json').write_text(json.dumps(q,indent=2),encoding='utf8')
 with zipfile.ZipFile(root/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
  for f in root.iterdir():
   if f.suffix!='.zip':z.write(f,f.name)
 print(json.dumps(dict(node=host,root=str(root),plan_SHA=sha(root/'plan.json'),source_SHA=sha(root/'source.zip'))))
