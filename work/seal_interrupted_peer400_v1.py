"""Save complete small peer CPU originals plus exact references; no large redownload."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def read(f):return json.loads(f.read_text(encoding='utf8'))
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args();files=[];proofs={}
for node,peer in [('A','B'),('B','A')]:
 cap=read(ev/(node+'_interrupted_peer400_seekfix_capture.json'));archive=ev/(node+'_interrupted_peer400_seekfix_original.zip');r=read(ev/(node+'_interrupted_peer400_seekfix_result.json'));source=read(ev/(peer+'_interrupted_capture.json'))
 assert cap['natural_exit']==0 and sha(archive)==cap['archive_SHA'] and archive.stat().st_size==cap['archive_bytes']
 assert r['archive_SHA']==source['archive_SHA'] and r['warm400_checkpoint_SHA']==source['warm400_checkpoint_SHA'] and r['all_Adam_steps']==400 and r['Adam_parameter_states']==345 and (r['public_Release_download'] or r.get('retained_public_Release_origin_archive_reuse'))
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for n,h in json.loads(z.read('member_SHA.json')).items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
 proofs[node]=r
 for suffix in ('interrupted_peer400_seekfix_capture.json','interrupted_peer400_seekfix_result.json','interrupted_peer400_seekfix_exit.json','interrupted_peer400_seekfix_original.zip','interrupted_capture.json','interrupted_publication.json','interrupted_publication_client_exit.json','interrupted_compute_cost.json'):
  files.append(ev/(node+'_'+suffix))
native=read(ev/'recovery400_native_C_D_verification.json');nativefile=Path(native['D_original']);assert sha(nativefile)==native['archive_SHA'];files.append(ev/'recovery400_native_C_D_verification.json')
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('interrupted_pair_peer400_'+a.tag);dest.mkdir();archive=dest/'complete_small_peer400_originals_and_refs.zip';members={f.name:sha(f) for f in files}
assert sum(f.stat().st_size for f in files)<40_000_000 and shutil.disk_usage('C:/').free>200_000_000 and shutil.disk_usage('D:/').free>sum(f.stat().st_size for f in files)+500_000_000
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,f.name)
 z.writestr('member_SHA.json',json.dumps(members))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
result=dict(status='INTERRUPTED_TWO_ORIGINALS_RELEASE_AND_PEER400_CPU_C_D_SMALL_ORIGINALS_VERIFIED',actualclock_UTC=a.stamp,D_original=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,proofs=proofs,original_training_natural_exit_unknown=True,no_large_D_capacity_claimed=True)
(ev/'interrupted_preservation_peer400_C_D_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(dict(status=result['status'],SHA=result['archive_SHA'],bytes=result['archive_bytes'])))
