import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args();receipt={}
files=[]
for n in ('A','B'):
 for stem in ('recovery400_first_native_failure','recovery400_native'):
  c=ev/(n+'_'+stem+'_capture.json');z=ev/(n+'_'+stem+'_original.zip');r=json.loads(c.read_text());assert sha(z)==r['archive_SHA'] and z.stat().st_size==r['archive_bytes']
  with zipfile.ZipFile(z) as f:
   assert f.testzip() is None and len(f.namelist())==len(set(f.namelist()))
   for name,h in json.loads(f.read('member_SHA.json')).items():assert hashlib.sha256(f.read(name)).hexdigest()==h
  files.extend([c,z]);receipt[n+'_'+stem]=r
 result=json.loads((ev/(n+'_recovery400_native_result.json')).read_text());exit=json.loads((ev/(n+'_recovery400_native_exit.json')).read_text())
 assert exit['natural_exit']==0 and result['final_steps']==440 and result['prediction_parameter_Adam_scheduler_python_numpy_torch_RNG_identical'] and result['qualified_modes']==['regression_aux','factorized_aux']
 files += [ev/(n+'_recovery400_native_result.json'),ev/(n+'_recovery400_native_exit.json'),ev/(n+'_interruption_observation.json'),ev/(n+'_interrupted_latest_progress.json'),ev/(n+'_warm400_ready.json')]
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('candidate_recovery_native_'+a.tag);dest.mkdir()
archive=dest/'complete_native_success_and_failure_small_originals.zip';members={f.name:sha(f) for f in files}
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,f.name)
 z.writestr('member_SHA.json',json.dumps(members))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
out=dict(status='NATIVE_RECOVERY400_TO440_BOTH_MODES_TWO_NODES_C_D_ALL_ORIGINAL_SHA_CRC_UNIQUE_VERIFIED',actualclock_UTC=a.stamp,D_original=str(archive),archive_SHA=sha(archive),bytes=archive.stat().st_size,proofs=receipt,real_recovery_started=False,prior_training_natural_exit_unknown=True)
(ev/'recovery400_native_C_D_verification.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(dict(status=out['status'],SHA=out['archive_SHA'],bytes=out['bytes'])))
