"""Verify actual live recovery against interrupted epoch records and preserve small originals."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
def read(f):return json.loads(f.read_text(encoding='utf8'))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);p.add_argument('--tag',required=True);a=p.parse_args();files=[];proofs={}
for node in ('A','B'):
 dispatch=read(ev/(node+'_recovery400_actual_dispatch.json'));preflight=read(ev/(node+'_recovery400_fresh_preflight.json'));detached=read(ev/(node+'_recovery400_detached_dispatch.json'));restore=read(ev/(node+'_recovery400_restore.json'));progress=read(ev/(node+'_recovery400_progress.json'));old=read(ev/(node+'_interrupted_history_original.json'))
 assert progress['epoch']>=11 and progress['steps']==progress['epoch']*40 and restore['restored_optimizer_steps']==restore['scheduler_step']==400 and restore['full_RNG_restored'] and detached['detached_session']
 expected=old[progress['epoch']-1]
 assert expected['state_SHA']==progress['state_SHA'] and expected['prediction_SHA']==progress['prediction_SHA'] and expected['DEV_batch_MSE']==progress['DEV_batch_MSE'] and not preflight['compute'].strip()
 proofs[node]=dict(child_PID=dispatch['pid'],actual_dispatch_UTC=dispatch['actual_UTC'],epoch=progress['epoch'],steps=progress['steps'],recorded_state_and_prediction_SHA_identical=True,input_SHA=restore['input_SHA'],current_training_not_completed=True)
 for suffix in ('recovery400_actual_dispatch.json','recovery400_fresh_preflight.json','recovery400_detached_dispatch.json','recovery400_restore.json','recovery400_progress.json','interrupted_history_original.json'):files.append(ev/(node+'_'+suffix))
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('matched_recovery_actual_start_'+a.tag);dest.mkdir();archive=dest/'all_small_original_recovery_start_evidence.zip';members={f.name:sha(f) for f in files}
assert shutil.disk_usage('C:/').free>200_000_000 and shutil.disk_usage('D:/').free>500_000_000
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,f.name)
 z.writestr('member_SHA.json',json.dumps(members))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
r=dict(status='BOTH_ACTUAL_RECOVERY400_TRAJECTORIES_MATCH_PRIOR_AND_SMALL_ORIGINALS_C_D_PASSED',actualclock_UTC=a.stamp,proofs=proofs,D_original=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,no_new_final_VAL_TEST_five=True)
(ev/'recovery_actual_start_C_D_verification.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r))
