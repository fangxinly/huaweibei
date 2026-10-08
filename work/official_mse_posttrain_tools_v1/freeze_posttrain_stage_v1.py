"""Create new stage plan only from actual completed preserved original receipts."""
import argparse,ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['audit','infer','score'],required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stamp',required=True);p.add_argument('--received-archive');a=p.parse_args()
base=Path(__file__).resolve().parents[2];bundle=base/'work/autonomous_mse100_resume_qualified_20261008T182923Z';tools=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((a.evidence/n).read_text(encoding='utf8'))
cap=read('A100_capture_receipt.json');exit=read('A100_natural_exit.json');pub=read('A100_publication_receipt.json');client=read('A100_publication_client_exit.json');tr=read('A100_training_result.json');members=read('A100_member_SHA.json')
native=read('B_posttrain_native_result.json');ne=read('B_posttrain_native_exit.json')
assert native['status']=='CPU_SYNTHETIC_POSTTRAIN_FLOW_ON_OFF_AND_INDEPENDENT_FIVE_PASSED' and ne['natural_exit']==0 and native['source_SHA']==sha(tools/'qualify_posttrain_contract_v1.py') and not native['real_data_or_labels_indexed']
assert cap['status']=='ACTUAL_RESUMED_TRAIN100_COMPLETE' and cap['natural_exit']==exit['natural_exit']==client['natural_exit']==0 and pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED' and any(r['source_sha256']==cap['archive_SHA'] for r in pub['assets'])
assert tr['metadata']['steps']==4000 and tr['metadata']['prefix_updates_counted_in4000'] and tr['checkpoint_SHA']==members['out/complete_final_and_selected.pt']
stamp=datetime.datetime.strptime(a.stamp,'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc)
assert all(datetime.datetime.fromisoformat(r['actual_UTC'])<=stamp for r in (cap,pub,tr,native)), 'Stage timestamp precedes required actual evidence'
a.out.mkdir(exist_ok=False)
for f in bundle.iterdir():
 if f.suffix in ('.py','.npy'):shutil.copy2(f,a.out/f.name)
for f in tools.glob('*.py'):
 if f.name!=Path(__file__).name:shutil.copy2(f,a.out/f.name)
for n in ('B_posttrain_native_result.json','B_posttrain_native_exit.json'):shutil.copy2(a.evidence/n,a.out/n)
plan=json.loads((bundle/'qualified_resume_plan.json').read_text(encoding='utf8'));old=json.loads((base/'work/official_anchored_upgrade_20261008T053429Z/official_score_protocol.json').read_text(encoding='utf8'))
plan.update(status='ACTUAL_OFFICIAL_MSE_'+a.stage.upper()+'_STAGE_FROZEN',actualclock_stage_freeze_UTC=a.stamp,posttrain_stage_budget_seconds=dict(audit=3600,infer=1800,score=300),training_original_reference=dict(archive_SHA=cap['archive_SHA'],member_SHA=members,training_plan_SHA=tr['metadata']['plan_SHA'],public_release=pub['release_url']),selected_state_SHA=tr['metadata']['selected_state_SHA'],official_role_IDs=old['official_role_IDs'],fixed_CaReFlow_five=old['fixed_CaReFlow_five'],CaReFlow_baseline_original_reference=old['CaReFlow_baseline_original_reference'])
plan['posttrain_native_CPU_qualification']={n:sha(a.out/n) for n in ('B_posttrain_native_result.json','B_posttrain_native_exit.json')}
if a.received_archive:
 assert a.stage=='audit'
 dv=read('A100_D_full_original_verification.json');assert dv['status']=='NEW_D_COMPLETE_ORIGINAL_SHA_CRC_UNIQUE_ALL_MEMBERS_PASSED' and dv['archive_SHA']==cap['archive_SHA'] and dv['archive_bytes']==cap['archive_bytes']
 plan['received_original_transport']=dict(method='D_verified_original_SFTP',remote_archive=a.received_archive,D_verification=dv,previous_failed_download=read('B100_first_audit_capture.json'),public_download=False)
if a.stage in ('infer','score'):
 audit=read('B100_audit_result.json');ae=read('B100_audit_exit.json');assert ae['natural_exit']==0 and audit['checkpoint_SHA']==tr['checkpoint_SHA'] and audit['selected_state_SHA']==tr['metadata']['selected_state_SHA'] and audit['all_Adam_steps']==4000
 assert all(datetime.datetime.fromisoformat(r['actual_UTC'])<=stamp for r in (audit,ae)), 'Stage timestamp precedes actual full CPU audit'
 plan['training_Release_B_CPU_gate']=dict(original=cap,publication=pub,B_original_CPU=audit,B_natural_exit=ae)
if a.stage=='score':
 inference=read('A100_inference_result.json');ic=read('A100_inference_capture.json');ie=read('A100_inference_exit.json');ip=read('A100_prediction_publication.json');predaudit=read('B100_prediction_byte_audit.json');assert ic['natural_exit']==ie['natural_exit']==0 and inference['prediction_SHA']==predaudit['prediction_SHA'] and predaudit['SHA_CRC_unique_member_SHA_passed'] and ip['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED'
 assert all(datetime.datetime.fromisoformat(r['actual_UTC'])<=stamp for r in (inference,ic,ie,ip,predaudit)), 'Stage timestamp precedes actual prediction preservation'
 plan.update(prediction_Release_B_CPU_gate=dict(capture=ic,publication=ip,B_byte_audit=predaudit),prediction_SHA=inference['prediction_SHA'],score_once_token='/data/coding/official_mse100_score_once_'+tr['metadata']['selected_state_SHA'])
plan['source_sha256']={f.name:sha(f) for f in a.out.iterdir() if f.suffix in ('.py','.npy')};path=a.out/('official_'+a.stage+'_protocol.json');path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
for f in a.out.glob('*.py'):ast.parse(f.read_text(encoding='utf8'))
with zipfile.ZipFile(a.out/'stage_source.zip','x',zipfile.ZIP_DEFLATED) as z:
 for f in a.out.iterdir():
  if f.suffix!='.zip':z.write(f,f.name)
print(json.dumps(dict(stage=a.stage,plan_SHA=sha(path),bundle_SHA=sha(a.out/'stage_source.zip'))))
