"""Freeze final audit/infer/score only from actual original candidate receipts."""
import argparse,ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path

base=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--stage',choices=['audit','infer','score'],required=True)
p.add_argument('--training-node',choices=['A','B'],required=True)
p.add_argument('--execution-node',choices=['A','B','C'],required=True)
for name in ('evidence','tools','out'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--stamp',required=True);a=p.parse_args()
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def read(f):return json.loads(f.read_text(encoding='utf8'))
def receipt(name):return read(a.evidence/(a.training_node+'100_'+name+'.json'))
cap=receipt('capture_receipt');exit=receipt('natural_exit');pub=receipt('publication_receipt');client=receipt('publication_client_exit');tr=receipt('training_result');members=receipt('member_SHA')
assert cap['natural_exit']==exit['natural_exit']==client['natural_exit']==0
assert cap['ZIP_CRC_unique_all_members'] and pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED'
assert any(v['source_sha256']==cap['archive_SHA'] for v in pub['assets'])
m=tr['metadata'];mode=dict(A='factorized_aux',B='regression_aux')[a.training_node]
assert m['steps']==4000 and m['epochs']==100 and m['candidate_mode']==mode
assert m['prefix_updates_counted_in4000'] and m['recovery_replayed_updates_not_new_independent_replicate']
assert tr['checkpoint_SHA']==members['out/complete_final_and_selected.pt']
native_ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
nr=read(native_ev/'A_candidate_posttrain_native_result.json');ne=read(native_ev/'A_candidate_posttrain_native_exit.json');nc=read(native_ev/'A_candidate_posttrain_native_capture.json')
assert nr['status']=='CANDIDATE_BOTH_MODES_SAVED_TAIL_AND_INDEPENDENT_FIVE_CPU_PASSED'
assert ne['natural_exit']==nc['natural_exit']==0 and not nr['real_data_or_labels_indexed']
assert nr['source_SHA']==sha(a.tools/'qualify_candidate_posttrain_v1.py')
with zipfile.ZipFile(native_ev/'A_candidate_posttrain_native_original.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    manifest=json.loads(z.read('member_SHA.json'))
    for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    # Bind every executable being reused to the actual native source capture.
    for f in a.tools.glob('*.py'):assert sha(f)==manifest['original_source/'+f.name],f.name
assert sha(native_ev/'A_candidate_posttrain_native_original.zip')==nc['archive_SHA']
clock=datetime.datetime.strptime(a.stamp,'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc)
for r in (cap,exit,pub,client,tr,nr,ne,nc):assert datetime.datetime.fromisoformat(r['actual_UTC'])<=clock
assert (a.stage=='audit' and a.execution_node!=a.training_node) or (a.stage!='audit' and a.execution_node==a.training_node)
a.out.mkdir()
for f in a.tools.iterdir():
    if f.suffix in ('.py','.npy'):shutil.copy2(f,a.out/f.name)
shutil.copy2(native_ev/'A_candidate_posttrain_native_result.json',a.out/'candidate_posttrain_native_result.json')
shutil.copy2(native_ev/'A_candidate_posttrain_native_exit.json',a.out/'candidate_posttrain_native_exit.json')
source=base/'work'/(a.training_node+'_candidate_recovery400_qualified_20261009T002805Z')
plan=read(source/'qualified_resume_plan.json');baseline=read(base/'work/official_anchored_upgrade_20261008T053429Z/official_score_protocol.json')
if a.execution_node=='C':
    cr=read(native_ev/'C_candidate_pureCPU_runtime_result.json');cc=read(native_ev/'C_candidate_pureCPU_runtime_capture.json')
    assert cc['natural_exit']==0 and cr['status']=='NEW_ISOLATED_CANDIDATE_PURE_CPU_RUNTIME_PINNED_PASSED'
    assert cr['runtime_versions']==plan['runtime_versions'] and cr['CPU_only']
    assert cr['UUID']=='GPU-daa2c09a-4ce5-26dd-b375-6b61242c6795'
    plan['GPU_UUID']={**plan['GPU_UUID'],'C':cr['UUID']}
    plan['pure_CPU_runtime_original_reference']=dict(runtime=cr,capture=cc,execution_requires_new_fresh_preflight=True)
plan.update(status='ACTUAL_OFFICIAL_CANDIDATE_'+a.stage.upper()+'_STAGE_FROZEN',actualclock_stage_freeze_UTC=a.stamp,candidate_mode=mode,training_node=a.training_node,execution_node=a.execution_node,final_checkpoint_bytes=tr['checkpoint_bytes'],posttrain_stage_budget_seconds=dict(audit=3600,infer=1800,score=300),training_original_reference=dict(archive_SHA=cap['archive_SHA'],member_SHA=members,training_plan_SHA=m['plan_SHA'],public_release=pub['release_url']),selected_state_SHA=m['selected_state_SHA'],official_role_IDs=baseline['official_role_IDs'],fixed_CaReFlow_five=baseline['fixed_CaReFlow_five'],CaReFlow_baseline_original_reference=baseline['CaReFlow_baseline_original_reference'],recovery_compute_disclosure=dict(logical_updates=4000,recorded_interrupted_last_step=dict(A=3893,B=3873)[a.training_node],known_replayed_updates=dict(A=3493,B=3473)[a.training_node],unknown_unlogged_interrupted_updates_not_excluded=True,not_new_independent_replicate=True))
plan['posttrain_native_CPU_qualification']={n:sha(a.out/n) for n in ('candidate_posttrain_native_result.json','candidate_posttrain_native_exit.json')}
if a.stage=='audit':
    # Pure CPU needs neither encoder weights nor data; its own exact dependency
    # and source checks remain in force. Lower floor applies to CPU storage only.
    plan['asset_sha256']={};plan['remote_free_floor_bytes']=500_000_000
    plan['CPU_only_archive_plus_checkpoint_space_required']=True
if a.stage in ('infer','score'):
    audit=receipt('audit_result');ae=receipt('audit_exit')
    assert audit['status']=='CANDIDATE_FULL4000_ADAM_SELECTION_AND_CPU_TAIL_PASSED' and ae['natural_exit']==0
    assert audit['checkpoint_SHA']==tr['checkpoint_SHA'] and audit['selected_state_SHA']==m['selected_state_SHA'] and audit['all_Adam_steps']==4000
    assert audit['candidate_mode']==mode and datetime.datetime.fromisoformat(audit['actual_UTC'])<=clock
    plan['training_Release_B_CPU_gate']=dict(original=cap,publication=pub,peer_original_CPU=audit,peer_natural_exit=ae)
if a.stage=='score':
    inf=receipt('inference_result');ic=receipt('inference_capture');ie=receipt('inference_exit');ip=receipt('prediction_publication');pa=receipt('prediction_byte_audit')
    assert ic['natural_exit']==ie['natural_exit']==0 and inf['prediction_SHA']==pa['prediction_SHA']
    assert pa['SHA_CRC_unique_member_SHA_passed'] and ip['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED'
    assert inf['selected_state_SHA']==m['selected_state_SHA']
    for r in (inf,ic,ie,ip,pa):assert datetime.datetime.fromisoformat(r['actual_UTC'])<=clock
    plan.update(prediction_Release_B_CPU_gate=dict(capture=ic,publication=ip,peer_byte_audit=pa),prediction_SHA=inf['prediction_SHA'],score_once_token='/data/coding/candidate_'+mode+'_score_once_'+m['selected_state_SHA'])
plan['source_sha256']={f.name:sha(f) for f in a.out.iterdir() if f.suffix in ('.py','.npy')}
path=a.out/('official_'+a.stage+'_protocol.json');path.write_text(json.dumps(plan,indent=2),encoding='utf8')
for f in a.out.glob('*.py'):ast.parse(f.read_text(encoding='utf8'))
with zipfile.ZipFile(a.out/'stage_source.zip','x',zipfile.ZIP_DEFLATED) as z:
    for f in a.out.iterdir():
        if f.suffix!='.zip':z.write(f,f.name)
print(json.dumps(dict(stage=a.stage,mode=mode,execution_node=a.execution_node,plan_SHA=sha(path),source_SHA=sha(a.out/'stage_source.zip'))))
