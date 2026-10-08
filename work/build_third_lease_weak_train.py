"""Freeze the single fixed40 loss experiment only after the actual D/B joint passes."""
import ast,datetime,hashlib,json,shutil,sys,zipfile
from pathlib import Path
base=Path('D:/CodexBackups/selective_flow_20261003_1105')
pre=base/'weak_retention_pilot40_20261007T143015Z'
stamp,clock=sys.argv[1:3]
root=Path('work')/('weak_retention_train40_'+stamp)
assert not root.exists()
joint=pre/'precheck_D_B_CPU_joint.json'
j=json.loads(joint.read_text(encoding='utf8'));assert j['status']=='GROUP5_PRECHECK_D_OTHER_CPU_COMPLETE'
assert j['whole_checkpoint_sha256']=='8ef4824966d2a1d4d5ce4a45b9a82f136af0240f6d6e64bf48fb1243f2116504'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
write=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
plan=json.loads((pre/'bundle/precheck_execution_protocol.json').read_text(encoding='utf8'))
assert j['protocol_sha256']==sha(pre/'bundle/precheck_execution_protocol.json')
root.mkdir();bundle=root/'bundle';bundle.mkdir();parents=bundle/'parents';parents.mkdir()
for p in (pre/'bundle').iterdir():
    if p.is_file() and p.suffix in ('.py','.npy') or p.name=='split.json':shutil.copy2(p,bundle/p.name)
for name,h in plan['source_sha256'].items():
    assert sha(bundle/name)==h;ast.parse((bundle/name).read_text(encoding='utf8'))
shutil.copy2(joint,parents/'precheck_D_B_CPU_joint.json')
free=shutil.disk_usage(base).free
assert free>=4400000000,free
capacity=dict(actual_utc=utc(),actual_D_free_after_complete_precheck_bytes=free,
    final_complete_resume_including_selected_max_bytes=3400000000,additional_margin_bytes=1000000000,
    required_before_train_bytes=4400000000,existing_precheck_full_D_bytes=2225144414,
    basis='Previous same-model final complete resume including selected2966458931B. No additional intermediate full local downloads.',
    full_ten_training_storage_reserved=False)
write(parents/'actual_local_capacity.json',capacity)
plan.update(status='PILOT40_EXECUTION_FROZEN',execution_enabled=True,authorized_stages=['train'],
    actualclock_training_protocol_freeze_UTC=clock,actual_training_protocol_written_UTC=utc(),
    precheck_D_other_CPU_joint={'anchored_flow':{'path':'parents/precheck_D_B_CPU_joint.json','sha256':sha(joint)}},
    complete_storage_reservation_reference={'path':'parents/actual_local_capacity.json','sha256':sha(parents/'actual_local_capacity.json')},
    remaining_queue_execution_budget_seconds=6000,
    execution_budget_basis='Actual new precheck projects5219.702440872788 seconds; allow6000 plus at least7200 saving reserve.',
    single_candidate='Same fixed40 anchored flow, adding coefficient1 FIT-only weak harm excess; no sweep',
    objective=dict(plan['objective'],weak_harm_excess=1.0),
    precheck_plan_SHA=j['protocol_sha256'],
    pending=[],new_real_training_started=False,
    selection_and_reporting_scope='INNER strictmin earliest all-row MSE only. Prior official TEST participates in pooled FIT/INNER under human merge authorization; separate official-role VAL/TEST scores are descriptive, not independent holdout.',
    prospective_once_scores=['selected INNER five metrics; preserve fold0 OUTER predictions for eventual complete grouped evaluation; fixed official-role VAL and TEST descriptive five metrics after actual D/B preservation, without deployment selection'])
write(bundle/'training_execution_protocol.json',plan)
archive=root/'training_bundle.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(bundle.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(bundle).as_posix())
with zipfile.ZipFile(archive) as z:assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
operator_source=Path('work/dispatch_third_lease_weak_train.py')
operator=dict(stage='train',operator_source_SHA=sha(operator_source),
    archive='/data/coding/weak_retention_train40_source_'+stamp+'.zip',archive_SHA=sha(archive),
    bundle='/data/coding/weak_retention_train40_source_'+stamp,
    assets='/data/coding/multimodal_flow_public_20261007T141430Z',
    protocol_SHA=sha(bundle/'training_execution_protocol.json'),
    root='/data/coding/weak_retention_train40_run_'+stamp,
    execution='/data/coding/weak_retention_train40_execution_'+stamp,
    operator_record='/data/coding/weak_retention_train40_operator_'+stamp)
write(root/'dispatch_plan.json',operator);shutil.copy2(operator_source,root/operator_source.name)
record=dict(status='SINGLE_FIXED40_WEAK_LOSS_TRAINING_PROTOCOL_FROZEN_NOT_DISPATCHED',actual_utc=utc(),
    protocol_SHA=operator['protocol_SHA'],archive_SHA=operator['archive_SHA'],source_SHA=plan['source_sha256'],
    joint_SHA=sha(joint),same_as_new_precheck_source_and_initialization=True,capacity=capacity,
    scope=dict(fold=0,epochs=40,updates=1880,fit=1494,inner=264,outer=437))
write(root/'freeze_record.json',record)
dest=base/root.name;shutil.copytree(root,dest)
current=dict(local=str(root.resolve()),D=str(dest),local_bundle=str(bundle.resolve()),stamp=stamp,**operator)
write(Path('work/third_lease_weak_training_current.json'),current)
print(json.dumps(current))
