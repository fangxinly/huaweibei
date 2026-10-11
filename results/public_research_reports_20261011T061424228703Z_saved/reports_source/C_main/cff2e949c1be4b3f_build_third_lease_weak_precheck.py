"""Freeze one loss change against the completed fixed40 pilot; no old source edits."""
import ast,datetime,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105')
OLD=BASE/'anchored_flow_pilot40_20261007T091746Z/bundle'
WEAK=BASE/'weak_retention_training_candidate_20261007T124321Z/source/anchored_flow.py'
stamp=sys.argv[1]; clock=sys.argv[2]
root=Path('work')/('weak_retention_pilot40_'+stamp);root.mkdir();bundle=root/'bundle';bundle.mkdir()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dump=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
oldplan=json.loads((OLD/'pilot40_execution_protocol.json').read_text())
for n,h in oldplan['source_sha256'].items():assert sha(OLD/n)==h
for p in OLD.iterdir():
    if p.is_file() and (p.suffix in ('.py','.npy') or p.name=='split.json'):shutil.copy2(p,bundle/p.name)
original=(OLD/'anchored_flow.py').read_text(); candidate=WEAK.read_text()
def node(text,name):return next(n for n in ast.parse(text).body if getattr(n,'name',None)==name)
assert ast.dump(node(original,'AnchoredFlow'))==ast.dump(node(candidate,'AnchoredFlow'))
obj=node(candidate,'objective');lines=candidate.splitlines(keepends=True)
newobjective=''.join(lines[obj.lineno-1:obj.end_lineno])
prior_obj=node(original,'objective');prior_lines=original.splitlines(keepends=True)
anchored=''.join(prior_lines[:prior_obj.lineno-1])+newobjective+'\n'+''.join(prior_lines[prior_obj.end_lineno:])
(bundle/'anchored_flow.py').write_text(anchored,encoding='utf8')
assert ast.dump(node(anchored,'objective'))==ast.dump(node(candidate,'objective'))
assert ast.dump(node(anchored,'synthetic_check'))==ast.dump(node(original,'synthetic_check'))
runtime=bundle/'fold_runtime.py';source=runtime.read_text()
old="    if end>dt.datetime(2026,10,7,13,30,tzinfo=dt.timezone.utc):raise PermissionError('Unconfirmed lease extension')"
new="""    lease=plan['human_asset_and_lease_budget_reference']
    if lease.get('human_confirmed_new_duration_hours')!=24 or lease.get('third_lease_endpoints')!=[
        'REDACTED_SERVER_HOST.invalid:53423','REDACTED_SERVER_HOST.invalid:53428',
        'REDACTED_SERVER_HOST.invalid:53495']:
        raise PermissionError('New human lease evidence absent')
    if end>dt.datetime(2026,10,8,14,0,tzinfo=dt.timezone.utc):raise PermissionError('Beyond conservative new24h lease')"""
assert source.count(old)==1;runtime.write_text(source.replace(old,new),encoding='utf8')
for p in bundle.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
tested=subprocess.run([sys.executable,'-B','-m','unittest','test_fold_contract','test_pilot_budget'],cwd=bundle,capture_output=True,text=True)
(root/'local_guard_tests.stdout.log').write_text(tested.stdout);(root/'local_guard_tests.stderr.log').write_text(tested.stderr)
assert tested.returncode==0,tested.stderr

activation=json.loads(Path('outputs/第三租期恢复与弱情绪损失原生检查实际接续.json').read_text(encoding='utf8'))
free=shutil.disk_usage(BASE).free
capacity={'actual_utc':utc(),'D_free_before_precheck_bytes':free,'single_precheck_max_bytes':2300000000,
          'single_final_resume_including_selected_max_bytes':3400000000,'additional_margin_bytes':1000000000,
          'required_before_precheck_bytes':6700000000,'required_before_train_after_precheck_bytes':4400000000,
          'basis':'Measured prior same-model precheck2225144350B and final2966458931B; selected weights already embedded in complete final resume.',
          'human_authorized_exact_four_cleanup_record':activation['human_authorized_cleanup_actual'],
          'no_ten_model_or_full_fivefold_capacity_reserved':True}
assert free>=capacity['required_before_precheck_bytes']
dump(root/'actual_local_capacity.json',capacity)
plan=dict(oldplan)
plan.update(status='GROUP5_PRECHECK_ONLY_FROZEN',execution_enabled=True,
    actualclock_source_start_UTC=clock,actual_source_finish_UTC=utc(),
    methods=['anchored_flow'],authorized_methods=['anchored_flow'],authorized_stages=['precheck'],authorized_folds=[0],
    assigned_gpu_uuid={n:activation['public_restoration'][n]['receipt']['uuid'] for n in ('A','B','C')},
    precheck_D_other_CPU_joint={},
    source_sha256={p.name:sha(p) for p in sorted(bundle.glob('*.py'))},
    conservative_lease_end_UTC='2026-10-08T14:00:00+00:00',
    human_asset_and_lease_budget_reference={'human_confirmed_new_duration_hours':24,
        'human_reply':'24小时之后 那你保留合适的数据','platform_expiry_verified':False,
        'third_lease_endpoints':['REDACTED_SERVER_HOST.invalid:53423','REDACTED_SERVER_HOST.invalid:53428','REDACTED_SERVER_HOST.invalid:53495'],
        'actual_public_restoration_D':activation['D'],'conservative_new_end_not_platform_confirmation':True},
    complete_storage_reservation_reference={'path':str((root/'actual_local_capacity.json').resolve()),'sha256':sha(root/'actual_local_capacity.json')},
    remaining_queue_execution_budget_seconds=5000,
    checkpoint_cadence='Mutable recovery every10; only fixed precheck and one final complete state reserved locally, no extra intermediate downloads.',
    score_rule='Same previously explored fold0 INNER selection; no OUTER/TEST metric used for updates or selection. Full fivefold incomplete.',
    global_score_token='/data/coding/multimodal_flow_public_20261007T141430Z/locks/weak_pilot40.score_once',
    source_parent_fixed40_plan_SHA=sha(OLD/'pilot40_execution_protocol.json'),
    candidate_native_objective_source_SHA=sha(WEAK),
    loss_only_change='FIT abs(y)<=1 mean relu((p-y)^2-(stop_gradient(b)-y)^2), coefficient1; all four existing losses, architecture, initial seed/order/scheduler and inference unchanged.',
    coupled_comparator_limitation='No gradient through comparator term; b remains shared and can drift. Synthetic pass is not a risk bound.',
    prospective_training_scope={'epochs':40,'updates':1880,'FIT':1494,'INNER':264,'OUTER':437,'fold':0},
    new_real_training_started=False,full_fivefold_complete=False,
    pending=['This exact new objective must pass actual full-model precheck and complete D plus other-node CPU before train protocol freeze.'])
for n,h in plan['order_sha256'].items():assert sha(bundle/n)==h
assert sha(bundle/'split.json')==plan['split_sha256']
dump(bundle/'precheck_execution_protocol.json',plan)
record={'actual_utc':utc(),'status':'SINGLE_WEAK_LOSS_PRECHECK_SOURCE_FROZEN_NOT_DISPATCHED',
        'source_model_AST_same':True,'objective_AST_same_as_B_native_checked_candidate':True,
        'old_synthetic_device_and_decoder_checks_retained':True,
        'plan_SHA':sha(bundle/'precheck_execution_protocol.json'),'source_sha256':plan['source_sha256'],
        'source_or_order_changes_beyond_objective_and_new_lease_runtime':False,
        'local_guard_tests_natural_exit':tested.returncode,'public_root':'/data/coding/multimodal_flow_public_20261007T141430Z'}
dump(root/'freeze_record.json',record)
archive=root/'precheck_bundle.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in bundle.iterdir():
        if p.is_file():z.write(p,p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
record.update(archive_SHA=sha(archive),archive_bytes=archive.stat().st_size)
dump(root/'freeze_record.json',record)
dest=BASE/('weak_retention_pilot40_'+stamp);shutil.copytree(root,dest)
current={'local':str(root.resolve()),'D':str(dest),'bundle':str((root/'bundle').resolve()),'stamp':stamp,
         'plan_SHA':record['plan_SHA'],'archive_SHA':record['archive_SHA'],
         'remote_bundle':'/data/coding/weak_retention_pilot40_source_'+stamp,
         'remote_run':'/data/coding/weak_retention_pilot40_precheck_'+stamp,
         'remote_execution':'/data/coding/weak_retention_pilot40_precheck_execution_'+stamp,
         'public_root':record['public_root'],'plan_name':'precheck_execution_protocol.json'}
dump(Path('work/third_lease_weak_pilot_current.json'),current)
print(json.dumps(current))
