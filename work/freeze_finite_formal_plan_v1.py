from pathlib import Path
import datetime,hashlib,json,zipfile
r=Path(__file__).resolve().parents[1];w=r/'work';d=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_risk_preflight_20261005T1434Z')
audit=json.loads((r/'outputs/有限任务风险三GPU机制与原输入独立核验.json').read_text(encoding='utf-8'))
assert audit['status']=='THREE_REAL_FINITE_RISK_TRAIN_PREFLIGHTS_AND_FULL_RAW_REPLAY_INDEPENDENTLY_AUDITED'
pre=json.loads((d/'a/checks/preflight.json').read_text());base_sources=dict(pre['base_source_sha256'])
with zipfile.ZipFile('D:/CodexBackups/selective_flow_20261003_1105/new_p4_assets_20261005T1220Z/common_assets.zip') as z:
    oldfiles=[n for n in z.namelist() if n.startswith('frozen_v5/') and n.endswith('.py')]
    assert len(oldfiles)==4
    for n in oldfiles:base_sources[n]=hashlib.sha256(z.read(n)).hexdigest()
teacher=Path('D:/CodexBackups/selective_flow_20261003_1105/soft_teacher_cache_20261005T1246Z/teacher_cache_v1')
files=['train_finite_task_risk_v1.py','finite_task_risk_feedback_v1.py','finite_task_risk_runtime_v1.py']
plan=dict(status='FROZEN_BEFORE_NEW_FINITE_FORMAL_TRAINING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=91817,epochs=100,shared_fixed_epochs=10,
    deployment='/data/coding/finite_task_risk_v1_deployment_20261005T1450Z',base_root='/data/coding/soft_vector_research_20261005T1220Z',
    arms={'a':'fixed','b':'finite_scalar','c':'finite_vector'},batch_size=32,drop_last=True,steps_per_epoch=40,total_steps=4000,
    lr=1e-4,warmup_steps=400,weight_decay=.01,head_aux_weight=.01,separate_donor_and_head_optimizers=True,separate_clip_norm=1.,
    controller=dict(inner_steps=3,step_size=.25,trust_fraction=.25,scalar_risk_temperature_relative_squared_residual_rms=.05,beta=pre['scales']['beta'],scales=pre['scales']),
    scope='Frozen fullTRAINfit/DEVselectedA encoder/firstflow/terminal/reader/decoder; only six donor nets and matched214506residual-head parameters train,520506total. Cached deterministic coordinates with fullraw32TRAINlabel-free hook replay; not end-to-end or unbiased validation.',
    head_holdout_limits='1153TRAIN head-fit/128head-label holdout, fixed seed91990; main donor labels all1281TRAIN. Detached head feature coordinates and separate optimizers/clips prevent direct head training on heldout labels; teacher already fullTRAINfit. Not whole-pipelinecrossfit.',
    selection='Earliest strict minimum mean of two officialDEV MSE batches128/101 over100; restore fixed mode ifbest<=10, otherwise requestedmode. DEV never head target, TESTneveraccessed. Global229metrics separately.',
    budget=dict(observed20fixed_seconds={n:x['twenty_seconds'] for n,x in audit['reports'].items()},observed8row_mode_probe_seconds={n:{m:t['forward_backward_seconds'] for m,t in x['mode_checks'].items()} for n,x in audit['reports'].items()},maximum_planned_epochs=100,conservative_expected_minutes_per_arm=120,observed_fullraw_peak_bytes=audit['full_inference']['peak_allocated_bytes'],budget_not_authorization_to_stop_healthy_training=True),
    diagnostic_plan='After selected checkpoint, freeze20 conditions default/alloff/6strict final-channelocclusions/6cyclicdonorreplacements/3strategyoverrides/3receiverpairsoff. Off masks applied to final transmitted messages, not merely rawF (vector controller could regenerate a zero raw channel). Swap recomputeslabel-freecontroller. Saveofficial229residual/finiteutility/delta/trust arrays and actualsquaredrisk; no semantictruth.',
    checkpoint_contract='Addon not fullweight. Complete model combine oldA frozen encoder and new traineddonor/head/controller buffers; strict full disk reload/originalinputDEV229replay; wholeSHA/ZIPCRC permanentD and independent CPUsevenfiles+sharedphase/diagnostics. Save immediate on completion, not at lease end.',
    preflight_audit=audit,initial_tensor_sha256=audit['common']['initial_tensor_sha256'],preflight_after20_tensor_sha256=audit['common']['after20_tensor_sha256'],orders_sha256=audit['common']['orders_sha256'],scales_sha256=audit['common']['scales_sha256'],head_fit_rows_sha256=audit['common']['head_fit_rows_sha256'],
    source_sha256={n:hashlib.sha256((w/n).read_bytes()).hexdigest() for n in files},base_source_sha256=base_sources,
    teacher_file_sha256={n:hashlib.sha256((teacher/n).read_bytes()).hexdigest() for n in ['collection.json','frozen_terminal.pt','train_cache.npz','dev_cache.npz']},test_requested=False,
    interpretation='Head20pilot heldout sign.6875 equals TRAINfit majority baseline, not evidence of residual predictive advantage. Finite-risk estimator still unknown true risk, no superiority/stability/SOTA/semanticclaim.')
for filename in ['train_scales.json','head_fit_rows.npy','head_heldout_rows.npy']:
    target=w/filename
    assert not target.exists();target.write_bytes((d/'a/checks'/filename).read_bytes())
for path in [w/'finite_formal_plan_v1.json',r/'outputs/有限任务风险三臂100轮正式冻结计划.json']:
    with path.open('x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=plan['status'],sha256=hashlib.sha256((w/'finite_formal_plan_v1.json').read_bytes()).hexdigest())))
