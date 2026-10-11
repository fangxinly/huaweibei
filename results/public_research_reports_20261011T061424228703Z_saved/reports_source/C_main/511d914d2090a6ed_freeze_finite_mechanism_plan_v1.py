from pathlib import Path
import datetime, hashlib, json
r=Path(__file__).resolve().parents[1];w=r/'work'
files=['finite_task_risk_feedback_v1.py','finite_task_risk_runtime_v1.py','check_finite_task_risk_v1.py']
plan=dict(status='MECHANISM_PLAN_FROZEN_BEFORE_GPU_PREFLIGHT_NOT_FORMAL_100_EPOCH_PLAN',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_seed=91817,
    requested_arms=['fixed','finite_scalar','finite_vector'],base_root='/data/coding/soft_vector_research_20261005T1220Z',
    preflight_root='/data/coding/finite_task_risk_preflight_20261005T1428Z',
    task_risk='Exact finite square-loss difference2rho*delta+delta^2; conditional residual estimator is an assumption, not known risk.',
    capacity='Same six donor nets and three601->100->100 gradient head networks, aggregate300head outputs to one normalized residual;520506totaltrainable/214506heads.',
    supervision='Residual head fit1153TRAIN rows only, fixed seeded128head-label holdout. Main donor loss all1281TRAIN. Frozen reference encoder previously fullTRAINfit/DEVselected, not pipelinecrossfit.',
    split_seed=91817+173,head_aux_weight=.01,
    scales='Residual RMS and initial rawdonor channel RMS from designated1153TRAIN rows only; beta=residual_RMS^2/(2*median reference normalized-message sensitivity norm2), no DEV fitting.',
    calibration_reference='.125^2*sum_six(sF_i^2*||J_receiver||^2); J label-free cached TRAIN only used for predeclared beta scale, not controllerinput.',
    scalar='Finite marginal utility against fixed otherchannels, sigmoid(U/(.05*TRAINresidualRMS^2)); gradients flow through exact frozen terminal and donor messages, rho stopped.',
    vector='Three projectedGD steps eta.25; objective.5||u-u0||^2+beta*(2rhohat*delta+delta^2)/TRAINresidualRMS^2; normalized raw message coordinates; trust radius.25||u0||; no convex/globaloptimal/true-risk guarantee.',
    gradients='Main rho head stopped; auxiliary donor/frozen coordinates stopped. Separate head and donor AdamW optimizers and clips1 prevent global clipping coupling from head-label holdout.',
    shared_phase='Future formal100 arms must identical first10fixed then90strategies; not started until formalplan and fullraw32TRAINreplay and these checks pass.',
    preflight_budget=dict(fixed_updates=20,batch_size=32,controller_probe_rows=8,double_difference_rows=2,maximum_intended_peak_bytes=4*1024**3,maximum_intended_seconds_per_node=600,formal_epochs_not_started=True),
    required_checks=['fresh UUID/empty compute/space/actual argv','TR scales/split/order/init/SHA agreement','labelreplace/noGrad inference equality','main/aux/frozen gradient isolation','20fixed updates and frozen state invariant','nonzero double directional difference of real unrolled decoder','exactTRAIN squareloss identity','zero-message/zero-residual reference','trust bound','TRAINrawfullclassifier3modes replay before formaltraining'],
    source_sha256={n:hashlib.sha256((w/n).read_bytes()).hexdigest() for n in files},
    dev_requested=False,test_requested=False,limits='Only mechanical feasibility, not efficacy or semantics. No old training repeated; public immutable sources preserved.')
for p in [w/'finite_mechanism_plan_v1.json',r/'outputs/有限任务风险真实机制预检冻结计划.json']:
    with p.open('x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
print(plan['status'])
