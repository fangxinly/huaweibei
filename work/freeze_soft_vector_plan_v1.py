from pathlib import Path
import datetime,hashlib,json,numpy as np
r=Path(__file__).resolve().parents[1];w=r/'work';ch=w/'new_p4_checks';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reports={n:json.loads((ch/(n+'_full_preflight.json')).read_text()) for n in 'abc'}
for key in ['initial_tensor_sha256','orders_sha256','after20_tensor_sha256','frozen_tensor_sha_before','trainable_parameter_count','teacher_collection_sha256']:
 assert len({x[key] for x in reports.values()})==1,key
assert len({sha(ch/(n+'_after20_addon.pt')) for n in 'abc'})==1
for n,x in reports.items():
 assert x['status']=='FULL_FROZEN_TERMINAL_20_TRAIN_UPDATES_CHECKED' and x['baseline_cache_replay_max_error']==0
 assert x['main_task_head_gradients_none'] and x['auxiliary_donor_gradients_none'] and x['label_replacement_max_error']==0
 assert x['frozen_tensor_sha_before']==x['frozen_tensor_sha_after']
 assert x['peak_allocated_bytes']<2*1024**3 and x['elapsed_seconds']<60
 for source,h in x['source_sha256'].items():assert sha(w/source)==h
 assert sha(ch/(n+'_orders.npy'))==x['orders_sha256']
 orders=np.load(ch/(n+'_orders.npy'));assert orders.shape==(100,1281)
 assert np.all(np.sort(orders,axis=1)==np.arange(1281))
 assert np.array_equal(orders,np.stack([np.random.RandomState(91815+1327+e).permutation(1281) for e in range(100)]))
full=json.loads((ch/'full_inference_check_v1.json').read_text());assert full['status']=='FULL_CLASSIFIER_LABEL_FREE_VECTOR_HOOK_THREE_MODES_REPLAY_CHECKED'
assert full['source_sha256']==sha(w/'check_soft_full_inference_v1.py')
for x in full['modes'].values():assert x['full_vs_cached_prediction_max_error']<2e-5 and x['label_replacement_max_error']==0
plan={'status':'FROZEN_BEFORE_FORMAL_TRAINING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':91815,'epochs':100,'kappa':.1,'kappa_reason':'Predeclared regularization lambda=0.1*d*TRAIN_RMS^2; typical-gradient residual about1/11. No DEV search. Soft constraint deliberately retains a positive parallel component.','arms':{'a':'fixed','b':'scalar','c':'soft_projected'},'shared_fixed_epochs':10,'estimand':'Three continuations after identical10epoch fixed-feedback warmup in frozen full-TRAIN-fitted/DEV-selected A coordinates; not from-scratch or unbiased validation.','batch_size':32,'train_rows':1281,'dev_rows':229,'drop_last_each_epoch':True,'steps_per_epoch':40,'total_steps':4000,'lr':1e-4,'warmup_steps':400,'schedule':'linear warmup400; linear decay remaining3600','auxiliary_gradient_loss_weight':.01,'weight_decay':.01,'clip_norm':1.,'risk_scale':.05,'trainable_parameters':520506,'gradient_head_parameters':214506,'scope':'Only six donor networks and three matched gradient heads train. Encoder/first-flow/terminal/reader/fusion/predictor frozen eval. No old FM/cycle/pair losses, no dropout. Auxiliary TRAIN normalized-gradient MSE; main MSE and auxiliary stop-gradient paths isolated.','selection':'Minimum mean of two official DEV batch MSEs (128/101), earliest strictminimum over100 epochs; ifbest<=10 restorefixedphase. Official229 globalmetrics reportedseparately.','diagnostics':'Postcomplete freeze selected checkpoint; DEV gradient calibration/marginal shutdown/donor replacement/estimated dot versus actual risk and soft residual, same conditions allarms. These are task-head mechanism measures, not semantic shared/complement/interference truth.','checkpoint_contract':'Best addon is not full checkpoint. Combine immutable fullAencoderstate with traineddonor+gradientheads, strictload fullmodeland229replay, preservefull.pt allSHA locallyDandindependentCPUrotated node; save sharedphase separately.','budget':{'observed20_seconds':{n:x['elapsed_seconds'] for n,x in reports.items()},'observed_peak_allocated_bytes':{n:x['peak_allocated_bytes'] for n,x in reports.items()},'formal_expected_minutes_conservative':30,'max_epochs':100,'no_repeat_old_experiments':True},'test_requested':False,'source_sha256':reports['a']['source_sha256'],'common_initial_tensor_sha256':reports['a']['initial_tensor_sha256'],'orders_sha256':reports['a']['orders_sha256'],'teacher_collection_sha256':reports['a']['teacher_collection_sha256'],'lease':'Human24h; firstobservedOct5UTC12:08:17, estimatedOct6UTC12:08:17, providerexpiryunverified; immediateoncompletion saves plus20h/22h/23.5h realcaptures.','preflight_reports':reports,'full_inference_report':full}
(w/'formal_plan_v1.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
(r/'outputs/连续向量三臂正式实验冻结计划.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
(r/'outputs/连续向量真实模型机制核验.json').write_text(json.dumps({'status':'THREE_REAL_FROZEN_TERMINAL_CHECKS_AND_FULL_INFERENCE_AUDITED','reports':reports,'full_inference':full,'common_after20_file_sha256':sha(ch/'a_after20_addon.pt')},indent=2),encoding='utf-8')
print(json.dumps({'status':plan['status'],'plan_sha256':sha(w/'formal_plan_v1.json'),'full_inference':full['modes']}))
