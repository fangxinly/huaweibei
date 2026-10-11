import datetime,json,hashlib
from pathlib import Path
F=Path('outputs/parallel_gpu_20261003_1800')
cases=[
 ('seven_aligned','D:/CodexBackups/selective_flow_20261003_1105/completed_refresh_20261003T2313Z/source_supervised_seed128_4060Ti/local_checkpoint_verification.json'),
 ('full_only_aligned','D:/CodexBackups/selective_flow_20261003_1105/completed_refresh_20261003T2338Z/source_full_control_seed128_4060Ti/local_checkpoint_verification.json'),
 ('seven_train_AV_null','D:/CodexBackups/selective_flow_20261003_1105/completed_refresh_20261004T0037Z/source_train_shuffle_seed128_4060Ti/local_checkpoint_verification.json')]
rows={}
for name,path in cases:
 p=Path(path);r=json.loads(p.read_text(encoding='utf-8'))
 rows[name]={'completed_seed128_audit':str(p),'audit_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selection':r['selection'],'metrics':r['metrics_recomputed'],'checkpoint_reuse_if_same_seed_protocol_confirmed':True}
plan={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PREPARED_PENDING_USER_MEANING_OF_FIVE_FOLDS','user_request':'该训练5折的就训练5折','recommended_scope':'Replicate seven-source supervision and matched full-only objective; keep matched train-AV-null as mechanism control. These are performance replications, not installation of unreliable weak-source guards or claimed semantics.','cases':rows,'fixed_five_seed_option':{'seeds':[128,0,1,2,3],'new_training_jobs':12,'reuse_completed_seed128_weights_without_retraining':True,'train_samples':1281,'dev_samples':229,'test_samples':685,'epochs':100,'train_batch':32,'dev_batch':128,'selection':'author mean dev batchMSE','optimizer':'unchanged','test_policy':'Evaluate once after frozen model and epoch100/min-dev selection; no test-driven structure selection'},'cross_validation_option':'Requires an explicitly defined resplit protocol; never silently replace official 1281/229/685 with random five folds.','pending_existing_pilots':['full_null','text_one'],'not_currently_promoted':['V7 compression no dev benefit','V8 text-four worse MAE','Unreliable AV task guards'],'GPU_execution_started_by_this_plan':False,'future_weights_bytes_estimate':12*741868263,'local_D_free_bytes_checked':22617055232,'permanent_backup_complete':False}
with (F/'multi_repeat_candidate_plan_20261004T0058Z.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2)
print(json.dumps({'cases':list(rows),'new_jobs_if_five_seeds':12,'additional_weights_bytes':plan['future_weights_bytes_estimate'],'execution_pending_clarification':True}))

