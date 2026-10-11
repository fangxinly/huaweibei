import ast,hashlib,json,os,shutil,sys,zipfile
from pathlib import Path
import numpy as np
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
ws=Path(__file__).resolve().parent.parent;stamp=sys.argv[1];clock=sys.argv[2]
candidate=ws/'work/official_anchored_upgrade_candidate';source=ws/'work'/('official_anchored_upgrade_'+stamp)
d=Path('D:/CodexBackups/selective_flow_20261003_1105')/('official_anchored_upgrade_actual_'+stamp)
source.mkdir();d.mkdir();cbackup=ws/'work'/('official_anchored_upgrade_large_backup_'+stamp);cbackup.mkdir()
space={'D':shutil.disk_usage(d)._asdict(),'C':shutil.disk_usage(cbackup)._asdict()};assert space['D']['free']>=2100000000 and space['C']['free']>=1450000000,space
for f in candidate.iterdir():
    if f.is_file():
        if f.suffix=='.py':compile(f.read_text(encoding='utf-8-sig'),str(f),'exec')
        shutil.copy2(f,source/f.name);assert sha(f)==sha(source/f.name)
text=(source/'train_official.py').read_text();tree=ast.parse(text)
for n in ast.walk(tree):
    if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='data' and isinstance(n.slice,ast.Constant):assert n.slice.value in ('train','dev')
old=read(ws/'work/anchored_VAL_TEST_eval_20261008T051335Z/official_roles_protocol.json')
split=read(ws/'work/weak_retention_train40_20261007T145201Z/bundle/split.json');ids=split['canonical_row_ids'];assert len(ids)==2195
official={'train':ids[:1281],'dev':ids[1281:1510]}
identity=hashlib.sha256(json.dumps(official,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest();assert identity=='9462739ff690facc2c10eec2e72e79a02c1b7c0a6ee346e893716894ed690747'
for a,b in (('train','dev'),):assert not set(s.split('[')[0] for s in official[a]) & set(s.split('[')[0] for s in official[b])
orders=source/'prospective_common_TRAIN_position_orders_seed128_100.npy';assert sha(orders)=='d3cfd5621cc7a51275c7847410519148f62443e5e2b07b356286fead34d000e4';o=np.load(orders,allow_pickle=False);assert o.shape==(100,1281) and all(np.array_equal(np.sort(v),np.arange(1281)) for v in o)
devfile=ws/'outputs/正式双方fullTRAIN100固定best一次DEV五项实际结果.json';testfile=ws/'outputs/正式双方TEST685一次五项实际结果.json';dev=read(devfile);test=read(testfile)
assert dev['methods']['careflow']['best_epoch']==93 and test['natural_exit_code']==0
baseline={'VAL':dev['methods']['careflow']['fixed_DEV_five'],'TEST':test['metrics']['careflow']}
assert abs(baseline['VAL']['MAE']-.604523826276)<1e-10 and abs(baseline['TEST']['MAE']-.619535293923368)<1e-12
ref={'DEV_source':str(devfile),'DEV_source_SHA':sha(devfile),'TEST_source':str(testfile),'TEST_source_SHA':sha(testfile),'original_DEV_result_SHA':dev['fixed_DEV_result_SHA'],'original_TEST_score_receipt_SHA':test['score_receipt_sha256'],'selected_state_SHA':dev['methods']['careflow']['best_state_SHA'],'selected_epoch':93,'original_training_plan_SHA':dev['training_plan_SHA'],'complete_D_CPU_qualification':dev['methods']['careflow']['training_joint_SHA'],'whole_original_model_SHA':dev['methods']['careflow']['whole_D_files']['selected_best_full']['sha256']}
storage={'actualclock_UTC':clock,'actual_pid':os.getpid(),'fullargv':[sys.executable]+sys.argv,'physical_volumes':space,'D_priority_root':str(d),'C_remaining_part_root':str(cbackup),'part_bytes':1000000000,'parts0_1_on_D_parts2_3_on_C':True,'maximum_complete_archive_bytes':3300000000,'whole_archive_verified_without_local_full_extraction':True,'other_node_B_complete_restore_required':True,'no_old_file_deletion':True}
write(d/'physical_local_capacity_and_split_preservation.json',storage)
p={'status':'OFFICIAL_ANCHORED_UPGRADE_COMMON_BUDGET_FROZEN','actualclock_UTC':clock,'allowed_stages':['native'],'source_sha256':{f.name:sha(f) for f in source.iterdir() if f.is_file()},'asset_sha256':old['asset_sha256'],'runtime_versions':old['runtime_versions'],'GPU_UUID':old['GPU_UUID'],'conservative_lease_end_UTC':'2026-10-08T14:00:00+00:00','trusted_human_lease_provenance':'Human supplied this third batch then answered 24小时之后; conservative endpoint inherited; not platform verification','remote_free_floor_bytes':10000000000,'stage_budget_seconds':{'native':120,'train':7200,'audit':900,'infer':600,'score':180},'training_budget_seconds':7200,'GPU_peak_ceiling_bytes':6*1024**3,'checkpoint_bytes_ceiling':3200000000,'complete_archive_bytes_ceiling':3300000000,'archive_part_bytes':1000000000,'local_preservation':storage,'task_seed':128,'orders_seed':128,'epochs':100,'updates':4000,'batch':32,'drop_last':True,'dev_batch':128,'DEV_selection':'author_batch_MSE_strict_earliest','official_train_dev_IDs':official,'official_role_IDs':{'VAL':ids[1281:1510],'TEST':ids[1510:]},'official_ID_identity_SHA':identity,'orders_file':orders.name,'orders_SHA':sha(orders),'method':'original_AnchoredFlow_donor_minus_zero_low_rank_message_joint_clean_training','method_objective':'Huber(delta=1)+.01*context_square','parameter_learning_rates':{'original_core':1e-5,'scalar_gain':.001,'six_low_rank_messages':.001},'weight_decay':{'ordinary':.01,'bias_LayerNorm':0.,'gain':0.,'message':.01},'warmup_steps':400,'scheduler_total_steps':4000,'native_CPU_D_qualification':None,'train_once_token':'/data/coding/official_anchored_upgrade_'+stamp+'/train_once.lock','score_once_token':'/data/coding/official_anchored_upgrade_'+stamp+'/score_once.lock','fixed_CaReFlow_five':baseline,'CaReFlow_baseline_original_reference':ref,'comparison_scope':'same official data/pretrained assets/seed/orders/100epoch4000updates/VAL selection/metric rules; per-method loss/parameter LR differs prospectively; measured runtime and capacity reported','old_TEST_history_disclosed':True,'no_old_task_checkpoint_or_merged_fitted_state_allowed':True,'whole_goal_complete':False}
plan=source/'official_native_protocol.json';write(plan,p);shutil.copy2(plan,d/plan.name)
zipname=d/'frozen_official_upgrade_source.zip'
with zipfile.ZipFile(zipname,'x',zipfile.ZIP_DEFLATED,6) as z:
    for f in source.iterdir():z.write(f,f.name)
with zipfile.ZipFile(zipname) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
pointer={'source':str(source),'D':str(d),'C_part_backup':str(cbackup),'ZIP':str(zipname),'ZIP_SHA':sha(zipname),'native_plan_SHA':sha(plan),'actualclock_UTC':clock,'role_overlap_with_training':{'VAL':0,'TEST':0},'comparison_baseline':ref,'real_new_training_started':False}
pointerpath=ws/'work/official_upgrade_pointer.json'
if pointerpath.exists():
    prior=read(pointerpath);write(d/'previous_immutable_source_pointer.json',prior)
    pointerpath.write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding='utf8')
else:write(pointerpath,pointer)
write(d/'source_freeze_receipt.json',pointer);print(json.dumps(pointer,ensure_ascii=False))
