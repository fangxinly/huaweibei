import ast,datetime,hashlib,json,pathlib,shutil,subprocess,sys

base=pathlib.Path(__file__).resolve().parents[1]
D=pathlib.Path('D:/CodexBackups/selective_flow_20261003_1105/paired_fulltrain100_complete_actual_20261007T030311Z')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
out=D/'paired_fixed_selected_DEV_five_20261007T034922Z';out.mkdir()
sources=('paired_fixed_selected_DEV_five_candidate_v1.py','sentiment_metrics_careflow_v1.py')
for n in sources:
 p=base/'work'/n;ast.parse(p.read_text(encoding='utf-8'));shutil.copyfile(p,out/n)
parents={}
for m in ('minimal_fixed_F','careflow'):
 root=D/m;t=root/'actual_training_joint_after_history_serialization_repair.json';f=root/'actual_fresh_public_selected_saved_joint.json'
 tr,fr=read(t),read(f)
 assert tr['status']=='ACTUAL_PAIRED_METHOD_FULLTRAIN_GPU_D_OTHER_CPU_CAPTURE_JOINT_PASSED_NO_FINAL_TEST'
 assert fr['status']=='ACTUAL_PAIRED_FULLTRAIN_D_ORIGINAL_OTHER_CPU_AND_FRESH_PUBLIC_SELECTED_REPLAY_JOINT_PASSED_NO_FINAL_TEST'
 assert tr['whole_checkpoint_D_SHA_CRC_passed'] and fr['all_original_model_and_array_gates_passed']
 assert fr['whole_training_joint_sha256']==sha(t)
 assert tr['CPU_history_repair_plan_sha256']=='c4f6485b43d9045f0222ddb3e2206c4a25cada3d753f3142ec724c4e1e6e92fc'
 assert not tr['final_TEST_executed'] and not fr['final_TEST_executed']
 r=root/'a/run/out/actual_stage_receipt.json';meta=read(r)['metadata']
 assert meta['best_epoch']==({'minimal_fixed_F':89,'careflow':93}[m])
 parents[m]={'method_D_root':str(root),'training_joint':{'path':str(t),'sha256':sha(t)},'fresh_joint':{'path':str(f),'sha256':sha(f)},'GPU_receipt_sha256':sha(r),'best_epoch':meta['best_epoch'],'best_state_SHA':meta['best_state_SHA'],'arrays_sha256':{n:sha(root/'a/run/out'/n) for n in ('original_DEV_selection_targets.npy','selected_best_DEV_replay.npz')}}
plan={'status':'PAIRED_ALREADY_SELECTED_DEV_FIVE_DESCRIPTIVE_PROTOCOL_FROZEN','actual_local_freeze_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'D_root':str(D),'training_plan_sha256':'8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb','methods':['minimal_fixed_F','careflow'],'source_sha256':{n:sha(out/n) for n in sources},'parents':parents,'new_model_or_forward_or_final_TEST_enabled':False,'all_five_same_previously_selected_checkpoint':True,'no_readout_seed_feature_or_threshold_choice':True,'DEV_already_explored_and_used_for_selection_not_independent_confirmation':True,'historical_old_baseline_TEST_access_not_erased':True,'author_metric_rules':'Acc7 clip-round all; y==0 excluded Acc2/pred>=0 and support-weighted F1; unmodified all-row MAE/Pearson; FP64 aggregation','overall_goal_remains_incomplete':True}
pp=out/'plan.json';pp.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'D':str(out),'plan_SHA':sha(pp),'status':plan['status'],'no_arrays_decoded_while_freezing':True}))
