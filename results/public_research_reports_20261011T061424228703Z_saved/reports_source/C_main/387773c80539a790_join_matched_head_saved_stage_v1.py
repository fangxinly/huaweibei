"""Future local joint of original A/B captures already saved completely on D."""
import argparse,datetime,hashlib,json,sys,zipfile
from pathlib import Path
p=argparse.ArgumentParser()
for n in ['D-root','stage','head-state','parent-joint','out']:p.add_argument('--'+n,required=True)
a=p.parse_args();root=Path(a.D_root)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
assert root.resolve().drive.upper()=='D:'
verified={}
for role in ('a','b'):
 base=root/role;capture=read(base/'capture_receipt.json');ce=read(base/'capture_actual_exit.json');manifest=read(base/'member_manifest.json')
 assert ce['exit_code']==0 and ce['natural_wait_verified'] and ce['child_pid']==capture['pid'] and ce['child_full_argv'][1:]==capture['argv'] and ce['original_receipt_sha256']==sha(base/'capture_receipt.json')
 assert sha(base/'snapshot.zip')==capture['snapshot_sha256'] and capture['members']==len(manifest['small_members'])+1
 with zipfile.ZipFile(base/'snapshot.zip') as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==capture['members'] and z.read('member_manifest.json')==(base/'member_manifest.json').read_bytes()
  for n,item in manifest['small_members'].items():assert hashlib.sha256(z.read(n)).hexdigest()==item['sha256']==sha(base/n)
 assert manifest['original_receipt_sha256']==capture['original_receipt_sha256'] and manifest['original_exit_sha256']==capture['original_exit_sha256']
 verified[role]={'receipt_sha256':sha(base/'capture_receipt.json'),'exit_sha256':sha(base/'capture_actual_exit.json'),'snapshot_sha256':capture['snapshot_sha256'],'members':capture['members'],'actual_capture_utc':capture['actual_utc'],'large_references_not_downloads':capture['large_references']}
paths={'fit':'out/actual_matched_head_fit_receipt.json','predict':'out/actual_candidate_collection_receipt.json','score':'out/actual_matched_head_evaluation_receipt.json'}
statuses={'fit':'ACTUAL_HEADFIT232_GPU_PARENT_D_OTHER_CPU_FIT_SOURCE_STATE_CAPTURE_JOINT_PASSED','predict':'ACTUAL_HEADEVAL201_LABEL_FREE_NINE_PREDICTIONS_D_OTHER_CPU_JOINT_PASSED','score':'ACTUAL_HEADEVAL201_ALL_NINE_SCORES_D_OTHER_CPU_JOINT_PASSED_DEVELOPMENT_ONLY'}
assert a.stage in paths
original=root/'a/run'/paths[a.stage];r=read(original);cpu=read(root/'b/run/cpu_original_receipt.json');plan=read(root/'a/run/source/matched_head_execution_plan.json');parent=read(a.parent_joint);hs=sha(a.head_state)
assert plan['status']=='COMPLETE_MATCHED_HEAD_EXECUTION_PROTOCOL_FROZEN' and r['plan_sha256']==sha(root/'a/run/source/matched_head_execution_plan.json')
assert r['head_state_sha256']==hs and read(a.head_state)['plan_sha256']==r['plan_sha256']
cpu_key={'fit':'original_fit_receipt_sha256','predict':'original_collection_receipt_sha256','score':'original_score_receipt_sha256'}[a.stage]
assert cpu[cpu_key]==sha(original) and cpu['head_state_sha256']==hs
expected_cpu={'fit':'ACTUAL_MATCHED_HEADFIT232_ORIGINAL_FULL_STATE_ARRAY_OBJECTIVE_CPU_AUDIT_PASSED_NOT201_EVAL','predict':'ACTUAL_HEADEVAL201_ORIGINAL_LABEL_FREE_ARRAY_NINE_PREDICTION_CPU_AUDIT_PASSED_NOT_SCORES','score':'ACTUAL_HEADEVAL201_ALL_NINE_ORIGINAL_METRICS_RISK_CPU_AUDIT_PASSED_DEVELOPMENT_ONLY'}
assert cpu['status']==expected_cpu[a.stage]
assert verified['a']['receipt_sha256'] and verified['b']['receipt_sha256']
if a.stage=='fit':
 assert sha(a.parent_joint)==r['original_candidate_joint_sha256'] and parent['status']=='ACTUAL_HEADFIT232_SINGLE_CANDIDATE_GPU_D_B_ARRAY_SOURCE_REFERENCE_JOINT_PASSED'
 assert sha(root/'a/run/out/head_state.json')==hs and sha(root/'a/run/out/original_headFIT232_supervised.npz')==r['original_supervised_array_sha256']
elif a.stage=='predict':
 assert sha(a.parent_joint)==r['head_fit_joint_sha256'] and parent['status']==statuses['fit']
 assert sha(root/'a/run/out/frozen_headEVAL201_predictions.npz')==r['frozen_predictions_sha256'] and sha(root/'a/run/out/original_headEVAL201_label_free.npz')==r['original_array_sha256']
else:
 assert sha(a.parent_joint)==r['collection_joint_sha256'] and parent['status']==statuses['predict']
 assert sha(root/'a/run/out/original_headEVAL201_scored_all_nine.npz')==r['original_scored_arrays_sha256'] and sha(root/'a/run/out/all_nine_development_results.json')==r['all_nine_results_sha256']
result={'status':statuses[a.stage],'actual_local_joint_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':r['plan_sha256'],'head_state_sha256':hs,'parent_joint_sha256':sha(a.parent_joint),'original_stage_receipt_sha256':sha(original),'original_other_CPU_receipt_sha256':sha(root/'b/run/cpu_original_receipt.json'),'captures':verified,'whole_small_originals_D_downloaded_and_CRC_SHA_unique_passed':True,'large_references_are_not_new_weight_downloads':True,'CPU_model_forward':False,'formal_CaReFlow_benchmark_or_overall_research_complete':False}
if a.stage=='predict':result.update(frozen_predictions_sha256=r['frozen_predictions_sha256'],original_label_free_array_sha256=r['original_array_sha256'])
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
