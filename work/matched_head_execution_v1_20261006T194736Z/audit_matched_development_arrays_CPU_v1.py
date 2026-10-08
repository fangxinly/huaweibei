"""Future original201 zero-label + nine frozen predictions audit, CPU arrays only."""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from matched_development_independent_math_v1 import readouts
p=argparse.ArgumentParser()
for n in ['run','head-state','out']:p.add_argument('--'+n,required=True)
a=p.parse_args();root=Path(a.run)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
plan=read(root/'source/matched_head_execution_plan.json');assert plan['status']=='COMPLETE_MATCHED_HEAD_EXECUTION_PROTOCOL_FROZEN'
for n,h in plan['source_and_role_sha256'].items():assert sha(root/'source'/n)==h
r=read(root/'out/actual_candidate_collection_receipt.json');e=read(root/'natural_exit.json');m=read(a.head_state);hs=sha(a.head_state)
assert e['exit_code']==0 and e['natural_wait_verified'] and e['child_pid']==r['pid'] and e['child_full_argv'][1:]==r['argv'] and e['original_receipt_sha256']==sha(root/'out/actual_candidate_collection_receipt.json')
assert r['source_sha256']==plan['source_and_role_sha256']['fixed_head_development_collector_v1.py'] and r['plan_sha256']==sha(root/'source/matched_head_execution_plan.json')
assert not r['task_labels_read'] and not r['head_fit_or_optimizer_updates'] and r['headEVAL201_inputs_used'] and not r['headEVAL201_labels_used'] and not r['actual_performance_or_gain_measured'] and r['rng_unchanged']
assert hs==r['head_state_sha256'] and m['plan_sha256']==r['plan_sha256']
path=root/'out/original_headEVAL201_label_free.npz';pred=root/'out/frozen_headEVAL201_predictions.npz';assert sha(path)==r['original_array_sha256'] and sha(pred)==r['frozen_predictions_sha256']
rows=np.load(root/'source/head_development_eval_0.npy',allow_pickle=False);mapping=read(root/'source/train_row_video_mapping.json')
with np.load(path,allow_pickle=False) as z:
 assert len(rows)==201 and np.array_equal(z['row_ids'],rows) and z['video_ids'].tolist()==[mapping[int(i)]['video_id'] for i in rows] and z['segment_ids'].tolist()==[mapping[int(i)]['segment_id'] for i in rows] and len(np.unique(z['video_ids']))==9
 f=z['pF'].astype(np.float64);c=z['pC'].astype(np.float64);d=c-f
 assert np.isfinite(f).all() and np.isfinite(c).all() and np.array_equal(z['pF'],z['pF_restored']) and np.array_equal(z['pC'],z['pC_repeat']) and np.array_equal(z['delta'],d) and np.array_equal(z['W_raw'],4*d*d)
 outputs=readouts(m['model'],f,c)
with np.load(pred,allow_pickle=False) as z:
 assert set(z.files)==set(outputs)|{'row_ids','head_state_sha256'} and np.array_equal(z['row_ids'],rows) and str(z['head_state_sha256'].item())==hs
 assert all(np.array_equal(z[n],v) for n,v in outputs.items())
freeze=read(root/'out/nine_prediction_freeze.json');assert sha(root/'out/nine_prediction_freeze.json')==r['nine_prediction_freeze_sha256'] and freeze['frozen_predictions_sha256']==r['frozen_predictions_sha256'] and freeze['head_state_sha256']==hs and not freeze['201_task_labels_indexed']
assert datetime.datetime.fromisoformat(m['actual_frozen_utc'])<datetime.datetime.fromisoformat(freeze['actual_utc'])
assert r['forward_calls']==30 and r['batch_sizes']==[32]*6+[9] and len(r['semantic_batch_checks'])==7
for b in r['semantic_batch_checks']:assert max(b['first_euler_errors'])==0 and b['F_repeat_error']<=1e-6 and b['C_repeat_error']<=1e-6
for b in r['dummy_label_checks']:assert b['dummy0_vs7_error']==0 and not b['labels_read']
assert all(not j['labels_read'] for j in r['input_access_journal'])
result={'status':'ACTUAL_HEADEVAL201_ORIGINAL_LABEL_FREE_ARRAY_NINE_PREDICTION_CPU_AUDIT_PASSED_NOT_SCORES','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),'original_collection_receipt_sha256':sha(root/'out/actual_candidate_collection_receipt.json'),'plan_sha256':r['plan_sha256'],'head_state_sha256':hs,'original_label_free_array_sha256':r['original_array_sha256'],'frozen_predictions_sha256':r['frozen_predictions_sha256'],'all_nine_independent_replay_error':0,'official_task_labels_or_assets_accessed':False,'GPU_or_CPU_model_forward':False}
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
