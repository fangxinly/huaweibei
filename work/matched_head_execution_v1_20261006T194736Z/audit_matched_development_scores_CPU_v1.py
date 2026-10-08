"""Future independent arithmetic audit of the original once201 scored batch."""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from matched_development_independent_math_v1 import readouts,independent_metrics,compare_metrics
p=argparse.ArgumentParser()
for n in ['run','head-state','out']:p.add_argument('--'+n,required=True)
a=p.parse_args();root=Path(a.run);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
plan=read(root/'source/matched_head_execution_plan.json');assert plan['status']=='COMPLETE_MATCHED_HEAD_EXECUTION_PROTOCOL_FROZEN'
for n,h in plan['source_and_role_sha256'].items():assert sha(root/'source'/n)==h
r=read(root/'out/actual_matched_head_evaluation_receipt.json');e=read(root/'natural_exit.json');results=read(root/'out/all_nine_development_results.json');hs=sha(a.head_state)
assert e['exit_code']==0 and e['natural_wait_verified'] and e['child_pid']==r['pid'] and e['child_full_argv'][1:]==r['argv'] and e['original_receipt_sha256']==sha(root/'out/actual_matched_head_evaluation_receipt.json')
assert sha(root/'out/all_nine_development_results.json')==r['all_nine_results_sha256'] and hs==r['head_state_sha256'] and not r['GPU_used'] and not r['head_fit_or_reference_update'] and not r['other_task_label_roles_indexed'] and not r['output_selection']
array=root/'out/original_headEVAL201_scored_all_nine.npz';assert sha(array)==r['original_scored_arrays_sha256'];rows=np.load(root/'source/head_development_eval_0.npy',allow_pickle=False);mapping=read(root/'source/train_row_video_mapping.json');maximum=0.;riskerr=0.
with np.load(array,allow_pickle=False) as z:
 v=z['video_ids'];y=z['labels'];f=z['F'];c=z['C'];assert np.array_equal(z['row_ids'],rows) and len(rows)==201 and len(np.unique(v))==9 and v.tolist()==[mapping[int(i)]['video_id'] for i in rows] and z['segment_ids'].tolist()==[mapping[int(i)]['segment_id'] for i in rows]
 assert str(z['head_state_sha256'].item())==hs and np.isfinite(y).all() and len(r['task_label_journal'])==1 and r['task_label_journal'][0]['row_ids']==rows.tolist() and r['task_label_journal'][0]['role']=='headEVAL201'
 outputs=readouts(read(a.head_state)['model'],f,c);assert set(results['outputs'])==set(outputs)
 for n,pred in outputs.items():
  assert np.array_equal(z[n],pred);item=results['outputs'][n];maximum=max(maximum,compare_metrics(item['five_metrics_same_prediction'],independent_metrics(pred,y)))
  values=[];qvalues=[]
  for vid in sorted(np.unique(v)):
   mask=v==vid;mse=float(np.dot(pred[mask]-y[mask],pred[mask]-y[mask])/np.sum(mask));q=float(np.mean((pred[mask]-f[mask])**2-2*(pred[mask]-f[mask])*(y[mask]-f[mask])))
   riskerr=max(riskerr,abs(mse-item['per_video'][str(vid)]['MSE']),abs(q-item['per_video'][str(vid)]['paired_Q_vs_F']));values.append(mse);qvalues.append(q)
  riskerr=max(riskerr,abs(float(np.mean(values))-item['video_equal_MSE']),abs(float(np.mean(qvalues))-item['video_equal_paired_Q_vs_F']))
 assert riskerr<1e-12
assert r['source_sha256']==plan['source_and_role_sha256']['score_frozen_matched_heads201_v1.py'] and r['plan_sha256']==sha(root/'source/matched_head_execution_plan.json') and not results['readout_or_model_selection']
result={'status':'ACTUAL_HEADEVAL201_ALL_NINE_ORIGINAL_METRICS_RISK_CPU_AUDIT_PASSED_DEVELOPMENT_ONLY','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),'original_score_receipt_sha256':sha(root/'out/actual_matched_head_evaluation_receipt.json'),'head_state_sha256':hs,'original_scored_array_sha256':r['original_scored_arrays_sha256'],'independent_five_metric_max_error':maximum,'independent_video_risk_max_error':riskerr,'additional_official_label_asset_access':False,'GPU_or_CPU_model_forward':False,'CaReFlow_benchmark_or_overall_research_complete':False}
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
