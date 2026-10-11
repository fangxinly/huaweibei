"""Future source-pinned headFIT232-only CPU fitting stage. No evaluation access."""
import argparse,datetime,hashlib,json,os,pickle,sys,time
from pathlib import Path
import numpy as np
from matched_fixed_residual_heads_v2 import HeadLabels,design,fit,predict

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def utc():return datetime.datetime.now(datetime.timezone.utc)

def run(a):
 start=time.perf_counter();bundle=Path(a.bundle);plan=read(bundle/'matched_head_execution_plan.json')
 if plan['status']!='COMPLETE_MATCHED_HEAD_EXECUTION_PROTOCOL_FROZEN':raise PermissionError('FULL_EXECUTION_PROTOCOL_NOT_READY')
 for n,h in plan['source_and_role_sha256'].items():
  if sha(bundle/n)!=h:raise ValueError('FULL_SOURCE_AND_ROLE_SHA:'+n)
 parent=read(a.parent_joint)
 if sha(a.parent_joint)!=plan['candidate_joint_sha256'] or parent['status']!='ACTUAL_HEADFIT232_SINGLE_CANDIDATE_GPU_D_B_ARRAY_SOURCE_REFERENCE_JOINT_PASSED':raise PermissionError('ACTUAL_GPU_D_B_CANDIDATE_JOINT_REQUIRED')
 e=read(a.evidence)
 if e['scope']!='MATCHED_HEADFIT232_CPU_ONLY_V1' or e['human_lease_source']!='DIRECT_HUMAN_NEW_P4_24H_20261006':raise PermissionError('WRONG_AUTHORIZED_SCOPE')
 # Evidence is a record of separately verified human assets and real queries;
 # JSON alone is not evidence of human authorization or lease ownership.
 if not 0<=(utc()-datetime.datetime.fromisoformat(e['actual_query_utc'])).total_seconds()<=300:raise PermissionError('ACTUAL_QUERY_FIVE_MINUTE_FRESH')
 if (datetime.datetime.fromisoformat(e['conservative_lease_end_utc'])-utc()).total_seconds()<1200+7200:raise PermissionError('EXECUTION_AND_TWO_HOUR_SAVE_RESERVE')
 if e['remote_free_bytes']<4*1024**3 or e['D_free_bytes']<6*1024**3:raise PermissionError('SAVE_SPACE')
 if e['plan_sha256']!=sha(bundle/'matched_head_execution_plan.json'):raise PermissionError('ACTUAL_PINNED_PLAN_QUERY')
 if not isinstance(e['full_python_argv'],list) or e['gpu_uuid'] not in plan['authorized_GPU_UUIDs']:raise PermissionError('ACTUAL_IDENTITY_AND_FULLARGV')
 inputs=Path(a.candidate_arrays)
 if sha(inputs)!=plan['original_headFIT232_array_sha256'] or sha(inputs)!=parent['original_array_sha256']:raise ValueError('ACTUAL_FIT232_ARRAY_SHA')
 rows=np.load(bundle/'head_development_fit_0.npy',allow_pickle=False);evalrows=np.load(bundle/'head_development_eval_0.npy',allow_pickle=False)
 mp=read(bundle/'train_row_video_mapping.json')
 with np.load(inputs,allow_pickle=False) as z:
  if not np.array_equal(z['row_ids'],rows) or not np.isfinite(z['pF']).all() or not np.isfinite(z['pC']).all():raise ValueError('FROZEN_INPUT_ROWS_OR_NONFINITE')
  f=z['pF'].astype(np.float64);c=z['pC'].astype(np.float64);d=z['delta'].copy();v=z['video_ids'].copy();segments=z['segment_ids'].copy()
  if not np.array_equal(z['pC'].astype(np.float64)-f,d) or not np.array_equal(4*d*d,z['W_raw']):raise ValueError('ORIGINAL_DELTA_W_FORMULA')
  if not np.array_equal(z['pF'],z['pF_restored']) or not np.array_equal(z['pC'],z['pC_repeat']):raise ValueError('ORIGINAL_BOTH_PATH_EXACT_REPLAY')
 if v.tolist()!=[mp[int(i)]['video_id'] for i in rows] or len(np.unique(v))!=9:raise ValueError('FROZEN_VIDEO_ROLE')
 if set(v.tolist()) & {mp[int(i)]['video_id'] for i in evalrows}:raise PermissionError('HEAD_ROLE_VIDEO_OVERLAP')
 if np.all(d==0):raise ValueError('EXACT_ZERO_DELTA_STOP_BEFORE_NEW_LABELS')
 data=Path(a.asset_base)/'assets/mosi.pkl'
 if sha(data)!=plan['official_mosi_pickle_sha256']:raise ValueError('OFFICIAL_DATA_ASSET_SHA')
 out=Path(a.out)
 if out.exists():raise FileExistsError('NEW_UNIQUE_FIT_OUTPUT_NO_RERUN')
 out.mkdir(parents=True,exist_ok=False)
 common=design(f,d,v)
 np.savez_compressed(out/'before_label_common_design.npz',row_ids=rows,video_ids=v,**common)
 design_sha=sha(out/'before_label_common_design.npz')
 write(out/'before_label_gate.json',{'actual_utc':utc().isoformat(),'plan_sha256':sha(bundle/'matched_head_execution_plan.json'),'source_sha256':sha(__file__),'parent_joint_sha256':sha(a.parent_joint),'array_sha256':sha(inputs),'common_design_sha256':design_sha,'permitted_rows':rows.tolist(),'roles_verified_before_labels':True})
 # Trusted full pickle deserialization, TRAIN entry only; no other entries
 # indexed. Only examples[i][1] for the exact permitted232 rows is accessed.
 with data.open('rb') as stream:container=pickle.load(stream)
 examples=container['train'];del container
 guard=HeadLabels(examples,rows,evalrows);labels=guard.get_fit(rows)
 if not np.isfinite(labels).all():raise ValueError('NONFINITE_FIT_LABEL_NO_ROW_DROPPING')
 model=fit(f,d,labels,v,common);outputs=predict(model,f,d,c)
 stored={k:(val.tolist() if isinstance(val,np.ndarray) else val) for k,val in model.items()}
 write(out/'head_state.json',{'scope':'FIXED_HEADFIT232_ONLY_U_W_AND_CONSTANT','actual_frozen_utc':utc().isoformat(),'plan_sha256':sha(bundle/'matched_head_execution_plan.json'),'original_input_array_sha256':sha(inputs),'model':stored,'fit_row_ids':rows.tolist(),'fit_video_ids':v.tolist(),'headEVAL201_inputs_or_labels_used':False})
 hs=sha(out/'head_state.json');reloaded=read(out/'head_state.json')['model'];replay=predict(reloaded,f,d,c)
 maxerr=max(float(np.max(np.abs(replay[n]-p))) for n,p in outputs.items())
 if maxerr!=0:raise ValueError('OWN_FULL_HEAD_STATE_DISK_REPLAY_NOT_EXACT')
 np.savez_compressed(out/'original_headFIT232_supervised.npz',row_ids=rows,video_ids=v,segment_ids=segments,pF=f,pC=c,delta=d,labels=labels,residual=labels-f,head_state_sha256=np.asarray(hs),**{n:p for n,p in outputs.items() if n not in ['F','C']})
 X=np.column_stack([np.ones(len(f)),np.divide(f-model['mean'][0],model['std'][0],out=np.zeros_like(f),where=model['std'][0]>0),np.divide(d-model['mean'][1],model['std'][1],out=np.zeros_like(d),where=model['std'][1]>0)])
 objective={n:float(w@((X@model[n]-(labels-f))**2)+.01*np.dot(model[n][1:],model[n][1:])) for n,w in [('U',model['normalized_U_weights']),('W',model['normalized_W_weights'])]}
 elapsed=time.perf_counter()-start
 if elapsed>1200:raise RuntimeError('FIT_STAGE_BUDGET')
 r={'status':'ACTUAL_HEADFIT232_MATCHED_U_W_CPU_FIT_COMPLETE_NOT201_EVALUATION','actual_utc':utc().isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),'plan_sha256':sha(bundle/'matched_head_execution_plan.json'),'original_candidate_joint_sha256':sha(a.parent_joint),'original_input_array_sha256':sha(inputs),'original_supervised_array_sha256':sha(out/'original_headFIT232_supervised.npz'),'head_state_sha256':hs,'before_label_common_design_sha256':design_sha,'task_label_journal':guard.journal,'rows':232,'videos':9,'objective_FIT_only_not_generalization':objective,'objective_diagnostics_FIT_only':model['fit_objective_diagnostics'],'whole_head_disk_replay_max_error':maxerr,'elapsed_seconds':elapsed,'GPU_used':False,'reference_model_or_statistics_modified':False,'headEVAL201_inputs_or_labels_used':False,'all_other_task_label_roles_used':False,'new_development_or_benchmark_scores':False}
 write(out/'actual_matched_head_fit_receipt.json',r);print(json.dumps(r),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--parent-joint',required=True);p.add_argument('--candidate-arrays',required=True);p.add_argument('--asset-base',required=True);p.add_argument('--evidence',required=True);p.add_argument('--out',required=True);run(p.parse_args())
