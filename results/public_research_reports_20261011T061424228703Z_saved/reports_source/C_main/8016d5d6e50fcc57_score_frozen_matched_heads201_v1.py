"""Future one locked development evaluation, all nine vectors, no selection."""
import argparse,datetime,hashlib,json,os,pickle,sys,time
from pathlib import Path
import numpy as np
from matched_fixed_residual_heads_v2 import HeadLabels,predict,video_equal_weights
from sentiment_metrics_careflow_v1 import metrics,SEMANTICS

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def run(a):
 start=time.perf_counter();bundle=Path(a.bundle);plan=read(bundle/'matched_head_execution_plan.json')
 if plan['status']!='COMPLETE_MATCHED_HEAD_EXECUTION_PROTOCOL_FROZEN':raise PermissionError('FULL_EXECUTION_PROTOCOL_NOT_READY')
 evidence=read(a.evidence);now=datetime.datetime.now(datetime.timezone.utc)
 if evidence['scope']!='MATCHED_HEADEVAL201_SCORE_ONLY_V1' or not evidence['human_provenance_verified'] or evidence['lease_source']!='DIRECT_HUMAN_NEW_P4_24H_20261006':raise PermissionError('ORIGINAL_HUMAN_LEASE_AND_SCORE_SCOPE')
 if not 0<=(now-datetime.datetime.fromisoformat(evidence['actual_query_utc'])).total_seconds()<=300 or (datetime.datetime.fromisoformat(evidence['lease_end_utc'])-now).total_seconds()<8400:raise PermissionError('FRESH_QUERY_EXECUTION_AND_TWO_HOUR_SAVE')
 if evidence['gpu_uuid'] not in plan['authorized_GPU_UUIDs'] or evidence['compute_processes']!=[] or not isinstance(evidence['python_full_argv'],list) or not evidence['asset_and_source_actual_SHA_verified'] or evidence['plan_sha256']!=sha(bundle/'matched_head_execution_plan.json') or evidence['remote_free_bytes']<4*1024**3 or evidence['permanent_D_free_bytes']<6*1024**3:raise PermissionError('ACTUAL_UUID_ARGV_SOURCE_SPACE')
 for n,h in plan['source_and_role_sha256'].items():
  if sha(bundle/n)!=h:raise ValueError('SOURCE_ROLE_SHA:'+n)
 joint=read(a.collection_joint);collection=Path(a.collection_root)
 if joint['status']!='ACTUAL_HEADEVAL201_LABEL_FREE_NINE_PREDICTIONS_D_OTHER_CPU_JOINT_PASSED' or joint['plan_sha256']!=sha(bundle/'matched_head_execution_plan.json'):raise PermissionError('REAL_COLLECTION_D_OTHER_CPU_JOINT_REQUIRED')
 head=read(a.head_state);hs=sha(a.head_state)
 if hs!=joint['head_state_sha256'] or head['plan_sha256']!=joint['plan_sha256'] or head['scope']!='FIXED_HEADFIT232_ONLY_U_W_AND_CONSTANT':raise PermissionError('EXACT_FROZEN_FIT_HEAD')
 prediction=collection/'out/frozen_headEVAL201_predictions.npz';ps=sha(prediction);inputfile=collection/'out/original_headEVAL201_label_free.npz';freeze=read(collection/'out/nine_prediction_freeze.json')
 if ps!=joint['frozen_predictions_sha256'] or sha(inputfile)!=joint['original_label_free_array_sha256'] or freeze['head_state_sha256']!=hs or freeze['frozen_predictions_sha256']!=ps or freeze['201_task_labels_indexed']:raise PermissionError('ORIGINAL_PHYSICAL_NINE_PREDICTION_FREEZE')
 if datetime.datetime.fromisoformat(head['actual_frozen_utc'])>=datetime.datetime.fromisoformat(freeze['actual_utc']):raise PermissionError('FIT_HEAD_BEFORE201_PREDICTIONS')
 rows=np.load(bundle/'head_development_eval_0.npy',allow_pickle=False);fitrows=np.load(bundle/'head_development_fit_0.npy',allow_pickle=False);mapping=read(bundle/'train_row_video_mapping.json')
 with np.load(inputfile,allow_pickle=False) as z:
  if not np.array_equal(z['row_ids'],rows):raise ValueError('EXACT201_IDS')
  v=z['video_ids'].copy();segments=z['segment_ids'].copy();f=z['pF'].astype(np.float64);c=z['pC'].astype(np.float64);d=z['delta'].copy()
  if not np.array_equal(c-f,d) or not np.array_equal(z['pF'],z['pF_restored']) or not np.array_equal(z['pC'],z['pC_repeat']):raise ValueError('EXACT_ORIGINAL_CANDIDATE_REPLAY')
 if v.tolist()!=[mapping[int(i)]['video_id'] for i in rows] or segments.tolist()!=[mapping[int(i)]['segment_id'] for i in rows] or len(np.unique(v))!=9 or set(v)&{mapping[int(i)]['video_id'] for i in fitrows}:raise ValueError('EXACT_VIDEO_AND_SEGMENT_ROLES')
 vectors=predict(head['model'],f,d,c)
 with np.load(prediction,allow_pickle=False) as z:
  if set(z.files)!=set(vectors)|{'row_ids','head_state_sha256'} or not np.array_equal(z['row_ids'],rows) or str(z['head_state_sha256'].item())!=hs:raise ValueError('ALL_NINE_FROZEN_SCHEMA')
  if not all(np.array_equal(z[n],p) for n,p in vectors.items()):raise ValueError('NINE_PREDICTION_DISK_REPLAY')
 data=Path(a.asset_base)/'assets/mosi.pkl'
 if sha(data)!=plan['official_mosi_pickle_sha256']:raise ValueError('OFFICIAL_DATA_WHOLE_SHA')
 out=Path(a.out)
 if out.exists():raise FileExistsError('UNIQUE_NEW_ONE_EVALUATION_OUTPUT')
 out.mkdir(parents=True)
 write(out/'before201_label_gate.json',{'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':joint['plan_sha256'],'source_sha256':sha(__file__),'head_state_sha256':hs,'frozen_predictions_sha256':ps,'collection_joint_sha256':sha(a.collection_joint),'rows':rows.tolist(),'all_nine_disk_replay_error':0,'allowed_label_access':'exact201 development rows once, all outputs together'})
 with data.open('rb') as stream:container=pickle.load(stream)
 examples=container['train'];del container
 guard=HeadLabels(examples,fitrows,rows);y=guard.get_eval(rows,prediction,ps,a.head_state,hs)
 if not np.isfinite(y).all():raise ValueError('NO_NONFINITE_LABEL_OR_ROW_DROPPING')
 weights=video_equal_weights(v);r=y-f;report={};identity_errors=[]
 for n,p in vectors.items():
  delta=p-f;Q=delta*delta-2*delta*r;loss=(p-y)**2
  err=float(np.max(np.abs(Q-(loss-(f-y)**2))));identity_errors.append(err)
  pervideo={str(x):{'rows':int(np.sum(v==x)),'MSE':float(np.mean(loss[v==x])),'paired_Q_vs_F':float(np.mean(Q[v==x]))} for x in np.unique(v)}
  report[n]={'five_metrics_same_prediction':metrics(p,y),'video_equal_MSE':float(weights@loss),'video_equal_paired_Q_vs_F':float(weights@Q),'per_video':pervideo,'videos_with_strictly_lower_MSE_than_F':sum(x['paired_Q_vs_F']<0 for x in pervideo.values())}
 if max(identity_errors)>1e-10:raise ValueError('SQUARED_ERROR_PAIRED_Q_IDENTITY')
 comparisons={}
 for mode in ['free','interval','discrete']:
  comparisons['W_vs_U_'+mode]=report['W_'+mode]['video_equal_MSE']-report['U_'+mode]['video_equal_MSE']
 for h in ['U','W']:
  for mode in ['interval','discrete']:comparisons[h+'_'+mode+'_vs_free']=report[h+'_'+mode]['video_equal_MSE']-report[h+'_free']['video_equal_MSE']
 np.savez_compressed(out/'original_headEVAL201_scored_all_nine.npz',row_ids=rows,video_ids=v,segment_ids=segments,labels=y,head_state_sha256=np.asarray(hs),**vectors)
 write(out/'all_nine_development_results.json',{'scope':'one development evaluation; historical TRAIN exploration; not independent confirmation/full crossfit/CaReFlow benchmark','plan_sha256':joint['plan_sha256'],'head_state_sha256':hs,'metrics_semantics':SEMANTICS,'outputs':report,'predeclared_matched_comparisons':comparisons,'readout_or_model_selection':False,'paired_Q_identity_max_error':max(identity_errors)})
 elapsed=time.perf_counter()-start
 if elapsed>1200:raise RuntimeError('EVALUATION_CPU_TIME_BUDGET')
 receipt={'status':'ACTUAL_HEADEVAL201_ONE_DEVELOPMENT_EVALUATION_ALL_NINE_COMPLETE','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),'plan_sha256':joint['plan_sha256'],'collection_joint_sha256':sha(a.collection_joint),'head_state_sha256':hs,'frozen_predictions_sha256':ps,'original_scored_arrays_sha256':sha(out/'original_headEVAL201_scored_all_nine.npz'),'all_nine_results_sha256':sha(out/'all_nine_development_results.json'),'before_label_gate_sha256':sha(out/'before201_label_gate.json'),'task_label_journal':guard.journal,'rows':201,'videos':9,'GPU_used':False,'head_fit_or_reference_update':False,'other_task_label_roles_indexed':False,'output_selection':False,'benchmark_or_overall_research_completed':False,'elapsed_seconds':elapsed}
 write(out/'actual_matched_head_evaluation_receipt.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['bundle','collection-joint','collection-root','head-state','asset-base','evidence','out']:p.add_argument('--'+n,required=True)
 run(p.parse_args())
