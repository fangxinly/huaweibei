"""Synthetic role, chronology, arithmetic tests. No official inputs/labels."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from head_development_input_guard_v1 import allowed_head_inputs,candidate_gate
from matched_fixed_residual_heads_v2 import design,fit,predict,HeadLabels
from matched_development_independent_math_v1 import readouts,independent_metrics,compare_metrics
from sentiment_metrics_careflow_v1 import metrics
class Example:
 def __init__(self,index):self.index=index
 def __getitem__(self,k):
  if k==1:raise AssertionError('LABEL_INDEX_IN_INPUT_ONLY_GUARD')
  return np.asarray([self.index]) if k==0 else 'synthetic'
ids=np.arange(232,433);fitids=np.arange(232);rowvideo=['unused']*1281
for pos,i in enumerate(ids):rowvideo[int(i)]='evaluation_'+str(pos%9)
records=allowed_head_inputs([Example(i) for i in range(1281)],ids,ids,rowvideo);assert len(records)==201 and all(np.array_equal(x[1],np.zeros((1,1),np.float32)) for x in records)
negative=0
for rows in [ids[::-1],np.arange(201),np.r_[ids[:-1],ids[-2]],ids.astype(float)]:
 try:allowed_head_inputs([Example(i) for i in range(1281)],rows,ids,rowvideo)
 except PermissionError:negative+=1
 else:raise AssertionError('EXACT201_GUARD_NEGATIVE')
rng=np.random.default_rng(91819);fv=np.asarray(['fit_'+str(i%9) for i in range(232)]);f=rng.normal(size=232);c=f+rng.normal(scale=.2,size=232);d=c-f;y=rng.normal(size=232)
model=fit(f,d,y,fv,design(f,d,fv));ef=rng.normal(size=201).astype(np.float32);ec=(ef+rng.normal(scale=.2,size=201)).astype(np.float32);ed=ec.astype(np.float64)-ef
ev=np.asarray([rowvideo[int(i)] for i in ids]);diag=candidate_gate(ef,ec,ids,ev,ef.copy());assert not diag['labels_used']
same=predict(model,ef,ed,ec);independent=readouts(model,ef,ec);assert all(np.array_equal(same[n],independent[n]) for n in same)
ey=rng.normal(size=201);ey[0]=0.;ey[1]=0.;ey[2]=3.;ey[3]=-3.
metric_error=max(compare_metrics(metrics(vec,ey),independent_metrics(vec,ey)) for vec in same.values())
examples=[(None,np.asarray([[rng.normal()]]),None) for i in range(1281)]
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);head=root/'head.json';head.write_text(json.dumps({'synthetic':True}));hs=hashlib.sha256(head.read_bytes()).hexdigest();pred=root/'all_nine.npz';np.savez_compressed(pred,row_ids=ids,head_state_sha256=np.asarray(hs),**same);ps=hashlib.sha256(pred.read_bytes()).hexdigest()
 guard=HeadLabels(examples,fitids,ids);labels=guard.get_eval(ids,pred,ps,head,hs);assert labels.shape==(201,) and len(guard.journal)==1
 try:guard.get_eval(ids,pred,ps,head,hs)
 except PermissionError:negative+=1
 else:raise AssertionError('SECOND_EVALUATION_LABEL_ACCESS')
 guard=HeadLabels(examples,fitids,ids)
 try:guard.get_eval(ids,pred,'0'*64,head,hs)
 except PermissionError:negative+=1
 else:raise AssertionError('ALTERED_PREDICTIONS_BEFORE_LABELS')
 assert not guard.journal
 bundle=root/'bundle';bundle.mkdir();(bundle/'matched_head_execution_plan.json').write_text(json.dumps({'status':'LOCAL_PREPARATION_ONLY'}))
 commands=[['fixed_head_development_collector_v1.py','--asset-base','MUST_NOT_OPEN','--checkpoint','MUST_NOT_OPEN','--completion-joint','MUST_NOT_OPEN','--evidence','MUST_NOT_OPEN','--head-state','MUST_NOT_OPEN','--head-fit-joint','MUST_NOT_OPEN','--out',str(root/'out')],['score_frozen_matched_heads201_v1.py','--collection-joint','MUST_NOT_OPEN','--collection-root','MUST_NOT_OPEN','--head-state','MUST_NOT_OPEN','--asset-base','MUST_NOT_OPEN','--evidence','MUST_NOT_OPEN','--out',str(root/'out')],['matched_head_natural_stage_wrapper_v1.py','--root',str(root/'run'),'--stage','predict','--child-arguments','MUST_NOT_OPEN']]
 for command in commands:
  result=subprocess.run([sys.executable,str(Path(__file__).parent/command[0]),'--bundle',str(bundle),*command[1:]],capture_output=True,text=True)
  assert result.returncode!=0 and 'PROTOCOL_NOT_READY' in result.stderr and not (root/'out').exists() and not (root/'run').exists()
record={'status':'SYNTHETIC201_ROLE_CHRONOLOGY_AND_INDEPENDENT_ARITHMETIC_PREPARATION_PASSED','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'negative_role_and_label_checks':negative,'incomplete_protocol_entrypoints_rejected':3,'all_nine_independent_replay_max_error':0,'independent_five_metric_max_error':metric_error,'official_task_inputs_labels_GPU_or_remote_accessed':False,'actual201_collection_scoring_or_headFIT232_fit':False}
Path(__file__).with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
