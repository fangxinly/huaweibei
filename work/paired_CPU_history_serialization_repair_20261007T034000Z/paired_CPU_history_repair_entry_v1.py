"""Run every original CPU check; normalize only the proved JSON tuple/list field in a copied history view."""
import json,pathlib,sys
from repair_gate import repair_gate
def main():
 args=sys.argv[1:];extra={}
 for flag in ('--repair-plan','--repair-plan-sha'):
  assert args.count(flag)==1;i=args.index(flag);extra[flag]=args[i+1];del args[i:i+2]
 rp=repair_gate(extra['--repair-plan'],extra['--repair-plan-sha'])
 bundle=pathlib.Path(args[args.index('--bundle')+1]);sys.path.insert(0,str(bundle))
 import paired_fulltrain_CPU_audit_candidate_v1 as original
 assert original.sha(bundle/'paired_fulltrain_CPU_audit_candidate_v1.py')==rp['original_CPU_source_sha256']
 assert args[args.index('--plan-sha')+1]==rp['original_training_plan_sha256']
 oldverify=original.verify_training_arrays;oldwrite=original.write
 def verify(np,root,plan,receipt,resume,selected):
  history=original.read(pathlib.Path(root)/'out/history.json');rh=resume['history']
  assert isinstance(history,list) and isinstance(rh,list) and len(history)==len(rh)==100
  normalized=[]
  for x,y in zip(history,rh):
   assert set(x)==set(y)
   assert isinstance(x['dropped_TRAIN_rows'],list) and type(y['dropped_TRAIN_rows']) is tuple
   assert len(x['dropped_TRAIN_rows'])==len(y['dropped_TRAIN_rows'])==1
   assert type(x['dropped_TRAIN_rows'][0]) is int and type(y['dropped_TRAIN_rows'][0]) is int
   assert x['dropped_TRAIN_rows']==list(y['dropped_TRAIN_rows'])
   assert all(x[k]==y[k] and type(x[k]) is type(y[k]) for k in x if k!='dropped_TRAIN_rows')
   row=dict(y);row['dropped_TRAIN_rows']=list(y['dropped_TRAIN_rows']);normalized.append(row)
  view=dict(resume);view['history']=normalized
  result=oldverify(np,root,plan,receipt,view,selected)
  result['history_serialization_only']={'rows':100,'sole_field':'dropped_TRAIN_rows','JSON_type':'list','Torch_type':'tuple','all_values_and_other_field_types_exact':True,'checkpoint_mutated':False}
  return result
 def write(path,value):
  if pathlib.Path(path).name=='actual_stage_receipt.json':
   value=dict(value,argv=full,CPU_history_repair_plan_sha256=extra['--repair-plan-sha'],CPU_history_repair_plan=extra['--repair-plan'],original_CPU_source_sha256=rp['original_CPU_source_sha256'],CPU_history_repair_source_sha256=rp['source_sha256'][pathlib.Path(__file__).name],prior_failed_CPU_original_preserved=True)
  oldwrite(path,value)
 original.verify_training_arrays=verify;original.write=write
 full=sys.argv[:];sys.argv=[full[0],*args]
 try:original.main()
 finally:sys.argv=full
if __name__=='__main__':main()
