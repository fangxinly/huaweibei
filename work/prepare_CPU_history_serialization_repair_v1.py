import ast, datetime, hashlib, json, pathlib, zipfile

base=pathlib.Path(__file__).resolve().parent
old=base/'paired_official_fulltrain_v2_20261007T011543Z'
D=pathlib.Path('D:/CodexBackups/selective_flow_20261003_1105/paired_fulltrain100_complete_actual_20261007T030311Z')
failure=D/'CPU_history_failure_original_20261007T033805Z'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
fr=json.loads((failure/'original_failure_receipt.json').read_text())
assert sha(failure/'original_failure.zip')==fr['ZIP_SHA']=='26ba08440dffd5e1cbe95d86ee08fb346524da304d42cb3effc64231daaf45a8'
with zipfile.ZipFile(failure/'original_failure.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==31
 manifest=json.loads(z.read('original_failure_manifest.json'))
 for n,r in manifest['members'].items():assert hashlib.sha256(z.read(n)).hexdigest()==r['sha256']
 for m in ('careflow','minimal_fixed_F'):
  assert json.loads(z.read(m+'/natural_exit.json'))['exit_code']==1
diag=json.loads((failure/'readonly_history_representation_diagnosis.json').read_text())
assert sha(failure/'readonly_history_representation_diagnosis.json')=='5bf0388c61392135da6d3d2c9f07d68559dcde6d11c02cc0a27be5e02b07c747'
for v in diag['methods'].values():
 assert v['history_rows']==v['resume_history_rows']==100 and v['JSON_value_equal'] and not v['raw_equal']
 assert len(v['differences'])==100
 assert all(x['field']=='dropped_TRAIN_rows' and x['JSON_type']=='list' and x['resume_type']=='tuple' and x['JSON_value']==x['resume_value'] for x in v['differences'])
tag='20261007T034000Z'
dest=base/('paired_CPU_history_serialization_repair_'+tag);dest.mkdir()
gate='''import hashlib,json,pathlib,sys
def repair_gate(plan_path,expected):
 p=pathlib.Path(plan_path);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==expected
 plan=json.loads(data);assert plan['status']=='CPU_HISTORY_TUPLE_LIST_ONLY_PRESERVATION_REPAIR_PROTOCOL_FROZEN'
 assert plan['training_or_model_change_enabled'] is False and plan['TEST_enabled'] is False
 for n,h in plan['source_sha256'].items():assert hashlib.sha256((p.parent/n).read_bytes()).hexdigest()==h
 return plan
'''
(dest/'repair_gate.py').write_text(gate,encoding='utf-8')
entry='''"""Run every original CPU check; normalize only the proved JSON tuple/list field in a copied history view."""
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
   value=dict(value,CPU_history_repair_plan_sha256=extra['--repair-plan-sha'],CPU_history_repair_plan=extra['--repair-plan'],original_CPU_source_sha256=rp['original_CPU_source_sha256'],CPU_history_repair_source_sha256=rp['source_sha256'][pathlib.Path(__file__).name],prior_failed_CPU_original_preserved=True)
  oldwrite(path,value)
 original.verify_training_arrays=verify;original.write=write
 full=sys.argv[:];sys.argv=[full[0],*args]
 try:original.main()
 finally:sys.argv=full
if __name__=='__main__':main()
'''
# Keep the full supplemental arguments in the original receipt and exit association.
entry=entry.replace("full=sys.argv[:];sys.argv=[full[0],*args]", "full=sys.argv[:];sys.argv=[full[0],*args]")
entry=entry.replace("value=dict(value,CPU_history_repair_plan_sha256", "value=dict(value,argv=full,CPU_history_repair_plan_sha256")
(dest/'paired_CPU_history_repair_entry_v1.py').write_text(entry,encoding='utf-8')
wrapper=(old/'paired_fulltrain_CPU_natural_wrapper_candidate_v1.py').read_text()
wrapper=wrapper.replace('from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write, plan_gate','import json,hashlib,datetime,shutil\nfrom repair_gate import repair_gate\n')
wrapper=wrapper.replace("    p.add_argument('--plan-sha', required=True)","    p.add_argument('--plan-sha', required=True)\n    p.add_argument('--repair-plan',type=Path,required=True)\n    p.add_argument('--repair-plan-sha',required=True)")
wrapper=wrapper.replace('    plan = plan_gate(a.bundle, a.plan_sha)',"    rp=repair_gate(a.repair_plan,a.repair_plan_sha)\n    sys.path.insert(0,str(a.bundle))\n    from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write, plan_gate\n    plan = plan_gate(a.bundle,a.plan_sha)\n    require(a.plan_sha==rp['original_training_plan_sha256'],'Original training protocol remains exact')\n    import subprocess,datetime\n    raw={'actual_utc':now(),'UUID':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True),'compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),'fullargv':subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True),'space':subprocess.check_output(['df','-B1',str(a.root)],text=True)}\n    require(raw['UUID'].strip()==plan['original_CPU_gpu_UUID'] and not raw['compute'].strip(),'Fresh actual B identity/empty compute')\n    require(shutil.disk_usage(a.root).free>=plan['remote_free_floor_bytes'],'Fresh actual space floor')\n    require(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=8100)<datetime.datetime.fromisoformat(rp['conservative_lease_end_UTC']),'900s execution plus2h preservation margin')\n    for n,h in plan['asset_sha256'].items():require(sha(Path(rp['public_assets'])/n)==h,'Fresh original public asset')\n    write(a.root/'actual_repair_dispatch_fresh_direct.json',raw)")
wrapper=wrapper.replace("    source = a.bundle / 'paired_fulltrain_CPU_audit_candidate_v1.py'\n    require(sha(source) == plan['source_sha256'][source.name], 'Frozen CPU source mismatch')","    source=a.repair_plan.parent/'paired_CPU_history_repair_entry_v1.py'\n    require(sha(source)==rp['source_sha256'][source.name],'Supplemental frozen CPU source mismatch')\n    for n in (*rp['source_sha256'],'repair_plan.json'):\n        target=a.root/'repair_source'/n;target.parent.mkdir(exist_ok=True);require(not target.exists(),'Fresh repair source copy');shutil.copyfile(a.repair_plan.parent/n,target)")
(dest/'paired_CPU_history_repair_wrapper_v1.py').write_text(wrapper,encoding='utf-8')
joint=(old/'paired_fulltrain_saved_joint_candidate_v1.py').read_text()
joint=joint.replace('import argparse','import argparse\nfrom repair_gate import repair_gate')
joint=joint.replace("    p.add_argument('--extract', action='store_true')","    p.add_argument('--extract', action='store_true')\n    p.add_argument('--repair-plan',type=Path,required=True)\n    p.add_argument('--repair-plan-sha',required=True)")
joint=joint.replace('    root = a.D_root.resolve()',"    rp=repair_gate(a.repair_plan,a.repair_plan_sha)\n    require(a.plan_sha==rp['original_training_plan_sha256'],'Exact unchanged training protocol')\n    root = a.D_root.resolve()")
joint=joint.replace("        require(wrapper['child_source_sha256'] == plan['source_sha256'][source], 'Original wrapper/actual child source mismatch')","        expected=plan['source_sha256'][source] if role=='a' else rp['source_sha256']['paired_CPU_history_repair_entry_v1.py']\n        require(wrapper['child_source_sha256']==expected,'Actual original GPU or supplemental CPU source mismatch')\n        if role=='b':\n            require(cpu['CPU_history_repair_plan_sha256']==a.repair_plan_sha and cpu['original_CPU_source_sha256']==plan['source_sha256'][source] and cpu['prior_failed_CPU_original_preserved'] is True,'Supplemental CPU exact parent/failure/source binding')\n            for n,h in rp['source_sha256'].items():require(sha(root/'b/run/repair_source'/n)==h,'Captured actual supplemental CPU source')\n            require(sha(root/'b/run/repair_source/repair_plan.json')==a.repair_plan_sha,'Captured supplemental protocol')\n            checks=cpu['checks']['arrays_history_orders']['history_serialization_only']\n            require(checks=={'rows':100,'sole_field':'dropped_TRAIN_rows','JSON_type':'list','Torch_type':'tuple','all_values_and_other_field_types_exact':True,'checkpoint_mutated':False},'Strict sole-field normalization proof')")
joint=joint.replace("    write(a.out, result)","    result['CPU_history_repair_plan_sha256']=a.repair_plan_sha\n    result['CPU_history_repair_source_sha256']=rp['source_sha256']['paired_CPU_history_repair_entry_v1.py']\n    result['prior_CPU_natural1_failure_preserved']=True\n    result['original_training_source_or_weights_changed']=False\n    write(a.out, result)")
(dest/'paired_CPU_history_repair_saved_joint_v1.py').write_text(joint,encoding='utf-8')
for p in dest.glob('*.py'):ast.parse(p.read_text())
plan={'status':'CPU_HISTORY_TUPLE_LIST_ONLY_PRESERVATION_REPAIR_PROTOCOL_FROZEN','actual_local_freeze_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_training_plan_sha256':'8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb','original_CPU_source_sha256':sha(old/'paired_fulltrain_CPU_audit_candidate_v1.py'),'source_sha256':{p.name:sha(p) for p in sorted(dest.glob('*.py'))},'prior_original_failure_ZIP_SHA':fr['ZIP_SHA'],'diagnosis_SHA':sha(failure/'readonly_history_representation_diagnosis.json'),'sole_allowed_representation_change':'100 dropped_TRAIN_rows tuple->list in copied comparison view; one integer row each, every value/other field/type exact','training_or_model_change_enabled':False,'TEST_enabled':False,'CPU_model_forward':False,'public_assets':'/data/coding/multimodal_flow_public_20261006T1341Z','conservative_lease_end_UTC':'2026-10-07T13:30:00+00:00','not_platform_lease_confirmation':True}
(dest/'repair_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
save=D/dest.name;save.mkdir()
for p in dest.iterdir():(save/p.name).write_bytes(p.read_bytes())
with zipfile.ZipFile(dest/'upload.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(dest.iterdir()):
  if p.name!='upload.zip':z.write(p,p.name)
(save/'upload.zip').write_bytes((dest/'upload.zip').read_bytes())
print(json.dumps({'local':str(dest),'D':str(save),'repair_plan_SHA':sha(dest/'repair_plan.json'),'upload_SHA':sha(dest/'upload.zip'),'source_SHA':plan['source_sha256']},indent=2))
