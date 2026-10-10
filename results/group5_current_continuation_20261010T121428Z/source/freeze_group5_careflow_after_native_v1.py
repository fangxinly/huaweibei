"""Freeze one fresh careflow fold only after its native original is fully closed."""
import ast,datetime as dt,json,math,pathlib,shutil,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,seal
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC
def main():
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--native-pointer',type=P,required=True);args=a.parse_args()
 ptr=json.loads(args.native_pointer.read_bytes());q=ptr['qualification'];native_root=P(ptr['preservation_root']);meta=native_root/'metadata/original'
 assert q['status']=='SAME_METHOD_FOLD_NATIVE3_CPU_FULL_RELEASE_RESTORE_CLOSED' and q['method']=='careflow'
 assert all(q[k] for k in ['same_method_fold_native_CPU_qualified','Release_range_restore_qualified','fresh_initial_state_qualified'])
 assert q['original_archive_SHA']==ptr['expected_SHA'] and q['original_native_receipt_SHA']==digest(meta/'out/actual_stage_receipt.json')
 p=json.loads((meta/'plan.json').read_bytes());native=json.loads((meta/'out/actual_stage_receipt.json').read_bytes())
 assert native['method']=='careflow' and 'careflow' in p['authorized_methods'] and native['fold']==p['fold']==q['fold']
 assert q['source_SHA']==p['source_SHA'] and q['split_SHA']==p['split_SHA'] and native['clean_initial_state_SHA']==q['clean_initial_state_SHA']
 reference_ptr=json.loads((BASE/'work/group5_careflow0_complete_pointer.json').read_bytes());assert reference_ptr['closed_receipt']['status']=='CAREFLOW_FOLD0_FORMAL100_CPU_FULL_RELEASE_RESTORE_CLOSED_OUTER_UNSCORED'
 ref=P(reference_ptr['preservation_root'])/'metadata/original';r=json.loads((ref/'out/actual_stage_receipt.json').read_bytes());dispatch=json.loads((ref/'dispatch.json').read_bytes())
 wrapper=json.loads((ref/'wrapper_exit.json').read_bytes());natural=json.loads((ref/'out/natural_exit.json').read_bytes())
 assert r['method']=='careflow' and r['updates']==4700 and wrapper['natural_exit']==natural['exit']==0
 fit=sum(json.loads(s)['seconds'] for s in (ref/'out/FIT_steps.jsonl').read_text().splitlines())
 # A result receipt precedes final exit. Include the complete observed wrapper
 # lifetime instead of shortening the estimate to the receipt-writing time.
 end=max(dt.datetime.fromisoformat(wrapper['actual_UTC']),dt.datetime.fromisoformat(natural['actual_UTC']))
 whole=(end-dt.datetime.fromisoformat(dispatch['actual_UTC'])).total_seconds();overhead=whole-fit
 assert 0<fit<whole and p['updates']==4700
 # Include the independently projected target-fold FIT cost and actual whole
 # reference-stage hashing, INNER evaluation and checkpoint costs separately.
 budget=60*math.ceil(max(whole*1.55,native['projected_fit_seconds']+overhead*1.10)/60)
 now=dt.datetime.now(dt.timezone.utc);remaining=(dt.datetime.fromisoformat(p['lease_end_UTC'])-now).total_seconds()
 free={d:shutil.disk_usage(d+':/').free for d in ['C','D']}
 root=DC.parent/('g5careflow'+str(q['fold'])+'_measured_stage_gate_'+now.strftime('%Y%m%dT%H%M%SZ'));root.mkdir()
 evidence=dict(actual_UTC=now.isoformat(),method='careflow',fold=q['fold'],reference_formal_original_receipt_SHA=digest(ref/'out/actual_stage_receipt.json'),
  reference_whole_stage_seconds=whole,reference_FIT_seconds=fit,reference_nonFIT_seconds=overhead,target_native_conservative_FIT_seconds=native['projected_fit_seconds'],
  reference_wrapper_exit_SHA=digest(ref/'wrapper_exit.json'),reference_natural_exit_SHA=digest(ref/'out/natural_exit.json'),whole_stage_ends_at_original_observed_exit=True,
  whole_reference_margin_factor=1.55,nonFIT_margin_factor=1.10,total_stage_budget_seconds=budget,saving_budget_seconds=7200,
  remaining_lease_seconds=remaining,conservative_lease_end_UTC=p['lease_end_UTC'],lease_platform_confirmed=False,
  projection_not_guaranteed=True,resource_floors_not_lowered=True,new_training_once_consumed=False,source_native_qualification=q,
  local_C_free_bytes=free['C'],local_D_free_bytes=free['D'])
 enough=remaining>=budget+7200 and free['C']>2*3100*1024**2+200*1024**2 and free['D']>40*1024**2
 evidence.update(status='FRESH_STAGE_CAN_BE_FROZEN' if enough else 'NEW_TRAIN_NOT_FROZEN_LEASE_OR_STAGING_MARGIN',execution_enabled=enough)
 write(root/'actual_measured_stage_gate.json',evidence);shutil.copyfile(P(__file__),root/P(__file__).name)
 if not enough:
  write(BASE/'work/group5_careflow_next_gate_pointer.json',dict(root=str(root),gate=evidence,formal_training_dispatched=False));print(json.dumps(evidence),flush=True);return
 remote='/data/coding/g5careflow'+str(q['fold'])+'_train100_'+now.strftime('%Y%m%dT%H%M%SZ');payload=root/'payload';payload.mkdir()
 p.update(status='GROUP5_DIRECT_TRAIN_FROZEN',stage_budget_seconds=budget,saving_budget_seconds=7200,new_once_token=remote+'/train100.once',
  remote_required_bytes=20*1024**3,local_C_free_bytes=free['C'],local_D_free_bytes=free['D'],local_space_capture_UTC=now.isoformat(),
  same_method_fold_native_CPU_qualified=True,Release_range_restore_qualified=True,fresh_initial_state_qualified=True,
  native_precheck_receipt=q['remote_root']+'/out/actual_stage_receipt.json',qualification_original_SHA=q['original_archive_SHA'],
  qualification_GitHub_commit=ptr['github']['commit'],fresh_resume_disabled=True,queue_scope='ONLY_THIS_CAREFLOW_FOLD',measured_budget=evidence)
 write(payload/'plan.json',p);h=digest(payload/'plan.json')
 assets=str(P(p['python']).parent.parent.parent).replace('\\','/')
 wrapper="import datetime as dt,json,os,pathlib,subprocess\nr=pathlib.Path("+repr(remote)+")\ne=dict(os.environ,PYTHONPATH=str(r/'source'),PYTHONDONTWRITEBYTECODE='1')\na="+repr([p['python'],'-B',remote+'/source/group5_direct_runtime_v2.py','--plan',remote+'/plan.json','--plan-sha',h,'--bundle',remote+'/source','--assets',assets,'--out',remote+'/out','--method','careflow','--fold',str(q['fold']),'--stage','train'])+"\nwith (r/'stdout.log').open('xb') as f:\n c=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT,env=e)\n (r/'dispatch.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),child=c.pid,argv=a,plan_SHA="+repr(h)+")))\n code=c.wait()\n(r/'wrapper_exit.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),child=c.pid,natural_exit=code)))\nraise SystemExit(code)\n"
 ast.parse(wrapper);(payload/'wrapper.py').write_text(wrapper,encoding='utf8');shutil.copyfile(P(__file__),payload/P(__file__).name)
 proof=seal(payload,payload/'formal_plan_original.zip');write(root/'D_plan_receipt.json',proof)
 publish(P(proof['archive']),proof['archive_SHA'],'group5-careflow'+str(q['fold'])+'-formal-plan-'+proof['archive_SHA'][:12]+'.zip',root/'Release_plan_receipt.json')
 write(BASE/'work/group5_careflow_next_gate_pointer.json',dict(root=str(root),remote=remote,plan_SHA=h,gate=evidence,proof=proof,formal_training_dispatched=False))
 print(json.dumps(dict(root=str(root),remote=remote,plan_SHA=h,status='FORMAL_PLAN_FROZEN_NOT_DISPATCHED')),flush=True)
if __name__=='__main__':main()
