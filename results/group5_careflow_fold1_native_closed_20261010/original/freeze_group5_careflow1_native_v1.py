"""Freeze only a fresh second-fold native precheck, never another fold0 run."""
import ast,datetime as dt,json,pathlib,shutil,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,seal
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC

def main():
 now=dt.datetime.now(dt.timezone.utc);root=DC.parent/('g5careflow1_native_plan_'+now.strftime('%Y%m%dT%H%M%SZ'));root.mkdir();q=root/'payload';q.mkdir()
 origin=DC.parent/'g5careflow100_plan_20261010T044701Z/payload/plan.json';p=json.loads(origin.read_bytes())
 assert digest(origin)=='443b6eb01ba6f4f3f2e14119d93715abe8a8a6a2590019a42dc05a33f91e6845'
 remote='/data/coding/g5careflow1_native_'+now.strftime('%Y%m%dT%H%M%SZ');assets='/data/coding/batch5_continue_actual_20261009T163550Z/N2_public_runtime_actual_20261009T174820Z'
 free={d:shutil.disk_usage(d+':/').free for d in ['C','D']}
 assert free['C']>200*1024**2 and free['D']>40*1024**2
 assert (dt.datetime.fromisoformat(p['lease_end_UTC'])-now).total_seconds()>900+7200
 p.update(status='GROUP5_DIRECT_NATIVE_PRECHECK_FROZEN',fold=1,orders_SHA=p['source_SHA']['orders_fold1.npy'],updates=4700,
  stage_budget_seconds=900,saving_budget_seconds=7200,new_once_token=remote+'/careflow1_precheck.once',remote_required_bytes=12*1024**3,
  local_C_free_bytes=free['C'],local_D_free_bytes=free['D'],local_space_capture_UTC=now.isoformat(),
  same_method_fold_native_CPU_qualified=False,Release_range_restore_qualified=False,fresh_initial_state_qualified=False,
  queue_scope='ONLY_CAREFLOW_FOLD1_THREE_UPDATE_PRECHECK_NO_FORMAL_TRAIN_DISPATCH',
  preceding_formal_training_not_repeated=True,precheck_is_additional_qualification_cost=True)
 for key in ['native_precheck_receipt','qualification_original_SHA','qualification_GitHub_commit','measured_budget']:p.pop(key,None)
 write(q/'plan.json',p);h=digest(q/'plan.json')
 wrapper="""import datetime as dt,json,os,pathlib,subprocess
root=pathlib.Path(REMOTE)
env=dict(os.environ,PYTHONPATH=str(root/'source'),PYTHONDONTWRITEBYTECODE='1')
argv=[PYTHON,'-B',str(root/'source/group5_direct_runtime_v2.py'),'--plan',str(root/'plan.json'),'--plan-sha',PLAN_SHA,'--bundle',str(root/'source'),'--assets',ASSETS,'--out',str(root/'out'),'--method','careflow','--fold','1','--stage','precheck']
with (root/'stdout.log').open('xb') as log:
 child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env)
 (root/'dispatch.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),child=child.pid,argv=argv,plan_SHA=PLAN_SHA)))
 code=child.wait()
(root/'wrapper_exit.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),child=child.pid,natural_exit=code)))
raise SystemExit(code)
"""
 wrapper='REMOTE='+repr(remote)+'\nPYTHON='+repr(p['python'])+'\nASSETS='+repr(assets)+'\nPLAN_SHA='+repr(h)+'\n'+wrapper
 ast.parse(wrapper);(q/'wrapper.py').write_text(wrapper,encoding='utf8');shutil.copyfile(P(__file__),q/P(__file__).name)
 proof=seal(q,q/'native_plan_original.zip');write(root/'D_plan_receipt.json',proof)
 publish(P(proof['archive']),proof['archive_SHA'],'group5-careflow1-native-plan-'+proof['archive_SHA'][:12]+'.zip',root/'Release_plan_receipt.json')
 v=dict(root=str(root),remote=remote,plan_SHA=h,Release=json.loads((root/'Release_plan_receipt.json').read_bytes()),proof=proof,precheck_dispatched=False,formal_training_dispatched=False)
 write(BASE/'work/group5_careflow1_native_pointer.json',v);print(json.dumps(v),flush=True)

if __name__=='__main__':main()
