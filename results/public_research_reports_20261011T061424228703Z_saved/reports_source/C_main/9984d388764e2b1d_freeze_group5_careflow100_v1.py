"""Freeze one bounded fresh CaReFlow stage after original CPU/Release closure."""
import ast,datetime as dt,json,pathlib,shutil,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,seal
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC

def main():
    closed=json.loads((BASE/'work/group5_real_closed_pointer.json').read_bytes())
    assert closed['proof']['all_member_SHA_CRC_unique_exact_set_passed']
    assert closed['proof']['actual_GitHub_restoration']['all_member_SHA_CRC_unique_exact_set_passed']
    native=P(closed['root'])/'metadata/capture/metadata/out/actual_stage_receipt.json'
    pre=json.loads(native.read_bytes());assert pre['updates']==3 and pre['method']=='careflow' and pre['fold']==0
    cpu=json.loads((P(closed['root'])/'metadata/capture/metadata/independent_CPU_audit.json').read_bytes())
    assert cpu['status']=='INDEPENDENT_CPU_ORIGINAL_DIRECT_STATE_AUDIT_PASS'
    now=dt.datetime.now(dt.timezone.utc);root=DC.parent/('g5careflow100_plan_'+now.strftime('%Y%m%dT%H%M%SZ'));root.mkdir()
    source=root/'payload';source.mkdir();remote='/data/coding/g5careflow100_20261010T043810Z'
    assets='/data/coding/batch5_continue_actual_20261009T163550Z/N2_public_runtime_actual_20261009T174820Z'
    p=json.loads((DC.parent/'g5pipe_20261010T032228Z/careflow_native_precheck_plan.json').read_bytes())
    for n in ['group5_direct_runtime_v2.py','group5_epoch_resume_v1.py','group5_direct_CPU_audit_v2.py']:
        p['source_SHA'][n]=digest(BASE/'work'/n)
    free={d:shutil.disk_usage(d+':/').free for d in ['C','D']}
    # Serial original capture plus independently downloaded whole restoration.
    assert free['C']>2*3100*1024**2+200*1024**2 and free['D']>40*1024**2
    fit=pre['projected_fit_seconds'];overhead=100*64
    assert fit+overhead<16200 and (dt.datetime.fromisoformat(p['lease_end_UTC'])-now).total_seconds()>16200+7200
    p.update(status='GROUP5_DIRECT_TRAIN_FROZEN',stage_budget_seconds=16200,saving_budget_seconds=7200,
        local_C_free_bytes=free['C'],local_D_free_bytes=free['D'],local_space_capture_UTC=now.isoformat(),
        remote_required_bytes=20*1024**3,new_once_token=remote+'/careflow0_train100.once',
        same_method_fold_native_CPU_qualified=True,Release_range_restore_qualified=True,fresh_initial_state_qualified=True,
        native_precheck_receipt='/data/coding/g5native_20261010T032338Z/out/actual_stage_receipt.json',
        qualification_original_SHA=closed['proof']['archive_SHA'],qualification_GitHub_commit=closed['github']['commit'],
        fresh_resume_disabled=True,queue_scope='ONLY_CAREFLOW_FOLD0_NO_OTHER_STAGE_DISPATCH',
        measured_budget=dict(precheck_FIT_seconds=[r['seconds'] for r in pre['steps']],
            conservative_FIT_seconds=fit,additional_validation_hash_save_allowance_seconds=overhead,
            total_stage_budget_seconds=16200,projection_not_guaranteed=True))
    write(source/'plan.json',p);h=digest(source/'plan.json')
    wrapper='''import datetime as dt,json,os,pathlib,subprocess
P=pathlib.Path
root=P(REMOTE)
env=os.environ.copy();env['PYTHONPATH']=str(root/'source')
argv=[PYTHON,'-B',str(root/'source/group5_direct_runtime_v2.py'),'--plan',str(root/'plan.json'),'--plan-sha',PLAN_SHA,'--bundle',str(root/'source'),'--assets',ASSETS,'--out',str(root/'out'),'--method','careflow','--fold','0','--stage','train']
with (root/'stdout.log').open('xb') as log:
    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env)
    (root/'dispatch.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),child=child.pid,argv=argv,plan_SHA=PLAN_SHA)),encoding='utf8')
    code=child.wait()
(root/'wrapper_exit.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),child=child.pid,natural_exit=code)),encoding='utf8')
raise SystemExit(code)
'''
    prefix='REMOTE='+repr(remote)+'\nPYTHON='+repr(p['python'])+'\nASSETS='+repr(assets)+'\nPLAN_SHA='+repr(h)+'\n'
    wrapper=prefix+wrapper;ast.parse(wrapper);(source/'wrapper.py').write_text(wrapper,encoding='utf8')
    shutil.copyfile(BASE/'work/freeze_group5_careflow100_v1.py',source/'freeze_group5_careflow100_v1.py')
    shutil.copyfile(BASE/'work/group5_direct_CPU_audit_v2.py',source/'group5_direct_CPU_audit_v2.py')
    proof=seal(source,source/'formal_plan_original.zip');write(root/'D_plan_receipt.json',proof)
    publish(P(proof['archive']),proof['archive_SHA'],'group5-careflow0-formal-plan-'+proof['archive_SHA'][:12]+'.zip',root/'Release_plan_receipt.json')
    public=json.loads((root/'Release_plan_receipt.json').read_bytes())
    pointer=dict(root=str(root),remote=remote,plan_SHA=h,proof=proof,Release=public,formal_training_started=False)
    write(BASE/'work/group5_formal_plan_pointer.json',pointer)
    print(json.dumps(pointer,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
