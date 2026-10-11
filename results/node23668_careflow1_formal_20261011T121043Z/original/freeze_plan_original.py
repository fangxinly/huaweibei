"""Preserve actual runtime exit, then freeze the one still-unstarted formal fold."""
import ast,datetime as dt,hashlib,json,pathlib,shutil,sys,urllib.request
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;W=BASE/'work';sys.path.insert(0,str(W))
from group5_release_transport_v1 import digest,write,seal,verify_zip
from raw_TRAIN_save_and_publish_v1 import publish
from group5_publish_exact_local_v1 import publish_tree
from publish_test_selected_group5_preparation_v1 import DC,sync
def publish_complete_small(root,folder,name):
    proof=seal(folder,folder/(name+'.zip'));write(root/'D_preservation_receipt.json',proof)
    publish(P(proof['archive']),proof['archive_SHA'],name+'-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
    release=json.loads((root/'Release_receipt.json').read_bytes());restored=root/'GitHub_restored_complete.zip'
    with urllib.request.urlopen(release['url'],timeout=180) as r,restored.open('xb') as out:shutil.copyfileobj(r,out,1024**2)
    assert digest(restored)==proof['archive_SHA'];verified=verify_zip(restored);write(root/'GitHub_restoration.json',dict(SHA=digest(restored),**verified))
    public=root/'text_publication';public.mkdir()
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ['.py','.json','.md','.txt','.log']:
            target=public/'original'/p.relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for n in ['D_preservation_receipt.json','Release_receipt.json','GitHub_restoration.json']:shutil.copyfile(root/n,public/n)
    github=publish_tree(public,'results/'+root.name,'Preserve '+name+' with exact original verification')
    write(root/'GitHub_publication_receipt.json',github);return dict(root=str(root),proof=proof,Release=release,GitHub=github,actual_restoration=verified)
def main():
    node=json.loads((W/'new_node_23668_active.json').read_bytes());rp=json.loads((W/'new_node_23668_runtime_pointer.json').read_bytes())
    capture=W/'new_node_23668_preflight/runtime_success_original';receipt=json.loads((capture/'runtime_metadata/runtime_original_receipt.json').read_bytes())
    runtime_proof=json.loads((W/'new_node23668_runtime_complete_pointer.json').read_bytes())
    assert json.loads((capture/'canonical_runtime_wrapper_exit.json').read_bytes())['natural_exit']==0
    runtime_root=pathlib.Path(runtime_proof['root'])
    setup=json.loads((W/'new_node23668_setup_pointer.json').read_bytes());native=json.loads((W/'group5_careflow1_native_capture_pointer.json').read_bytes());q=native['qualification'];native_root=P(native['preservation_root'])/'metadata/original'
    assert q['CPU_original_exit']==q['native_original_exit']==0 and q['original_once_not_repeated'] and q['all_member_SHA_CRC_unique_exact_set_passed']
    assert q['actual_GitHub_restoration']['all_member_SHA_CRC_unique_exact_set_passed']
    assert digest(native_root/'out/actual_stage_receipt.json')==q['original_native_receipt_SHA']
    p=json.loads((native_root/'plan.json').read_bytes());assert p['fold']==1 and p['updates']==4700
    for n,h in q['source_SHA'].items():assert digest(P(setup['source_root'])/n)==h
    assert all(receipt['versions'].get(k)==v for k,v in p['runtime_versions'].items()) and p['asset_SHA']==receipt['assets_SHA']
    expected=json.loads((W/'new_node_23668_runtime_pointer.json').read_bytes()); fixed=json.loads((P(expected['root'])/'canonical_dispatch_plan.json').read_bytes())
    assert receipt['versions']==fixed['runtime_versions']
    free={x:shutil.disk_usage(x+':/').free for x in ['C','D']};reference_bytes=2978140883;required_C=200*1024**2
    assert free['C']>=required_C and free['D']>=40*1024**2
    relayroot=DC.parent/'node23668_stream_relay_20261011T1210Z'
    route_original=relayroot/'remote_original/actual_stream_receipt.json'
    route=json.loads(route_original.read_bytes())
    route_exit=json.loads((relayroot/'remote_original/natural_exit.json').read_bytes())
    assert route_exit['natural_exit']==0 and route['status']=='FULL_ORIGINAL_RELEASE_STREAM_RESTORED_VERIFIED'
    assert route['all_member_SHA_CRC_unique_exact_set_passed'] and route['whole_SHA']=='e9e37223723ab723c15dc3948fbccad79f48ee0da7f6526f2613d46ae773dd90'
    assert json.loads((relayroot/'receipts/runtime23668_route_0000.json').read_bytes())['remote_digest']=='sha256:'+route['whole_SHA']
    oldgate=json.loads((W/'group5_careflow_next_gate_pointer.json').read_bytes())['gate'];assert oldgate['total_stage_budget_seconds']==10260 and oldgate['whole_stage_ends_at_original_observed_exit']
    stage_budget=4*oldgate['total_stage_budget_seconds'];saving=7200
    now=dt.datetime.now(dt.timezone.utc);assert (dt.datetime.fromisoformat(node['conservative_lease_end_UTC'])-now).total_seconds()>=stage_budget+saving
    root=DC.parent/('node23668_careflow1_formal_'+now.strftime('%Y%m%dT%H%M%SZ'));root.mkdir();payload=root/'payload';payload.mkdir();shutil.copytree(P(setup['source_root']),payload/'source')
    remote=rp['remote_root']+'/formal_careflow_fold1_'+now.strftime('%Y%m%dT%H%M%SZ')
    shutil.copyfile(W/'new_node23668_formal_supervisor_v2.py',payload/'supervisor.py');ast.parse((payload/'supervisor.py').read_text())
    shutil.copyfile(native_root/'out/actual_stage_receipt.json',payload/'native_precheck_receipt.json');shutil.copyfile(capture/'runtime_metadata/runtime_original_receipt.json',payload/'runtime_original_receipt.json')
    shutil.copyfile(W/'freeze_node23668_careflow1_v3.py',payload/'freeze_plan_original.py')
    shutil.copyfile(route_original,payload/'stream_route_original_receipt.json')
    p.update(status='GROUP5_DIRECT_TRAIN_FROZEN',execution_enabled=True,authorized_methods=['careflow'],GPU_UUID=node['GPU_UUID'],python=receipt['python'],assets_root=rp['canonical_runtime_root']+'/public',source_SHA=q['source_SHA'],runtime_versions=receipt['versions'],lease_end_UTC=node['conservative_lease_end_UTC'],stage_budget_seconds=stage_budget,saving_budget_seconds=saving,remote_required_bytes=20*1024**3,local_C_free_bytes=free['C'],local_D_free_bytes=free['D'],local_space_capture_UTC=now.isoformat(),original_capture_local_C_required_bytes=required_C,new_once_token=remote+'/careflow1_train100.once',same_method_fold_native_CPU_qualified=True,Release_range_restore_qualified=True,fresh_initial_state_qualified=True,native_precheck_receipt=remote+'/native_precheck_receipt.json',native_precheck_receipt_SHA=q['original_native_receipt_SHA'],runtime_original_receipt=remote+'/runtime_original_receipt.json',runtime_original_receipt_SHA=digest(payload/'runtime_original_receipt.json'),supervisor_SHA=digest(payload/'supervisor.py'),fresh_resume_disabled=True,queue_scope='ONLY_CAREFLOW_FOLD1_FRESH100_NO_REPEATED_NATIVE3',qualification_original_SHA=q['original_archive_SHA'],qualification_GitHub_commit=native['github']['commit'],native_precheck_GPU_is_this_node=False,new_GPU_native3_repeated=False,measured_budget=dict(original_measured_gate_seconds=10260,additional_hardware_uncertainty_factor=4,allocated_stage_seconds=stage_budget,new_GPU_actual_training_time_measured=False,projection_not_guaranteed=True,at_least2h_saving=saving),preservation_route=dict(original_local_C_maximum_bytes=3300*1024**2,original_reference_bytes=reference_bytes,one_local_original_then_remote_whole_Release_restoration=True,local_floor_bytes=200*1024**2,D_floor_bytes=40*1024**2,Release_part_bytes=512*1024**2,full_original_hash_CRC_unique_exactset_required=True,GitHub_credentials_remain_local=True,remote_whole_source_route_receipt_SHA=digest(capture/'source_route_receipt.json'),old_results_and_once_unchanged=True),GPU_capacity_MiB=22528,GPU_actual_utilization_monitoring_required=True,frozen_batch_or_epoch_changed=False)
    p['preservation_route']=dict(mode='SERVER_ORIGINAL_LOCAL_STREAM_RELAY_SERVER_WHOLE_RESTORE',remote_loopback_relay='http://127.0.0.1:18768',qualification_receipt_SHA=digest(route_original),qualified_small_original_bytes=route['bytes'],small_route_pass_not_large_model_proof=True,relay_source_SHA=digest(relayroot/'node23668_release_relay_v1.py'),remote_client_SHA=digest(relayroot/'node23668_remote_release_stream_v1.py'),local_payload_staging_bytes=0,remote_complete_original_and_restoration_required=True,local_floor_bytes=200*1024**2,D_floor_bytes=40*1024**2,Release_part_bytes=512*1024**2,full_original_hash_CRC_unique_exactset_required=True,GitHub_credentials_remain_local=True,old_results_and_once_unchanged=True)
    write(payload/'plan.json',p);plan_sha=digest(payload/'plan.json')
    proof=publish_complete_small(root,payload,'node23668-careflow1-formal-frozen-plan')
    pointer=dict(root=str(root),remote=remote,plan_SHA=plan_sha,supervisor_SHA=p['supervisor_SHA'],payload=str(payload),lease_end_UTC=p['lease_end_UTC'],source_SHA=p['source_SHA'],source_root_existing_remote=rp['remote_root']+'/source_route_actual/source',exact_runtime=runtime_proof,proof=proof,stage_budget_seconds=stage_budget,saving_budget_seconds=saving,training_dispatched=False,once_consumed=False,samefold_original_native3_not_repeated=True,new_GPU_timing_not_yet_measured=True)
    write(W/'new_node23668_formal_pointer.json',pointer)
    state=json.loads((DC/'D_current_research_state.json').read_bytes());state['latest_new_node23668_exact_runtime']=runtime_proof;state['latest_new_node23668_formal']=pointer;sync(state)
    print(json.dumps(dict(runtime_root=str(runtime_root),formal_root=str(root),remote=remote,plan_SHA=plan_sha,payload=str(payload),stage_budget_seconds=stage_budget,saving_budget_seconds=saving,training_dispatched=False)))
if __name__=='__main__':main()
