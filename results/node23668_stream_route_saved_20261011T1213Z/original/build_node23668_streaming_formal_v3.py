"""Create new wrapper/freezer revisions; preserve previous source and failures."""
import ast,hashlib,json,pathlib
W=pathlib.Path(__file__).resolve().parent
old=(W/'new_node23668_formal_supervisor_v1.py').read_text()
new=old.replace('subprocess,sys','subprocess,sys,urllib.request')
needle="    fresh=physical(p,root)\n    if fresh['remote_free_bytes']"
assert new.count(needle)==1
new=new.replace(needle,"""    route=p['preservation_route']
    if route['mode']!='SERVER_ORIGINAL_LOCAL_STREAM_RELAY_SERVER_WHOLE_RESTORE':raise PermissionError('Unknown preservation route')
    routefile=root/'stream_route_original_receipt.json'
    if sha(routefile)!=route['qualification_receipt_SHA']:raise PermissionError('Original route qualification differs')
    qualified=json.loads(routefile.read_bytes())
    if qualified['status']!='FULL_ORIGINAL_RELEASE_STREAM_RESTORED_VERIFIED' or not qualified['all_member_SHA_CRC_unique_exact_set_passed']:raise PermissionError('Actual small whole route not verified')
    with urllib.request.urlopen(route['remote_loopback_relay']+'/health',timeout=15) as response:health=json.load(response)
    if health['status']!='LOCAL_ONLY_RELEASE_RELAY_READY' or health['repository']!='fangxinly/huaweibei':raise PermissionError('Fresh stream route unavailable')
    fresh=physical(p,root)
    if fresh['remote_free_bytes']""")
new=new.replace('Local complete original staging observation stale','Local reserved space observation stale')
new=new.replace("        fresh=physical(p,root);write(root/'CPU_fresh_physical.json',fresh)","""        if (dt.datetime.fromisoformat(p['lease_end_UTC'])-now()).total_seconds()<max(7200,p['saving_budget_seconds']):raise PermissionError('Original CPU and saving reserve absent')
        fresh=physical(p,root);write(root/'CPU_fresh_physical.json',fresh)""")
new=new.replace("    write(root/'original_range_manifest.json',ranges)","""    ranges.update(repository='fangxinly/huaweibei',tag='autonomous-flow-health-20261008',publication_authorized=True)
    write(root/'original_range_manifest.json',ranges)""")
new=new.replace('ORIGINAL_CAPTURE_READY_FOR_LOCAL_UPLOAD_AND_REMOTE_FULL_RESTORATION','ORIGINAL_CAPTURE_READY_FOR_STREAM_UPLOAD_AND_REMOTE_FULL_RESTORATION')
ast.parse(new);(W/'new_node23668_formal_supervisor_v2.py').write_text(new,newline='\n')
oldf=(W/'freeze_node23668_careflow1_v2.py').read_text()
f=oldf.replace('reference_bytes=2978140883;required_C=3300*1024**2+200*1024**2','reference_bytes=2978140883;required_C=200*1024**2')
needle="    assert free['C']>=required_C and free['D']>=40*1024**2"
assert f.count(needle)==1
f=f.replace(needle,"""    assert free['C']>=required_C and free['D']>=40*1024**2
    relayroot=DC.parent/'node23668_stream_relay_20261011T1210Z'
    route_original=relayroot/'remote_original/actual_stream_receipt.json'
    route=json.loads(route_original.read_bytes())
    route_exit=json.loads((relayroot/'remote_original/natural_exit.json').read_bytes())
    assert route_exit['natural_exit']==0 and route['status']=='FULL_ORIGINAL_RELEASE_STREAM_RESTORED_VERIFIED'
    assert route['all_member_SHA_CRC_unique_exact_set_passed'] and route['whole_SHA']=='e9e37223723ab723c15dc3948fbccad79f48ee0da7f6526f2613d46ae773dd90'
    assert json.loads((relayroot/'receipts/runtime23668_route_0000.json').read_bytes())['remote_digest']=='sha256:'+route['whole_SHA']""")
f=f.replace('new_node23668_formal_supervisor_v1.py','new_node23668_formal_supervisor_v2.py').replace('freeze_node23668_careflow1_v2.py','freeze_node23668_careflow1_v3.py')
needle="    p.update(status='GROUP5_DIRECT_TRAIN_FROZEN'"
assert f.count(needle)==1
f=f.replace(needle,"    shutil.copyfile(route_original,payload/'stream_route_original_receipt.json')\n"+needle)
needle="    write(payload/'plan.json',p);plan_sha=digest(payload/'plan.json')"
assert f.count(needle)==1
f=f.replace(needle,"""    p['preservation_route']=dict(mode='SERVER_ORIGINAL_LOCAL_STREAM_RELAY_SERVER_WHOLE_RESTORE',remote_loopback_relay='http://127.0.0.1:18768',qualification_receipt_SHA=digest(route_original),qualified_small_original_bytes=route['bytes'],small_route_pass_not_large_model_proof=True,relay_source_SHA=digest(relayroot/'node23668_release_relay_v1.py'),remote_client_SHA=digest(relayroot/'node23668_remote_release_stream_v1.py'),local_payload_staging_bytes=0,remote_complete_original_and_restoration_required=True,local_floor_bytes=200*1024**2,D_floor_bytes=40*1024**2,Release_part_bytes=512*1024**2,full_original_hash_CRC_unique_exactset_required=True,GitHub_credentials_remain_local=True,old_results_and_once_unchanged=True)
    write(payload/'plan.json',p);plan_sha=digest(payload/'plan.json')""")
ast.parse(f);(W/'freeze_node23668_careflow1_v3.py').write_text(f,newline='\n')
print(json.dumps({'new_supervisor_SHA':hashlib.sha256(new.encode()).hexdigest(),'new_freezer_SHA':hashlib.sha256(f.encode()).hexdigest(),'original_supervisor_SHA':hashlib.sha256((W/'new_node23668_formal_supervisor_v1.py').read_bytes()).hexdigest(),'scientific_source_changed':False}))
