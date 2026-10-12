import datetime as dt,json,pathlib,shutil,sys,zipfile
P=pathlib.Path;B=P(__file__).resolve().parent.parent;W=B/'work';sys.path.insert(0,str(W))
from group5_release_transport_v1 import digest,write,verify_zip
from freeze_node23668_careflow1_v4 import publish_complete_small
from publish_test_selected_group5_preparation_v1 import DC,sync
ptr=json.loads((W/'node20631_official_queue_pointer.json').read_bytes());prep=P(ptr['root'])
original=B/'outputs/node20631_runtime_direct_staged_20261012T153611Z/runtime_and_route_closed_original.zip'
assert digest(original)=='ea6183d6f4aac94e54ea4aa8002949fd0c917d0fa87f50c119b84ed216dc53c7';verify_zip(original)
now=dt.datetime.now(dt.timezone.utc);root=DC.parent/('node20631_official_queue_saved_'+now.strftime('%Y%m%dT%H%M%SZ'));root.mkdir();payload=root/'payload';payload.mkdir()
with zipfile.ZipFile(original) as z:z.extractall(payload/'runtime_and_route_original')
shutil.copyfile(original,payload/'runtime_and_route_closed_original.zip')
closed=payload/'runtime_and_route_original';runtime=json.loads((closed/'runtime_metadata/runtime_original_receipt.json').read_bytes());route=json.loads((closed/'publication_route_actual_receipt.json').read_bytes())
assert json.loads((closed/'runtime_natural_exit.json').read_bytes())['natural_exit']==0
assert json.loads((closed/'route_natural_exit.json').read_bytes())['natural_exit']==0
assert route['status']=='FULL_ORIGINAL_RELEASE_STREAM_RESTORED_VERIFIED' and route['all_member_SHA_CRC_unique_exact_set_passed']
p=json.loads((prep/'prepared_plan.json').read_bytes());assert runtime['versions']==p['runtime_versions'] and runtime['assets_SHA']==p['asset_SHA'] and runtime['GPU_UUID']==p['GPU_UUID']
assert (dt.datetime.fromisoformat(p['lease_end_UTC'])-now).total_seconds()>p['per_job_budget_seconds']+7200
shutil.copytree(prep/'source',payload/'source')
for n in ['source_origin.json','source_AST_report.json','prepared_plan.json','实验协议.md']:shutil.copyfile(prep/n,payload/n)
shutil.copyfile(W/'freeze_node20631_official_queue_v1.py',payload/'freeze_original.py')
shutil.copyfile(W/'prepare_node20631_official_queue.py',payload/'prepare_original.py')
shutil.copyfile('C:/Users/21234/.codex/automations/automation/automation.toml',payload/'automation_current_original.toml')
free={d:shutil.disk_usage(d+':/').free for d in ['C','D']};assert free['C']>200*1024**2 and free['D']>40*1024**2
p.update(execution_enabled=True,python=runtime['python'],assets_root=str(P(runtime['python']).parent.parent.parent/'public').replace('\\','/'),runtime_receipt=ptr['remote'].rsplit('/',1)[0]+'/direct_staged_runtime/runtime/metadata/runtime_original_receipt.json',runtime_receipt_SHA=digest(closed/'runtime_metadata/runtime_original_receipt.json'),runtime_and_route_original_SHA=digest(original),local_space_capture_UTC=now.isoformat(),local_C_free_bytes=free['C'],local_D_free_bytes=free['D'],preservation_relay='http://127.0.0.1:18770',small_transport_route_receipt_SHA=digest(closed/'publication_route_actual_receipt.json'),small_route_not_large_model_proof=True,frozen_UTC=now.isoformat())
assert p['assets_root'].endswith('/direct_staged_runtime/runtime/public')
write(payload/'plan.json',p);h=digest(payload/'plan.json')
proof=publish_complete_small(root,payload,'node20631-official-five-seed-frozen-plan')
ptr.update(frozen_root=str(root),payload=str(payload),plan_SHA=h,python=p['python'],assets_root=p['assets_root'],runtime_passed=True,runtime_original_capture_SHA=digest(original),preservation=proof,training_dispatched=False,once_consumed=False,lease_end_UTC=p['lease_end_UTC'],Release_relay_root=str(DC.parent/'node20631_release_relay_20261012T1540Z'),Release_remote_port=18770)
write(W/'node20631_official_queue_pointer.json',ptr)
runtime_pointer=json.loads((W/'node20631_runtime_direct_staged_pointer.json').read_bytes());runtime_pointer.update(runtime_passed=True,actual_natural_exit=0,original_receipt_SHA=p['runtime_receipt_SHA'],preservation_root=str(root));write(W/'node20631_runtime_direct_staged_pointer.json',runtime_pointer)
state=json.loads((DC/'D_current_research_state.json').read_bytes());state['latest_node20631_official_queue']=ptr;state['latest_node20631_exact_runtime']=runtime_pointer;sync(state)
print(json.dumps(dict(root=str(root),payload=str(payload),plan_SHA=h,python=p['python'],assets_root=p['assets_root'],GitHub_commit=proof['GitHub']['commit'],training_dispatched=False)))
