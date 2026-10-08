import json,hashlib,zipfile,sys,shutil
from pathlib import Path
base=Path('D:/CodexBackups/selective_flow_20261003_1105');root=base/'lease_dynamic_actual_20261007T113004Z';clock=sys.argv[1]
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={'A':'GPU-53696803-875e-eec8-2231-29db63579891','B':'GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f','C':'GPU-417d3577-0525-788b-7296-0808a0f52012'}
qualified={'5a58f50c554890baa4324d49437155649a92c68032dcbc6d88138a6269e6ed31':2966458931}
prior=read(Path('outputs/正式双方fullTRAIN100完成与完整保存实际结果.json'))
for m in prior['methods'].values():
 for v in m['whole_D_files'].values():qualified[v['sha256']]=v['bytes']
result={}
for name,node in [('A','A'),('B','B'),('C','C'),('C_complete_roots_supplement','C')]:
 receipt=read(root/(name+'_dynamic_receipt.json'));archive=root/(name+'_dynamic_capture.zip');assert sha(archive)==receipt['sha256']
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  m=json.loads(z.read('dynamic_manifest.json'));p=json.loads(z.read('physical.json'))
  for n,h in m['member_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
  assert p['uuid']==expected[node] and p['node']==node and p['operator_source_sha256']=='b64dfc4a91eafca4e5da9dccb8e0bca5246b2d7f41de49b4fb63cfefb9387a0a'
  plan=json.loads(z.read('pinned_asset_runtime_plan.json'));assert p['native_runtime']==plan['runtime_exact_versions'] and p['public_asset_sha256']==plan['asset_sha256']
  assert p['fullargv'][1].endswith('/lease_dynamic_completed_refs_capture_v2.py') and p['pid']>0
  for v in m['large_original_refs']:assert not v['new_large_original_requires_D_transfer'] and qualified[v['sha256']]==v['bytes']
  result[name]=dict(node=node,actual_start_utc=m['actual_capture_start_utc'],actual_finish_utc=m['actual_capture_finish_utc'],archive_SHA=receipt['sha256'],members=len(z.namelist()),uuid=p['uuid'],compute=p['compute'],fullargv=p['fullargv'],large_original_refs=m['large_original_refs'],source_SHA=m['source_file_sha256'],free_bytes=p['free_bytes'])
report=dict(status='THREE_REAL_DYNAMIC_NODE_CAPTURES_D_SHA_CRC_UNIQUE_SOURCE_UUID_RUNTIME_COMPLETE_QUALIFIED_COMPLETED_STATE_REFS',actualclock_D_record_UTC=clock,scheduled_window_UTC='2026-10-07T11:30:00+00:00',captures=result,C_first_capture_real_but_supervisor_root_only=True,C_complete_model_roots_fresh_supplement_actual_not_backfill=True,large_originals_already_D_qualified_no_duplicate_download=True,original_workers_or_saved_states_changed=False,next_scheduled_window_UTC='2026-10-07T13:00:00+00:00',conservative_lease_end_UTC='2026-10-07T13:30:00+00:00',platform_expiry_confirmed=False,D_free_bytes=shutil.disk_usage('D:/').free)
(root/'actual_D_dynamic_joint.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
out=Path('outputs/第二租期11点30真实动态保存实际接续.json');out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
Path('outputs/第二租期11点30真实动态保存实际接续.md').write_text(f'11:30三机真实动态保存已经D SHA/ZIP CRC/唯一成员/source/fullargv/UUID/runtime联合通过。\n\n记录 {clock}。A/B/C原capture开始均11:30:29，A/B完成11:30:40，C初件11:30:31。C初件只有supervisor及预测原件，另在实际11:34左右对正确完整训练根新capture；不把目录后缀冒实际时间或回填11:30。A/B新40轮完整3GB和C原CaReFlow完整resume/best均fresh SHA匹配此前D完整资格，以SHArefs保存，无重下载。\n\n13:00实际动态保存仍待。13:30是保守执行界限，未平台核过；不续租、释放、关机、停训练或改源。\n',encoding='utf8')
print(json.dumps(dict(status=report['status'],joint_sha=sha(root/'actual_D_dynamic_joint.json'),D_free_bytes=report['D_free_bytes'])))
