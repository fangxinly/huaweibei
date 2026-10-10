"""Preserve only new runtime metadata, referencing already published public assets."""
import datetime as dt,json,pathlib,shutil,sys,zipfile
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,verify_zip,write
from raw_TRAIN_save_and_publish_v1 import publish
from group5_publish_exact_local_v1 import publish_tree
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync
p=json.loads((BASE/'work/group5_N1_runtime_small_pointer.json').read_bytes());root=P(p['root']);proof=p['proof'];original=root/P(proof['archive']).name
assert digest(original)==proof['archive_SHA'] and original.stat().st_size==proof['bytes']
verified=verify_zip(original);assert verified['members']==26
meta=root/'original';meta.mkdir()
with zipfile.ZipFile(original) as z:
 for n in z.namelist():
  target=(meta/n).resolve();assert target.is_relative_to(meta.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
r=json.loads((meta/'runtime/runtime_original_receipt.json').read_bytes());plan=json.loads((meta/'plan.json').read_bytes())
assert r['status']=='GROUP5_BATCH5_EXACT_PUBLIC_RUNTIME_RESTORED' and r['versions']==plan['runtime_versions']
assert r['plan_SHA']==digest(meta/'plan.json') and r['source_SHA']==digest(meta/'group5_public_runtime_prestaged_v2.py')==plan['source_SHA']
assert r['assets_SHA']==plan['asset_SHA'] and r['all_public_original_SHA_CRC_unique_exact_set_passed']
assert json.loads((meta/'wrapper_exit.json').read_bytes())['natural_exit']==json.loads((meta/'runtime/natural_exit.json').read_bytes())['exit']==0
assert not r['global_environment_changed'] and not r['package_index_contacted'] and r['models_created']==r['new_training_inference_scoring']==0
receipt=dict(status='N1_EXACT_PUBLIC_RUNTIME_SCP_RESTORED_ORIGINAL_METADATA_PRESERVED',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
 original_runtime=r,local_complete_metadata_SHA=proof['archive_SHA'],local_complete_metadata_bytes=proof['bytes'],**verified,
 no_model_or_scientific_execution=True,first_redundant_public_archive_capture_not_republished=True,
 first_redundant_capture_preserved_remote=json.loads((meta/'first_redundant_capture_original_receipt.json').read_bytes()),
 old_common_and_wheels_not_reuploaded_or_recompressed_for_publication=True)
write(root/'qualification_receipt.json',receipt);shutil.copyfile(P(__file__),root/P(__file__).name)
publish(original,proof['archive_SHA'],'group5-N1-runtime-complete-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
github=publish_tree(root,'results/group5_N1_public_runtime_complete_20261010','Preserve exact N1 isolated offline runtime and original metadata; reference existing public archives, no training')
write(root/'GitHub_receipt.json',github)
state=json.loads((DC/'D_current_research_state.json').read_bytes());state['latest_human_TEST_selected_group5']['latest_N1_public_runtime_complete']=dict(root=str(root),receipt=receipt,proof=proof,github=github)
state.update(updated_at_utc=github['actual_UTC'],github_source=github);sync(state)
with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+github['actual_UTC']+' N1独立公开运行环境UTC11:41:43自然0；同原资产SCP字节SHA/全部成员CRC与25包版本过，仅隔离离线安装，模型/数组标签/训练/推理/评分0。原HTTP失败完整保留。首个元数据包误包含已公开资产副本，原包保留不重传；新26成员小元数据ZIP'+proof['archive_SHA']+' D/Release/GitHub保存。\n')
p.update(qualification=receipt,github=github);write(BASE/'work/group5_N1_runtime_small_pointer.json',p)
print(json.dumps(dict(root=str(root),github=github,status=receipt['status'])),flush=True)
