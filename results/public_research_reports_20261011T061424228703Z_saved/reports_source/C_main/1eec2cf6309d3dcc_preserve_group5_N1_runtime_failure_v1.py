"""Save exact original network failure before choosing a new transport path."""
import datetime as dt,json,pathlib,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import write,digest,seal
from group5_publish_exact_local_v1 import publish_tree
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC,sync,OUT
root=DC.parent/'g5_N1_runtime_network_failure_20261010T112954Z'
original=root/'g5_N1_public_runtime_20261010T112547Z'
raw=bytes.fromhex((BASE/'work/group5_N1_runtime_failure_inventory.hex').read_text().strip())
(root/'remote_exact_original_inventory.json').write_bytes(raw)
rows=json.loads(raw)
assert {x.relative_to(original).as_posix() for x in original.rglob('*') if x.is_file()}=={x['name'] for x in rows}
for r in rows:
 p=original/r['name'];assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256']
failed=json.loads((original/'runtime/failed_natural_exit.json').read_bytes())
assert failed['error_type']=='RemoteDisconnected' and json.loads((original/'wrapper_exit.json').read_bytes())['natural_exit']==1
assert (original/'runtime/public_original_assets.zip').stat().st_size==0
receipt=dict(status='N1_PUBLIC_RUNTIME_NETWORK_FAILURE_ORIGINAL_SAVED',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
 original_failure=failed,original_remote='/data/coding/g5_N1_public_runtime_20261010T112547Z',all_captured_original_bytes_matched_remote_SHA=True,
 members=len(rows),new_approval_rejection=False,new_training_inference_score=0,models_created=0,assets_zip_bytes=0,
 offline_install_dispatched=False,healthy_task_stopped=False,failed_first_SCP_capture_preserved=True,
 next='Only a separately frozen unchanged-public-archive SCP transport may replace the failed HTTP download; original failure and source remain.')
write(root/'failure_preservation_receipt.json',receipt)
proof=seal(root,root/'complete_actual_N1_runtime_network_failure.zip')
pub=root.parent/('g5_N1_runtime_failure_publication_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));pub.mkdir()
write(pub/'D_receipt.json',proof)
publish(P(proof['archive']),proof['archive_SHA'],'group5-N1-runtime-network-failure-'+proof['archive_SHA'][:12]+'.zip',pub/'Release_receipt.json')
github=publish_tree(root,'results/group5_N1_runtime_network_failure_20261010','Preserve original N1 public-asset network failure before separate transport; no training dispatched')
write(pub/'GitHub_receipt.json',github)
state=json.loads((DC/'D_current_research_state.json').read_bytes());state['latest_human_TEST_selected_group5']['latest_N1_runtime_network_failure']=dict(root=str(root),receipt=receipt,preservation=proof,github=github)
state.update(updated_at_utc=receipt['actual_UTC'],github_source=github);sync(state)
with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+receipt['actual_UTC']+' N1公开资产HTTP下载RemoteDisconnected自然1，原ZIP0B、未建环境/未训练；完整原失败12文件逐字节远端SHA核对与D封存/Release/GitHub保存。非审批拒绝，可另冻原资产SCP传送，不重试原失败任务。\n')
write(BASE/'work/group5_N1_runtime_failure_pointer.json',dict(root=str(root),publication_root=str(pub),proof=proof,receipt=receipt,github=github))
print(json.dumps(dict(proof=proof,github=github)),flush=True)
