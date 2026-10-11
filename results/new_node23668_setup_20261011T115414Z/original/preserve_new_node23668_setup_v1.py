"""Save new-node originals and unchanged samefold source; never run science."""
import ast,datetime as dt,hashlib,json,pathlib,shutil,sys,zipfile
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;W=BASE/'work';sys.path.insert(0,str(W))
from group5_release_transport_v1 import digest,write,seal,verify_zip
from raw_TRAIN_save_and_publish_v1 import publish
from group5_publish_exact_local_v1 import publish_tree
from publish_test_selected_group5_preparation_v1 import DC,sync
from prepare_github_source_upload import sanitize
def main():
    tag=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ');root=DC.parent/('new_node23668_setup_'+tag);root.mkdir();original=root/'original';original.mkdir()
    node=json.loads((W/'new_node_23668_active.json').read_bytes());runtime=json.loads((W/'new_node_23668_runtime_pointer.json').read_bytes())
    native=json.loads((W/'group5_careflow1_native_capture_pointer.json').read_bytes());native_root=P(native['preservation_root'])/'metadata/original'
    native_plan=json.loads((native_root/'plan.json').read_bytes());q=native['qualification']
    assert q['CPU_original_exit']==q['native_original_exit']==0 and q['original_once_not_repeated']
    assert q['all_member_SHA_CRC_unique_exact_set_passed'] and q['actual_GitHub_restoration']['all_member_SHA_CRC_unique_exact_set_passed']
    assert digest(native_root/'out/actual_stage_receipt.json')==q['original_native_receipt_SHA']
    pipeline=json.loads((W/'group5_pipeline_pointer.json').read_bytes());archive=DC.parent/'g5pipe_20261010T032228Z/source/source.zip'
    assert digest(archive)=='f37d3ebfcb26b9f6a145ae79a8c0fc62c961646b198d0858f9289c5ccf58b666'
    source=original/'careflow_source';source.mkdir();members=[]
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        for name,wanted in q['source_SHA'].items():
            member='careflow/'+name
            if member in z.namelist():
                data=z.read(member);info=z.getinfo(member);origin=dict(archive_SHA=digest(archive),member=member,member_CRC=info.CRC,bytes=info.file_size)
            else:
                path=W/name;data=path.read_bytes();origin=dict(original_path=str(path),bytes=len(data))
            assert hashlib.sha256(data).hexdigest()==wanted
            (source/name).write_bytes(data);members.append(dict(name=name,SHA=wanted,**origin))
            if name.endswith('.py'):ast.parse(data)
    for n in ['plan.json','independent_CPU_audit.json','wrapper_exit.json']:
        shutil.copyfile(native_root/n,original/('qualified_native_'+n))
    shutil.copyfile(native_root/'out/actual_stage_receipt.json',original/'qualified_native_actual_stage_receipt.json')
    write(original/'careflow_source_inventory.json',members)
    shutil.copytree(W/'new_node_23668_preflight',original/'new_node_preflight')
    shutil.copytree(P(runtime['root']),original/'runtime_prepared')
    for n in ['new_node_23668_active.json','new_node_23668_pointer.json','new_node_23668_runtime_pointer.json','prepare_new_node_23668_runtime.py','prepare_new_node_23668_runtime_dispatch.py','prepare_new_node_23668_canonical_runtime.py','node23668_release_relay_v1.py','node23668_remote_release_stream_v1.py','preserve_new_node23668_setup_v1.py']:
        shutil.copyfile(W/n,original/n)
    status=dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='NEW_NODE_IDENTITY_AND_RUNTIME_PREPARATION_SAVED',new_endpoint='connect.cqa1.seetacloud.com:23668',GPU_UUID=node['GPU_UUID'],GPU='RTX 2080 Ti 22528 MiB',human_confirmed_lease_hours=30,conservative_lease_end_UTC=node['conservative_lease_end_UTC'],platform_expiry_verified=False,cgroup_CPU_cores=12,cgroup_RAM_bytes=48318382080,first_runtime_attempt_natural_exit=1,first_runtime_error='Official download-r2 mirror HTTP 403',first_runtime_failure_capture_SHA='00d6e97beb50a2a5e800aa0adc20df3c09c8b4fdc7326e2f755db8e792faf8d2',canonical_official_runtime_attempt='ACTUALLY_DISPATCHED_DOWNLOAD_IN_PROGRESS',canonical_wrapper_PID=1895,canonical_plan_SHA=runtime['canonical_dispatch_plan_SHA'],training_dispatched=False,new_task_once_consumed=False,new_scientific_arrays_decoded=False,method_fold_predictions_preserved=1,outer_scores_computed=0,old_completed_once_unchanged=True,expired_old_nodes_not_connected=True,relay_code='SOURCE_ONLY_NOT_EXECUTED_OR_QUALIFIED',saving_route_pending='Fresh C space currently supports one ~3GB original; full public restoration can be verified remotely; concrete route must qualify before train')
    write(original/'STATUS.json',status)
    # A release is an exact complete original only when no secret redaction is needed.
    for p in original.rglob('*'):
        if p.is_file() and p.suffix in ['.py','.json','.md','.txt','.log']:
            b,changes=sanitize(p.read_bytes())
            assert b==p.read_bytes() and not changes, 'Original requires separate private preservation: '+str(p)
    proof=seal(original,original/'complete_new_node_setup_original.zip');write(root/'D_preservation_receipt.json',proof)
    publish(P(proof['archive']),proof['archive_SHA'],'new-node23668-setup-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
    publication=json.loads((root/'Release_receipt.json').read_bytes());restored=root/'GitHub_restored_setup_original.zip'
    import urllib.request
    with urllib.request.urlopen(publication['url'],timeout=180) as response,restored.open('xb') as out:shutil.copyfileobj(response,out,1024**2)
    assert digest(restored)==proof['archive_SHA'];restoration=verify_zip(restored);write(root/'actual_GitHub_restoration.json',dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),SHA=digest(restored),**restoration))
    public=root/'text_publication';public.mkdir()
    for p in original.rglob('*'):
        if p.is_file() and p.suffix in ['.py','.json','.md','.txt','.log']:
            target=public/'original'/p.relative_to(original);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for n in ['D_preservation_receipt.json','Release_receipt.json','actual_GitHub_restoration.json']:shutil.copyfile(root/n,public/n)
    github=publish_tree(public,'results/new_node23668_setup_'+tag,'Preserve new GPU identity, runtime HTTP failure and exact source preparation')
    write(root/'GitHub_source_receipt.json',github)
    pointer=dict(root=str(root),status=status,proof=proof,Release=publication,GitHub=github,actual_GitHub_restoration=restoration,source_root=str(source))
    write(W/'new_node23668_setup_pointer.json',pointer)
    state=json.loads((DC/'D_current_research_state.json').read_bytes());state['latest_new_node23668']=pointer;sync(state)
    summary='\n\n最新 '+status['actual_UTC']+'：人类新节点23668已直接登录，RTX2080Ti/22GiB、12CPU/45GiB容器配额，GPU空闲；30小时人类租期保守末端 '+status['conservative_lease_end_UTC']+'，非平台确认。原官方镜像HTTP403自然1原件已保存；官方主地址固定Torch安装包实际下载进行中，尚未科学任务/新once/训练/推理/评分。旧节点不连接；1/25未评分预测、0/25评分及旧once不变。D保存 '+str(root)+'，GitHub '+github['commit']+'；正式fold1需确切runtime、fresh资源与完整保存路线通过后再冻结。\n'
    with (BASE/'outputs/研究接续状态.md').open('a',encoding='utf8') as f:f.write(summary)
    print(json.dumps(dict(root=str(root),proof=proof,GitHub=github,status=status['status'])))
if __name__=='__main__':main()
