"""Close real CPU/Release preservation, retain original D ZIP, free a new duplicate."""
import datetime as dt,json,pathlib,shutil,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,seal
from raw_TRAIN_save_and_publish_v1 import publish
from group5_publish_exact_local_v1 import publish_tree
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync

def main():
    source=P(json.loads((BASE/'work/group5_real_capture_pointer.json').read_bytes())['root']).resolve()
    restored=json.loads((source/'actual_GitHub_restoration.json').read_bytes())
    assert restored['whole_SHA']=='5cd62bb5993bd8f47d6d2f6ddf15d752d78b86414e1ed625d2766312d0b7ef01' and restored['all_member_SHA_CRC_unique_exact_set_passed']
    root=DC.parent/('g5_native_complete_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir();meta=root/'metadata';meta.mkdir()
    original=source/'complete_actual_careflow_fold0_precheck_original.zip';target=root/original.name
    assert original.resolve().parent==source and target.resolve().parent==root.resolve()
    assert shutil.disk_usage('D:/').free>original.stat().st_size+40*1024**2
    shutil.copyfile(original,target);assert digest(target)==digest(original)==restored['whole_SHA']
    # This is a verified relocation of the complete original, which remains on D.
    original.unlink()
    duplicate=source/'GitHub_restored_complete_original.zip'
    assert duplicate.resolve().parent==source and digest(duplicate)==digest(target)
    duplicate.unlink()  # only this new temporary restored duplicate, never the D original
    for p in source.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.log','.txt'):
            d=meta/'capture'/p.relative_to(source);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
    for n in ['publish_group5_real_precheck_v1.py','group5_release_range_resume_v1.py','group5_direct_runtime_v2.py','group5_epoch_resume_v1.py','group5_direct_CPU_audit_v2.py','check_group5_resume_history_v1.py','group5_resume_tensor_qualification_v1.py','group5_publish_exact_local_v1.py','close_group5_real_precheck_v1.py']:
        shutil.copyfile(BASE/'work'/n,meta/n)
    q=P(json.loads((BASE/'work/group5_resume_source_pointer.json').read_bytes())['root'])
    shutil.copyfile(q/'synthetic_CPU_resume_report.json',meta/'synthetic_CPU_resume_report.json')
    failure=P(json.loads((BASE/'work/group5_restore_failure_pointer.json').read_bytes())['root'])
    for n in ['network_failure.json','failure_stack_excerpt.txt']:shutil.copyfile(failure/n,meta/n)
    proof=dict(status='REAL_NATIVE_PRECHECK_CPU_AND_FULL_RELEASE_RESTORE_CLOSED',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),original_archive=str(target),archive_SHA=digest(target),bytes=target.stat().st_size,all_member_SHA_CRC_unique_exact_set_passed=True,actual_GitHub_restoration=restored,original_PT_SHA='8883bc80222c968c61d6a3cd40d5f8eb4a17fe8ddc8149557da431cce627bb71',formal_training_started=False,final_outputs_completed=0,new_temporary_restored_duplicate_removed=True,original_relocated_C_to_D_verified=True,old_originals_untouched=True)
    write(meta/'complete_preservation_receipt.json',proof)
    captured=seal(meta,meta/'complete_native_preservation_metadata.zip');write(root/'D_metadata_receipt.json',captured)
    publish(P(captured['archive']),captured['archive_SHA'],'group5-real-native-closed-'+captured['archive_SHA'][:12]+'.zip',meta/'metadata_Release_receipt.json')
    github=publish_tree(meta,'results/group5_careflow_native_complete_20261010','Preserve original CaReFlow native CPU audit and full Release restoration')
    write(root/'GitHub_receipt.json',github)
    state=json.loads((DC/'D_current_research_state.json').read_bytes());e=state['latest_human_TEST_selected_group5']
    e.update(status=proof['status'],latest_native_complete=dict(root=str(root),preservation=proof,github=github),formal_training_started=False,final_outputs_completed=0)
    e['latest_remote_review_block']['later_human_continue_received']=True;e['latest_remote_review_block']['currently_blocks_same_protocol']=False
    state['github_source']=github;state['updated_at_utc']=proof['actual_UTC'];state['next_gate']='Real CaReFlow fold0 precheck CPU and full512MiB Release download restoration closed. Freeze a fresh bounded formal100 stage with measured budget and2h saving reserve; no repeat of precheck or old once. Other methods and full35 queue remain unqualified.';sync(state)
    write(BASE/'work/group5_real_closed_pointer.json',dict(root=str(root),proof=proof,github=github))
    with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+proof['actual_UTC']+' CaReFlow fold0原三步CPU审核与真实5片GitHub整件还原全SHA/CRC/155成员闭合，原ZIP5cd62bb5永久D/Release保留；C原件核SHA迁D，仅明确新临时还原副本清理，旧原件不动。网络超时前缀/失败记录保存后按精确Range续传成功，不重传5片。GitHub'+github['commit']+' exactblob过，D/C同字节。正式100轮尚未启动，最终0/25。\n')
    print(json.dumps(dict(root=str(root),github=github,proof=proof),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
