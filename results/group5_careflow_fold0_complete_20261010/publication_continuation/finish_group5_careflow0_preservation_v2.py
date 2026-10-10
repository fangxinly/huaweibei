"""Finish publication of unchanged sealed metadata using the qualified range path."""
import datetime as dt,json,pathlib,shutil,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,verify_zip,seal,plan,upload,restore
from group5_publish_exact_local_v1 import publish_tree
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync
p=json.loads((BASE/'work/group5_careflow0_complete_pointer.json').read_bytes());root=P(p['preservation_root']);meta=root/'metadata';capture=P(p['root'])
proof=json.loads((root/'D_receipt.json').read_bytes());archive=P(proof['archive'])
assert digest(archive)==proof['archive_SHA'] and verify_zip(archive)['members']==280
receipt=json.loads((meta/'complete_preservation_receipt.json').read_bytes());closed=json.loads((root/'actual_GitHub_restoration.json').read_bytes())
assert closed['whole_SHA']==receipt['archive_SHA']==p['expected_SHA'] and closed['all_member_SHA_CRC_unique_exact_set_passed']
assert not receipt['outer_scores_computed']
closing=root/'publication_continuation';closing.mkdir();shutil.copyfile(P(__file__),closing/P(__file__).name)
for f in root.glob('publication_failure_*.json'):shutil.copyfile(f,closing/f.name)
write(closing/'continuation_reason.json',dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
 original_metadata_bytes=proof['bytes'],first_publisher_limit_bytes=10*1024**2,original_metadata_preserved_unchanged=True,
 failure='AssertionError: exact sealed metadata exceeds the small-publisher size limit',new_approval_rejection=False,
 new_training_inference_scoring=0,model_ranges_reuploaded=False,next='Same qualified range upload and whole download restoration for the unchanged metadata ZIP'))
ranges=plan(archive,'group5-careflow0-completion-metadata-'+proof['archive_SHA'][:12]+'.zip');write(closing/'range_manifest.json',ranges)
upload(ranges,closing/'Release_ranges_receipt.json')
assert shutil.disk_usage('D:/').free>proof['bytes']+40*1024**2
restored=restore(ranges,json.loads((closing/'Release_ranges_receipt.json').read_bytes()),root/'metadata_full_GitHub_restored_original.zip')
write(closing/'actual_metadata_GitHub_restoration.json',restored)
write(closing/'final_preservation_receipt.json',dict(status=receipt['status'],actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
 original_model_archive_SHA=p['expected_SHA'],original_model_archive_bytes=p['bytes'],original_model_members=265,
 complete_model_GitHub_restoration=closed,original_completion_metadata=proof,complete_metadata_GitHub_restoration=restored,
 training_natural_exit=0,independent_CPU_exit=0,method_fold_predictions_preserved=1,total_predictions=25,outer_scores_computed=False))
cp=seal(closing,closing/'complete_publication_continuation.zip');write(root/'publication_continuation_D_receipt.json',cp)
publish(P(cp['archive']),cp['archive_SHA'],'group5-careflow0-publication-closed-'+cp['archive_SHA'][:12]+'.zip',root/'publication_continuation_Release_receipt.json')
github=publish_tree(root,'results/group5_careflow_fold0_complete_20261010','Preserve complete CaReFlow fold0 original100 CPU audit, model and metadata full Release restoration; outer unscored')
write(root/'GitHub_receipt.json',github)
state=json.loads((DC/'D_current_research_state.json').read_bytes());g=state['latest_human_TEST_selected_group5']
g.update(status=receipt['status'],latest_careflow0_formal_complete=dict(root=str(root),receipt=receipt,preservation=proof,closure=cp,github=github),
 final_outputs_completed=0,method_fold_predictions_preserved=1,formal_stage_running=False,outer_scores_computed=False)
g['retired_SSH_ids']=sorted(set(g.get('retired_SSH_ids',[])+[23105,45985,10764]))
state.update(updated_at_utc=github['actual_UTC'],github_source=github,current_research_execution_blocker='No new approval rejection. Formal0 original natural0, CPU0 and complete model/metadata Release restoration verified. Remaining stages require their full qualification and fresh measured lease gate.',
 next_gate='Preserve native1 original and N1 runtime metadata. Current remaining lease cannot fit a new full careflow stage conservative projected FIT plus actual nonFIT overhead and at least2h saving; do not consume a training once. No OUTER scoring before all25 originals.')
sync(state)
with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+github['actual_UTC']+' CaReFlow fold0正式100轮4700更新自然0，INNER严格最早选81；独立CPU0，完整原ZIP'+p['expected_SHA']+' 265成员与GitHub六片整件下载还原闭合。12.4MB原回执超小文件发布器上限报错已保留，同原回执另走已资格分片/整件还原闭合，无重传模型/重跑训练。1/25未评分预测已保存、评分0/25；GitHub'+github['commit']+'。\n')
for n in ['GitHub_restored_complete_original.zip','complete_actual_careflow_fold0_train100_original.zip']:
 f=capture/n;assert f.resolve().parent==capture.resolve() and digest(f)==p['expected_SHA'];f.unlink()
p.update(closed_receipt=receipt,github=github,publication_continuation=cp,restored_temporary_duplicate_removed=True,
 new_downloaded_temporary_transport_copy_removed=True,remote_and_GitHub_complete_originals_preserved=True)
write(BASE/'work/group5_careflow0_complete_pointer.json',p)
print(json.dumps(dict(root=str(root),github=github,status=receipt['status'])),flush=True)
