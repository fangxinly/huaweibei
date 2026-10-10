"""Publish source/report/index as text; original numeric evidence stays in Release."""
import datetime as dt,json,pathlib,shutil,sys,traceback
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,verify_zip,seal
from group5_publish_exact_local_v1 import publish_tree
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync
p=json.loads((BASE/'work/group5_careflow0_complete_pointer.json').read_bytes());root=P(p['preservation_root']);meta=root/'metadata';capture=P(p['root'])
model_restore=json.loads((root/'actual_GitHub_restoration.json').read_bytes());metadata_restore=json.loads((root/'publication_continuation/actual_metadata_GitHub_restoration.json').read_bytes())
proof=json.loads((root/'D_receipt.json').read_bytes());receipt=json.loads((meta/'complete_preservation_receipt.json').read_bytes())
assert model_restore['whole_SHA']==p['expected_SHA'] and metadata_restore['whole_SHA']==proof['archive_SHA']
assert model_restore['all_member_SHA_CRC_unique_exact_set_passed'] and metadata_restore['all_member_SHA_CRC_unique_exact_set_passed']
assert digest(P(proof['archive']))==proof['archive_SHA'] and verify_zip(P(proof['archive']))['members']==280
failure=dict(observed_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),stage='v2_text_publication',original_process_exit=1,
 error_type='UnicodeDecodeError',original_error="UnicodeDecodeError: 'utf-8' codec can't decode byte 0x93 in position 0: invalid start byte; decoding with 'utf-8-sig' codec failed",
 source='group5_publish_exact_local_v1.py line18 sanitize(original); prepare_github_source_upload.py line17 utf-8-sig decode',
 meaning='Text publisher received original numeric NPY/NPZ evidence. Complete original binaries and their actual Release restoration remain intact.',
 new_approval_rejection=False,model_upload_or_training_repeated=False)
write(root/'publication_binary_text_failure_original.json',failure)
files=[f for f in root.rglob('*') if f.is_file()];folder=root/'text_publication';folder.mkdir();records=[];binary=[]
for f in sorted(files):
 rel=f.relative_to(root)
 if f.suffix in ('.zip','.npy','.npz'):
  binary.append(dict(path=rel.as_posix(),bytes=f.stat().st_size,SHA=digest(f)));continue
 raw=f.read_bytes();raw.decode('utf-8-sig');target=folder/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 records.append(dict(path=rel.as_posix(),bytes=len(raw),SHA=digest(f)))
shutil.copyfile(P(__file__),folder/P(__file__).name)
write(folder/'text_and_binary_publication_scope.json',dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
 text_sources_reports_indices=records,original_binary_evidence_in_unchanged_Release=binary,model_restore=model_restore,metadata_restore=metadata_restore,
 original_scores_computed=False,training_and_model_upload_not_repeated=True))
control=root/'text_publication_closure';control.mkdir();shutil.copyfile(P(__file__),control/P(__file__).name);write(control/'publication_scope.json',json.loads((folder/'text_and_binary_publication_scope.json').read_bytes()))
closure=seal(control,control/'complete_text_publication_closure.zip');write(root/'text_publication_closure_D_receipt.json',closure)
publish(P(closure['archive']),closure['archive_SHA'],'group5-careflow0-text-closure-'+closure['archive_SHA'][:12]+'.zip',root/'text_publication_closure_Release_receipt.json')
github=publish_tree(folder,'results/group5_careflow_fold0_complete_20261010','Preserve complete CaReFlow fold0 original100 CPU audit and model/metadata full restoration; source reports indices only, outer unscored')
write(root/'GitHub_receipt.json',github)
state=json.loads((DC/'D_current_research_state.json').read_bytes());g=state['latest_human_TEST_selected_group5']
g.update(status=receipt['status'],latest_careflow0_formal_complete=dict(root=str(root),receipt=receipt,preservation=proof,closure=closure,github=github),
 final_outputs_completed=0,method_fold_predictions_preserved=1,formal_stage_running=False,outer_scores_computed=False)
g['retired_SSH_ids']=sorted(set(g.get('retired_SSH_ids',[])+[23105,45985,10764]))
state.update(updated_at_utc=github['actual_UTC'],github_source=github,current_research_execution_blocker='No new approval rejection. Formal0 original100 natural0, independent CPU0, complete model and metadata Release restoration closed. Remaining stages need measured fresh lease qualification.',
 next_gate='Preserve native1 original and N1 runtime metadata. Current lease cannot fit native projected FIT plus actual nonFIT overhead and at least2h saving for another full careflow stage; no second training dispatch. No OUTER scoring until all25 originals.')
sync(state)
with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+github['actual_UTC']+' CaReFlow fold0正式100轮4700更新自然0，INNER严格最早选81，独立CPU0；完整原ZIP'+p['expected_SHA']+' 265成员与GitHub六片整件还原闭合。原12.4MB保存回执改用既定分片并整件还原过；小文件大小和二进制误入文本发布器两次原失败保留，无重跑训练/重传模型。源码/报告/index进Git、数值原件进Release。1/25未评分预测保存、五项评分0/25，GitHub'+github['commit']+'。\n')
for n in ['GitHub_restored_complete_original.zip','complete_actual_careflow_fold0_train100_original.zip']:
 f=capture/n;assert f.resolve().parent==capture.resolve() and digest(f)==p['expected_SHA'];f.unlink()
p.update(closed_receipt=receipt,github=github,publication_continuation=closure,restored_temporary_duplicate_removed=True,
 new_downloaded_temporary_transport_copy_removed=True,remote_and_GitHub_complete_originals_preserved=True)
write(BASE/'work/group5_careflow0_complete_pointer.json',p)
print(json.dumps(dict(root=str(root),github=github,status=receipt['status'])),flush=True)
