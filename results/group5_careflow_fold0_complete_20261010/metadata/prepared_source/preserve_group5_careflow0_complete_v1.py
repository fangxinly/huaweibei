"""Preserve the exact first complete CV stage, without training or scoring."""
import argparse, ast, datetime as dt, json, pathlib, shutil, sys, traceback, zipfile
P=pathlib.Path; BASE=P(__file__).resolve().parent.parent; sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,verify_zip,seal,plan,upload,restore
from group5_publish_exact_local_v1 import publish_tree
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync

def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['publish','restore','close']);args=a.parse_args()
    p=json.loads((BASE/'work/group5_careflow0_complete_pointer.json').read_bytes())
    capture=P(p['root']);original=capture/'complete_actual_careflow_fold0_train100_original.zip'
    if args.stage=='publish':
        assert original.stat().st_size==p['bytes'] and digest(original)==p['expected_SHA']
        proof=verify_zip(original)
        assert proof['members']==265
        root=DC.parent/('g5careflow0_complete_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        root.mkdir();meta=root/'metadata';meta.mkdir()
        with zipfile.ZipFile(original) as z:
            for name in z.namelist():
                if name.endswith(('.pt','.zip','.pyc')) or '/__pycache__/' in name:continue
                target=meta/'original'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
        r=json.loads((meta/'original/out/actual_stage_receipt.json').read_bytes())
        cpu=json.loads((meta/'original/independent_CPU_audit.json').read_bytes())
        assert r['method']==cpu['method']=='careflow' and r['fold']==cpu['fold']==0
        assert r['updates']==cpu['updates']==4700 and r['best_epoch']==81
        assert cpu['checkpoint_SHA']==r['checkpoint']['SHA']
        inventory=json.loads((meta/'original/member_manifest.json').read_bytes())
        assert next(v['sha256'] for v in inventory if v['name']=='out/resume_selected_full.pt')==cpu['checkpoint_SHA']
        shutil.copyfile(capture/'remote_original_archive_receipt.json',meta/'remote_original_archive_receipt.json')
        for failure in capture.glob('publication_failure_*.json'):shutil.copyfile(failure,meta/failure.name)
        assert not r['outer_labels_decoded'] and not cpu['new_solve_fit_score_or_outer_label_decode']
        assert cpu['status']=='INDEPENDENT_CPU_ORIGINAL_DIRECT_STATE_AUDIT_PASS'
        assert json.loads((meta/'original/wrapper_exit.json').read_bytes())['natural_exit']==0
        assert json.loads((meta/'original/CPU_exit_observed.json').read_bytes())['natural_exit']==0
        write(meta/'local_original_verification.json',dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),archive_SHA=digest(original),bytes=original.stat().st_size,**proof))
        write(meta/'continuation_observations.json',dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),retired_session_ids=[45985,23105],new_SSH_session=7670,
            old_session_error='Unknown process id',not_authentication_failure=True,not_training_failure=True,
            CPU_launch_first_input_error='AssertionError from a manually transcribed auditor SHA containing one extra digit; no CPU dispatch or training execution occurred in that failed command',
            correction='Exact original plan SHA and its exact auditor source SHA matched; independent CPU process then ran once and exited0. A separate local archive verification initially rejected a 65-character manually transcribed SHA; exact remote original receipt was then copied as machine-parsed hex chunks and matched the downloaded archive SHA. No upload occurred before correction.',
            original_training_once_not_repeated=True,outer_scores_computed=False))
        source=meta/'prepared_source';source.mkdir()
        for n in ['group5_composite_components_v1.py','group5_composite_runtime_v1.py','group5_composite_CPU_audit_v1.py','group5_public_runtime_v1.py','group5_pooled_score_v1.py','check_group5_pooled_score_v1.py','preserve_group5_careflow0_complete_v1.py']:
            ast.parse((BASE/'work'/n).read_text(encoding='utf8'));shutil.copyfile(BASE/'work'/n,source/n)
        write(source/'qualification_scope.json',dict(AST_pass=True,synthetic_score_source_checks_passed=10,original_arrays_labels_or_models_loaded_by_synthetic_checks=False,
            new_composite_native_execution=False,new_score_once_consumed=False,composite_resume_qualified=False))
        public=P(json.loads((BASE/'work/group5_idle_public_runtime_pointer.json').read_bytes())['root'])
        for n in ['prepared_runtime_template.json','common_Release_receipt.json','wheels_Release_receipt.json']:
            shutil.copyfile(public/n,source/n)
        manifest=plan(original,'group5-careflow-fold0-train100-'+p['expected_SHA'][:12]+'.zip')
        write(root/'range_manifest.json',manifest)
        p.update(preservation_root=str(root),local_verification=proof);write(BASE/'work/group5_careflow0_complete_pointer.json',p)
        upload(manifest,root/'Release_ranges_receipt.json')
        print(json.dumps(dict(status='COMPLETE_ORIGINAL_RELEASE_PARTS_VERIFIED_RESTORE_PENDING',root=str(root))),flush=True)
    elif args.stage=='restore':
        root=P(p['preservation_root']);manifest=json.loads((root/'range_manifest.json').read_bytes())
        assert shutil.disk_usage('C:/').free>manifest['bytes']+200*1024**2
        result=restore(manifest,json.loads((root/'Release_ranges_receipt.json').read_bytes()),capture/'GitHub_restored_complete_original.zip')
        write(root/'actual_GitHub_restoration.json',result);print(json.dumps(result),flush=True)
    else:
        root=P(p['preservation_root']);meta=root/'metadata';closed=json.loads((root/'actual_GitHub_restoration.json').read_bytes())
        assert closed['whole_SHA']==p['expected_SHA'] and closed['all_member_SHA_CRC_unique_exact_set_passed']
        assert digest(original)==p['expected_SHA']
        for n in ['range_manifest.json','Release_ranges_receipt.json','actual_GitHub_restoration.json']:
            shutil.copyfile(root/n,meta/n)
        shutil.copyfile(P(__file__),meta/'prepared_source'/P(__file__).name)
        receipt=dict(status='CAREFLOW_FOLD0_FORMAL100_CPU_FULL_RELEASE_RESTORE_CLOSED_OUTER_UNSCORED',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
            original_archive=str(original),archive_SHA=p['expected_SHA'],bytes=p['bytes'],method='careflow',fold=0,updates=4700,best_epoch=81,
            training_original_exit=0,independent_CPU_exit=0,all_member_SHA_CRC_unique_exact_set_passed=True,actual_GitHub_restoration=closed,
            method_fold_predictions_preserved=1,total_method_fold_predictions=25,outer_scores_computed=False,all_original_once_preserved=True,
            D_stores_metadata_C_stages_large_original_GitHub_permanent=True,other24_predictions_not_completed=True,
            permanent_original_archive_remote=p['remote']+'/'+original.name,
            permanent_original_archive_GitHub_parts=json.loads((root/'Release_ranges_receipt.json').read_bytes())['parts'],
            local_download_is_new_temporary_transport_copy=True)
        write(meta/'complete_preservation_receipt.json',receipt)
        def exact(path):return dict(path=str(path),SHA=digest(path))
        entry=dict(method='careflow',fold=0,receipt=exact(meta/'original/out/actual_stage_receipt.json'),
            CPU_audit=exact(meta/'original/independent_CPU_audit.json'),restoration=exact(meta/'actual_GitHub_restoration.json'),
            stage_plan=exact(meta/'original/plan.json'),original_member_manifest=exact(meta/'original/member_manifest.json'),
            original_archive_SHA=p['expected_SHA'],original_archive_bytes=p['bytes'],prediction_member='out/OUTER_prediction_only.npz',
            prediction_path=str(meta/'original/out/OUTER_prediction_only.npz'))
        write(meta/'prepared_future_score_entry.json',entry)
        proof=seal(meta,meta/'complete_actual_careflow0_completion_metadata.zip');write(root/'D_receipt.json',proof)
        publish(P(proof['archive']),proof['archive_SHA'],'group5-careflow0-complete-metadata-'+proof['archive_SHA'][:12]+'.zip',root/'metadata_Release_receipt.json')
        github=publish_tree(meta,'results/group5_careflow_fold0_complete_20261010','Preserve complete CaReFlow fold0 100 epochs, CPU audit and full Release restoration; outer unscored')
        write(root/'GitHub_receipt.json',github)
        state=json.loads((DC/'D_current_research_state.json').read_bytes());g=state['latest_human_TEST_selected_group5']
        g.update(status=receipt['status'],latest_careflow0_formal_complete=dict(root=str(root),receipt=receipt,preservation=proof,github=github),
                 final_outputs_completed=0,method_fold_predictions_preserved=1,formal_stage_running=False,outer_scores_computed=False)
        g.setdefault('retired_SSH_ids',[]).append(23105)
        state.update(updated_at_utc=receipt['actual_UTC'],github_source=github,current_research_execution_blocker='No new approval rejection. Original formal child2302 exited0; old SSH23105 retired; new same-node SSH7670 authenticated.',
            next_gate='First complete unscored method-fold prediction preserved and CPU/full Release restore closed. Continue remaining24 method-fold predictions with fresh source/native/resource/lease/once qualification. No OUTER score until all25 exact predictions preserved. No duplicate original once.')
        sync(state)
        with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+receipt['actual_UTC']+' CaReFlow fold0正式100轮4700更新已于UTC06:24:13自然0，INNER严格最早选81；独立CPU审核自然0，完整原ZIP'+p['expected_SHA']+' 265成员全SHA/CRC/unique/exactset与GitHub分片整件还原闭合。1/25组未评分预测保存，五项评分0/25；其他24组待，原once未重跑。GitHub'+github['commit']+'，D/C同步。\n')
        # Only a newly created, byte-verified downloaded duplicate is eligible.
        duplicate=capture/'GitHub_restored_complete_original.zip'
        assert duplicate.resolve().parent==capture.resolve() and digest(duplicate)==p['expected_SHA']
        duplicate.unlink()
        # The authoritative remote original and exact GitHub original remain.
        # The user's serial GitHub-storage instruction allows this new transit
        # copy to be cleared after actual full independent restoration closes.
        assert original.resolve().parent==capture.resolve() and original.name=='complete_actual_careflow_fold0_train100_original.zip'
        assert digest(original)==closed['whole_SHA']
        original.unlink()
        p.update(closed_receipt=receipt,github=github,restored_temporary_duplicate_removed=True,new_downloaded_temporary_transport_copy_removed=True,
                 remote_and_GitHub_complete_originals_preserved=True);write(BASE/'work/group5_careflow0_complete_pointer.json',p)
        print(json.dumps(dict(root=str(root),receipt=receipt,github=github)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        q=BASE/'work/group5_careflow0_complete_pointer.json'
        if q.exists():
            p=json.loads(q.read_bytes());root=P(p.get('preservation_root',p['root']))
            write(root/('publication_failure_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'),dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),original_preserved=True))
        raise
