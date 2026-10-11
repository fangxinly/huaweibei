"""Synthetic adversarial checks: no original labels/models/native GPU runtime."""
import copy, datetime as dt, importlib.util, json, pathlib, sys
P=pathlib.Path
sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw,sha,seal
from group5_test_selected_contract_v1 import validate_parent,validate_dispatch,validate_release_parts

def rejected(fn):
    try: fn()
    except (ValueError,PermissionError): return
    raise AssertionError('Forbidden operation was accepted')

def main():
    pointer=json.loads((P(__file__).parent/'test_selected_group5_pointer.json').read_bytes())
    parent=P(pointer['root']);root=parent/('contract_qualification_actual_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir()
    design=parent/'group5_design.json';assert sha(design)==pointer['design_SHA']
    spec=dict(row_ids=dict(fit=['synthetic_fit[0]'],inner=['synthetic_inner[0]'],outer=['synthetic_outer[0]']))
    good=dict(method='old_A_teacher',fold=0,historical_task_weights_used=False,
              public_pretraining_fresh_start=True,fit_ids=spec['row_ids']['fit'],
              inner_ids=spec['row_ids']['inner'],outer_labels_decoded=False,
              GitHub_original_restoration_verified=True)
    validate_parent('old_fixed_A',0,good,spec)
    for key,value in [('fold',1),('historical_task_weights_used',True),('fit_ids',['synthetic_outer[0]']),
                      ('outer_labels_decoded',True),('GitHub_original_restoration_verified',False)]:
        bad=copy.deepcopy(good);bad[key]=value
        rejected(lambda:validate_parent('old_fixed_A',0,bad,spec))
    rejected(lambda:validate_parent('old_fixed_A',0,None,spec))
    rejected(lambda:validate_parent('careflow',0,good,spec))
    now=dt.datetime.now(dt.timezone.utc)
    plan=dict(design_SHA=sha(design),execution_enabled=True,status='GROUP5_NEW_EXECUTION_QUALIFIED',
              source_synthetic_native_CPU_qualified=True,chunk_transport_restore_qualified=True,
              new_once_available=True,no_historical_task_weight_reuse=True,all_stage_parent_scope_qualified=True,
              GPU_UUID='SYNTHETIC_UUID_NOT_PHYSICAL',measured_staging_requirement_bytes=100,
              measured_remote_requirement_bytes=100,conservative_lease_end_UTC=(now+dt.timedelta(hours=4)).isoformat(),
              measured_remaining_stage_queue_seconds=3600,measured_saving_seconds=7200)
    physical=dict(actual_UTC=now.isoformat(),GPU_UUID='SYNTHETIC_UUID_NOT_PHYSICAL',compute_nonempty=False,
                  fullargv_captured=True,runtime_exact=True,assets_source_SHA_verified=True,
                  available_RAM_bytes=6*1024**3,C_free_bytes=200*1024**2,D_free_bytes=40*1024**2,
                  staging_free_bytes=100,remote_free_bytes=100)
    assert validate_dispatch(design.read_bytes(),plan,physical,'careflow',0,now)
    rejected(lambda:validate_dispatch(design.read_bytes(),json.loads(design.read_bytes()),physical,'careflow',0,now))
    for key,value in [('available_RAM_bytes',6*1024**3-1),('staging_free_bytes',99),
                      ('compute_nonempty',True),('runtime_exact',False),('GPU_UUID','WRONG_UUID'),
                      ('actual_UTC',(now-dt.timedelta(seconds=301)).isoformat())]:
        bad=copy.deepcopy(physical);bad[key]=value
        rejected(lambda:validate_dispatch(design.read_bytes(),plan,bad,'careflow',0,now))
    short=copy.deepcopy(plan);short['conservative_lease_end_UTC']=(now+dt.timedelta(seconds=10799)).isoformat()
    rejected(lambda:validate_dispatch(design.read_bytes(),short,physical,'careflow',0,now))
    manifest=dict(bytes=6,whole_SHA='synthetic',parts=[dict(name='part0',offset=0,bytes=3,SHA='a'),dict(name='part1',offset=3,bytes=3,SHA='b')])
    rows=[dict(name='part0',state='uploaded',bytes=3,digest='sha256:a'),dict(name='part1',state='uploaded',bytes=3,digest='sha256:b')]
    assert validate_release_parts(manifest,rows)
    rejected(lambda:validate_release_parts(manifest,[rows[0],rows[0]]))
    bad=copy.deepcopy(rows);bad[1]['digest']='sha256:wrong';rejected(lambda:validate_release_parts(manifest,bad))
    bad=copy.deepcopy(manifest);bad['parts'][1]['offset']=4;rejected(lambda:validate_release_parts(bad,rows))
    receipt=dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='NEW_GROUP5_SCOPE_RESOURCE_RELEASE_CONTRACT_SYNTHETIC_PASS',
        synthetic_only=True,adversarial_checks_passed=18,historical_teacher_crossfold_outer_label_unverified_parent_rejected=True,
        preparation_dispatch_rejected=True,stale_uuid_compute_RAM_space_lease_rejected=True,
        release_missing_duplicate_gap_wrong_digest_rejected=True,
        actual_native_model_or_chunk_upload_qualified=False,training_once_consumed=False,
        no_task_arrays_labels_models_or_SSH_loaded=True,design_SHA=sha(design),contract_SHA=sha(P(__file__).parent/'group5_test_selected_contract_v1.py'))
    (root/'receipt.json').write_bytes(raw(receipt))
    for name in ['check_test_selected_group5_contract_v1.py','group5_test_selected_contract_v1.py']:
        (root/name).write_bytes((P(__file__).parent/name).read_bytes())
    proof=seal(root,'complete_actual_group5_contract_synthetic.zip')
    (root/'D_preservation_receipt.json').write_bytes(raw(proof))
    print(json.dumps(dict(receipt=receipt,proof=proof)))

if __name__=='__main__':main()
