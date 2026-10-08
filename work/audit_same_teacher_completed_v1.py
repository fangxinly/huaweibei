"""Join original GPU collection, permanent D, other-node CPU receipts and captures."""
from pathlib import Path
import argparse, datetime, hashlib, json, zipfile, numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
d=a.directory;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
assert read(d/'completed_capture_audit.json')['phase']=='execute'
rows={};zips={}
for node in ['a','b','c']:
    zips[node]=zipfile.ZipFile(d/'snapshots_completed'/node/'snapshot.zip')
try:
    for fold,(node,target) in enumerate(zip(['a','b','c'],['b','c','a'])):
        r=d/node;plan=read(r/'same_teacher_collection_plan_v2.json');selected=plan['folds'][fold]
        receipt=read(r/'execute/receipt.json');original=read(r/'preservation_execute/receipt.json')
        package=read(r/'preservation_execute/independent_package_audit.json')
        cpu=r/'independent_cpu';cr=read(cpu/'cpu_preservation_receipt.json');ca=read(cpu/'cpu_array_audit.json')
        assert cr['training_node']==node and cr['target_cpu_node']==target and cr['fold']==fold
        assert cr['status']=='INDEPENDENT_OTHER_NODE_COLLECTION_ORIGINAL_SHA_CRC_PROCESS_RECEIPTS_AND_NUMPY_ARRAYS_VERIFIED'
        assert cr['package_sha256']==package['package_sha256']==original['sha256']
        assert cr['original_transport_receipt_sha256']==sha(r/'preservation_execute/receipt.json')
        assert cr['cpu_array_audit_sha256']==sha(cpu/'cpu_array_audit.json') and cr['cpu_exit_sha256']==sha(cpu/'cpu_exit.json')
        assert cr['source_sha256']==sha(Path(__file__).parent/'verify_same_teacher_collection_cpu_v1.py')
        assert cr['torch_imported'] is False and cr['cpu_full_model_forward'] is False and cr['old_full_weights_retransferred'] is False
        assert read(cpu/'cpu_exit.json')['exit_code']==0
        assert ca['status']=='ACTUAL_COLLECTION_FIT_INNER_SCALAR_ARRAYS_INDEPENDENT_AUDIT_PASSED'
        assert ca['original_gpu_receipt_sha256']==sha(r/'execute/receipt.json') and ca['fold']==fold
        assert cr['original_members']==package['members']
        prefix='group_teacher/cpu_preservation/same_teacher_collection_20261006T053835Z/'+node+'/'
        tz=zips[target];m=json.loads(tz.read('member_manifest.json'))
        for name,meta in cr['original_members'].items():
            assert m[prefix+'originals/'+name]==meta and sha(r/name)==meta['sha256']
        for name in ['cpu_preservation_receipt.json','cpu_array_audit.json','cpu_exit.json','cpu_audit.log']:
            assert tz.read(prefix+name)==(cpu/name).read_bytes()
        monitor=read(r/'completed_monitor.json')
        assert monitor['status']=='ACTUAL_COLLECTION_BOTH_PHASES_NATURAL_EXIT0_RETIRED_GPU_EMPTY' and monitor['compute']==''
        assert monitor['gpu'].split(',')[0].strip()==selected['expected_uuid']
        cprefix='group_teacher/'+plan['collection_subdirectory']+'/'
        assert zips[node].read(cprefix+'completed_monitor.json')==(r/'completed_monitor.json').read_bytes()
        assert receipt['before_tensor_sha256']==receipt['after_tensor_sha256']==selected['selected_model_tensor_sha256']
        array_records={}
        for role in ['fit','inner']:
            f=r/'execute'/(role+'_scalar_inputs.npz')
            with np.load(f,allow_pickle=False) as arr:
                assert set(arr.files)=={'row_ids','fold','mu'} and len(arr['row_ids'])==selected[role+'_rows']
                assert np.all(arr['fold']==fold) and np.isfinite(arr['mu']).all()
                array_records[role]={'rows':len(arr['row_ids']),'sha256':sha(f)}
        exit_proof=read(r/'execute_exit.json')
        rows[node]={'fold':fold,'arrays':array_records,'actual_child_pid':exit_proof['child_pid'],
            'actual_finished_utc':exit_proof['finished_utc'],'actual_exit_code':exit_proof['exit_code'],
            'seconds':receipt['seconds'],'peak_allocated_bytes':receipt['actual_peak_allocated_bytes'],
            'inner_replay_max_error':receipt['inner_strict_original_input_disk_replay_max_error'],
            'label_replacement_max_error':receipt['label_replacement_max_error'],'parameters_unchanged':True,
            'independent_cpu_node':target,'original_cpu_receipt_sha256':sha(cpu/'cpu_preservation_receipt.json'),
            'completed_capture_utc':read(d/'snapshots_completed'/node/'receipt.json')['utc']}
finally:
    for z in zips.values():z.close()
out=d/'completed_joint_audit.json';assert not out.exists()
report={'status':'THREE_SAME_FOLD_TEACHER_GPU_COLLECTIONS_D_ORIGINALS_OTHER_NODE_CPU_AND_ATOMIC_CAPTURE_JOINED_VERIFIED',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'nodes':rows,'calibration_fitted':False,
    'new_control_solver_executed':False,'new_controller_100_epochs':False,'whole_pipeline_crossfit':False,
    'new_outer_inference':False,'old_full_weights_retransferred':False,'source_sha256':sha(__file__)}
out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(report['status'])
