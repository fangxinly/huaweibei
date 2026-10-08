"""Prepare exact train argv only after both actually saved original precheck joints pass.

Does not connect, start a task, load a model, decode arrays or read labels.
"""
from pathlib import Path
import sys, json, shutil, datetime, hashlib
sys.path.insert(0,str(Path(__file__).parent/'paired_official_fulltrain_v2_20261007T011543Z'))
from paired_fulltrain_evidence_candidate_v1 import read,sha,write,require,PRECHECK_JOINT,plan_gate

stamp=sys.argv[1]
base=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_official_prechecks_v2_actual_20261007T012116Z')
bundle=Path(__file__).parent/'paired_official_fulltrain_v2_20261007T011543Z'
plan_sha='8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb'
plan=plan_gate(bundle,plan_sha)
require(shutil.disk_usage(base).free>=plan['local_D_free_floor_bytes'],'Actual D floor before formal argv')
moment=datetime.datetime.strptime(stamp,'%Y-%m-%d %H:%M:%S UTC').replace(tzinfo=datetime.timezone.utc)
require((datetime.datetime.fromisoformat(plan['conservative_lease_end_UTC'])-moment).total_seconds()>=18000,'Declared3h execution and2h save reserve')
name=moment.strftime('%Y%m%dT%H%M%SZ')
records={}
for method in plan['methods']:
    jointfile=base/method/'actual_original_precheck_GPU_D_B_CPU_capture_joint.json'
    j=read(jointfile)
    require(j['status']==PRECHECK_JOINT and j['method']==method and j['plan_sha256']==plan_sha and
            j['whole_checkpoint_D_SHA_CRC_passed'] and not j['CPU_model_forward'] and not j['final_TEST_executed'],
            'Complete genuine original precheck joint required for BOTH methods')
    require(j['orders_sha256']==plan['orders_sha256'] and j['official_row_ID_identity_sha256']==plan['official_train_dev_ID_identity_sha256'],'Common IDs/order source')
    records[method]=(jointfile,j)
dest=Path(__file__).parent/('paired_formal_actual_dispatch_v2_'+name)
require(not dest.exists(),'Fresh physical formal dispatch directory')
dest.mkdir()
payload={'status':'BOTH_ACTUAL_PRECHECK_JOINTS_PASSED_EXACT_FORMAL_TRAIN_ARGV_READY_NOT_LAUNCHED',
         'clock_utc':stamp,'plan_sha256':plan_sha,'formal_training_launched':False,'actual_fresh_remote_gates_pending':True,
         'methods':{},'final_TEST_enabled':False,'same_method_direct_readouts_no_201_choice':True}
for method,(jointfile,j) in records.items():
    root='/data/coding/paired_fulltrain100_'+method+'_'+name
    jointname=method+'_original_precheck_GPU_D_B_CPU_capture_joint.json'
    (dest/jointname).write_bytes(jointfile.read_bytes())
    remote_joint=root+'/'+jointname
    argv=['--root',root,'--bundle','/data/coding/'+bundle.name,'--assets','/data/coding/multimodal_flow_public_20261006T1341Z',
          '--method',method,'--stage','train','--plan-sha',plan_sha,'--precheck-root',j['original_root'],
          '--precheck-joint',remote_joint,'--precheck-joint-sha',sha(jointfile)]
    argument_file=dest/(method+'_formal_child_arguments.json')
    write(argument_file,argv)
    payload['methods'][method]={'root':root,'child_arguments_file':str(argument_file),'child_arguments_sha256':sha(argument_file),
        'precheck_joint_source':str(jointfile),'precheck_joint_sha256':sha(jointfile),'remote_precheck_joint':remote_joint,
        'precheck_root':j['original_root'],'precheck_weights_or_steps_inherited':False,'fresh_empty_Adam_scheduler0_and_all_RNG_required':True}
write(dest/'exact_formal_dispatch_arguments_receipt.json',payload)
D=base.parent/dest.name
require(not D.exists(),'Fresh D exact argv target')
shutil.copytree(dest,D)
print(json.dumps({'local_directory':str(dest),'D_directory':str(D),'receipt':payload}),flush=True)
