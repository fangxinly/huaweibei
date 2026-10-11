"""Local D joint of original GPU and independent B CPU capsules; no labels/model."""
import argparse
from pathlib import Path
from paired_final_TEST_pipeline_candidate_v1 import (
    plan_gate,load,require,sha,utc,write,original_prediction_gate,CPU_STATUS,STATES,CHECKPOINTS)

def joint(gpu, cpu, digest, method, output):
    gpu,cpu=Path(gpu),Path(cpu)
    gd,cd=(load(p/'actual_D_saved_capsule_audit.json') for p in (gpu,cpu))
    require(gd['stage']=='predict' and cd['stage']=='cpu' and gd['method']==cd['method']==method and
            gd['plan_sha256']==cd['plan_sha256']==digest and cd['node']=='B', 'Original distinct GPU/B CPU D audits')
    for p,d in ((gpu,gd),(cpu,cd)):
        require(d['status']=='ACTUAL_FINAL_TEST_COMPLETE_CAPSULE_D_SOURCE_ARGV_SHA_CRC_PASSED' and
                d['snapshot_sha256']==sha(p/'snapshot.zip') and
                d['capture_receipt_sha256']==sha(p/'capture_receipt.json') and
                d['stage_receipt_sha256']==sha(p/'run/out/actual_stage_receipt.json') and
                d['original_child_natural_exit_code']==0 and d['labels_read'] is False,
                'Physical D original capsule/audit linkage')
    plan=plan_gate(gpu/'run/source',digest)
    require(gd['node']==plan['fixed_models'][method]['node'], 'Original assigned prediction GPU')
    identity=sha(gpu/'run/source'/plan['identity_file'])
    r=original_prediction_gate(gpu/'run',plan,digest,method,identity)
    original_prediction_gate(cpu/'run/original',plan,digest,method,identity)
    c=load(cpu/'run/out/actual_stage_receipt.json')
    require(c['status']==CPU_STATUS and c['original_root']==r['root'] and
            c['original_receipt_sha256']==sha(gpu/'run/out/actual_stage_receipt.json') and
            c['original_natural_exit_sha256']==sha(gpu/'run/natural_exit.json') and
            c['prediction_sha256']==r['prediction_sha256'] and
            sha(cpu/'run/original/out/prediction.npz')==r['prediction_sha256'], 'Original CPU exact independent arrays linkage')
    result={'status':'ACTUAL_FINAL_TEST_PREDICTION_D_OTHER_CPU_CAPTURE_JOINT_PASSED',
            'actual_utc':utc(),'method':method,'labels_read':False,'state_sha256':STATES[method],
            'selected_checkpoint_sha256':CHECKPOINTS[method],'identity_sha256':identity,
            'execution_plan_sha256':digest,'prediction_sha256':r['prediction_sha256'],
            'original_child_natural_exit_code':0,'original_child_pid':gd['original_child_pid'],
            'other_CPU_child_pid':cd['original_child_pid'],'original_root':r['root'],
            'D_prediction_path':str(gpu/'run/out/prediction.npz'),
            'CPU_capture_prediction_path':str(cpu/'run/original/out/prediction.npz'),
            'GPU_D_audit_sha256':sha(gpu/'actual_D_saved_capsule_audit.json'),
            'CPU_D_audit_sha256':sha(cpu/'actual_D_saved_capsule_audit.json'),
            'joint_source_sha256':sha(__file__),'large_selected_checkpoint_is_prequalified_SHA_reference':True}
    write(output,result);return result

def main():
    p=argparse.ArgumentParser()
    for n in ('gpu','cpu','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--method',required=True)
    a=p.parse_args();print(joint(a.gpu,a.cpu,a.plan_sha,a.method,a.output))

if __name__=='__main__':main()
