import argparse,hashlib,json,shutil
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def qualification(folder,split=False):
    r=read(folder/'actual_capture_receipt.json');v=read(folder/('actual_split_D_C_verification.json' if split else 'actual_D_verification.json'))
    assert r['child_natural_exit']==v['natural_exit']==0 and r['archive_SHA']==v['archive_SHA'] and sha(folder/'actual_capture_receipt.json')==v['receipt_SHA']
    if not split:assert sha(folder/'complete_actual_capture.zip')==r['archive_SHA']
    return dict(D_root=str(folder),archive_SHA=r['archive_SHA'],receipt_SHA=v['receipt_SHA'],natural_exit=0,PID=r['child_PID'],complete_original_SHA_CRC_unique=True,split_D_C_complete=split)

def run(stage,clock):
    ws=Path(__file__).resolve().parent.parent;pointer=read(ws/'work/official_upgrade_pointer.json');source=Path(pointer['source']);d=Path(pointer['D']);p=read(source/'official_native_protocol.json');assert sha(source/'official_native_protocol.json')==pointer['native_plan_SHA']
    native=qualification(d/'B_native');p['native_CPU_D_qualification']=native
    if stage=='train':
        nativeout=read(d/'B_native/extracted/stdout.log');assert nativeout['finite_gradients'] and nativeout['message_contrast']['zero_donor_contrast_exact'] and nativeout['dummy_padding_permutation_state_RNG']
        assert shutil.disk_usage(d).free>=2100000000 and shutil.disk_usage(Path(pointer['C_part_backup'])).free>=1450000000
    else:
        tq=qualification(d/'A_train',True);small=d/'A_train/extracted_small';result=read(small/'out/training_result.json');assert result['status']=='CLEAN_OFFICIAL_ANCHORED_UPGRADE_TRAIN100_COMPLETE'
        m=read(small/'member_SHA.json');p['training_original_reference']=dict(archive_SHA=tq['archive_SHA'],member_SHA=m,training_plan_SHA=result['metadata']['plan_SHA'],D_C_reference=tq)
        p['selected_state_SHA']=result['metadata']['selected_state_SHA']
        if stage in ('infer','score'):
            aq=qualification(d/'B_audit');ar=read(d/'B_audit/extracted/out/audit_result.json');assert ar['status']=='OFFICIAL_UPGRADE_FULL_STATE_ADAM4000_SELECTION_AND_CPU_FLOW_COMPLETE' and ar['CPU_actual_DEV_flow_maxerror']<=1e-4
            p['training_D_B_CPU_gate']={'training':tq,'original_B_CPU':aq}
        if stage=='score':
            iq=qualification(d/'A_infer');ir=read(d/'A_infer/extracted/out/inference_result.json');assert ir['status']=='TRAIN_ONLY_SELECTED_UPGRADE_VAL_TEST_PREDICTION_COMPLETE'
            pred=d/'A_infer/extracted/out/fixed_official_VAL_TEST_prediction.npz';assert sha(pred)==ir['prediction_SHA'];p['prediction_SHA']=sha(pred);p['prediction_D_gate']=iq
    p['allowed_stages']=[stage];p['actualclock_stage_freeze_UTC']=clock
    name='official_'+stage+'_protocol.json';f=source/name;write(f,p);shutil.copy2(f,d/name);assert sha(f)==sha(d/name)
    r=dict(stage=stage,actualclock_UTC=clock,plan=str(f),plan_SHA=sha(f),D_plan=str(d/name));write(d/(stage+'_freeze_receipt.json'),r);print(json.dumps(r))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','audit','infer','score']);p.add_argument('--actualclock',required=True);a=p.parse_args();run(a.stage,a.actualclock)
