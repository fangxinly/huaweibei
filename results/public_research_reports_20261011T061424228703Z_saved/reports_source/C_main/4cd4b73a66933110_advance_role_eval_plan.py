"""Advance one immutable evaluation stage only after complete actual D receipts."""
import argparse, hashlib, json, shutil
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, v):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(v, f, ensure_ascii=False, indent=2)

def qualification(folder):
    receipt=read(folder/'actual_capture_receipt.json')
    verdict=read(folder/'actual_D_verification.json')
    assert receipt['child_natural_exit']==verdict['natural_exit']==0
    assert receipt['archive_SHA']==verdict['archive_SHA']==sha(folder/'complete_actual_capture.zip')
    assert verdict['receipt_SHA']==sha(folder/'actual_capture_receipt.json')
    assert all(verdict[k] for k in ('source_and_argv_passed','physical_original_not_local_capture','all_original_member_SHA','ZIPCRC','unique'))
    return dict(D_folder=str(folder), receipt_SHA=verdict['receipt_SHA'], archive_SHA=verdict['archive_SHA'], PID=verdict['PID'], natural_exit=0, complete_original_SHA_CRC_unique=True)

def run(stage):
    pointer=read(Path(__file__).parent/'anchored_roles_pointer.json')
    source=Path(pointer['source']); backup=Path(pointer['D'])
    original=source/'official_roles_protocol.json';assert sha(original)==pointer['plan_SHA']
    p=read(original);infer=backup/'A_infer';q=qualification(infer)
    result=read(infer/'extracted/out/prediction_result.json')
    assert result['status']=='FIXED_INCREMENT_VAL229_TEST685_ZERO_LABEL_FULL_ORIGINAL_INFERENCE_COMPLETE'
    cache=infer/'extracted/out/original_source_states.npz';pred=infer/'extracted/out/fixed_VAL_TEST_predictions.npz'
    assert sha(cache)==result['cache_SHA'] and sha(pred)==result['prediction_SHA']
    p['allowed_stages']=[stage];p['role_cache_reference']={'SHA':sha(cache),'D_path':str(cache)}
    p['role_prediction_reference']={'SHA':sha(pred),'D_path':str(pred)}
    p['role_inference_D_reference']=q
    if stage=='score':
        audit=backup/'B_audit';aq=qualification(audit);ar=read(audit/'extracted/out/CPU_role_replay_result.json')
        assert ar['status']=='VAL_TEST_ORIGINAL_SOURCE_FULL_FLOW_MESSAGE_CPU_REPLAY_COMPLETE'
        assert ar['prediction_SHA']==sha(pred) and ar['parameters_buffers_RNG_unchanged'] and ar['labels_not_indexed']
        assert max(v for role in ar['errors'].values() for v in role.values())<=p['CPU_replay_tolerance']
        p['original_D_CPU_score_gate']={'A_inference':q,'B_original_CPU_replay':aq,'CPU_errors':ar['errors']}
    name='official_roles_'+stage+'_protocol.json';path=source/name;write(path,p)
    shutil.copy2(path,backup/name);assert sha(path)==sha(backup/name)
    record=dict(stage=stage,plan=str(path),plan_SHA=sha(path),D_plan=str(backup/name),cache=str(cache),prediction=str(pred),prediction_SHA=sha(pred),cache_SHA=sha(cache),inference_D=q)
    write(backup/(stage+'_plan_freeze_receipt.json'),record);print(json.dumps(record,ensure_ascii=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['audit','score']);run(a.parse_args().stage)
