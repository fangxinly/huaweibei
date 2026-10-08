"""Qualify an original native contract capture on D without running model code."""
import argparse,hashlib,json,zipfile
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
p=argparse.ArgumentParser();p.add_argument('--node',choices=['A','B'],required=True);p.add_argument('--actualclock',required=True);a=p.parse_args()
ws=Path(__file__).resolve().parent.parent;pointer=read(ws/'work/residual_direction_v2_current.json');d=Path(pointer['D'])/(a.node+'_native');receipt=read(d/'actual_capture_receipt.json');archive=d/'complete_actual_capture.zip'
assert receipt['node']==a.node and receipt['stage']=='native' and receipt['child_natural_exit']==0
assert sha(archive)==receipt['archive_SHA'] and archive.stat().st_size==receipt['archive_bytes']
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    manifest=json.loads(z.read('member_SHA.json'))
    assert set(z.namelist())==set(manifest)|{'member_SHA.json'}
    for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    for name,h in pointer['source_sha256'].items():assert hashlib.sha256(z.read('original_source/'+name)).hexdigest()==h
    plan=json.loads(z.read('original_source/native_protocol.json'));assert hashlib.sha256(z.read('original_source/native_protocol.json')).hexdigest()==pointer['plan_SHA']
    pre=json.loads(z.read('actual_preflight.json'));exit=json.loads(z.read('natural_exit.json'));dispatch=json.loads(z.read('actual_dispatch.json'))
    assert pre['UUID']==plan['GPU_UUID'][a.node] and not pre['compute'].strip()
    assert pre['remaining_seconds']>plan['stage_budget_seconds']['native']+7200
    assert exit['natural_exit']==0 and exit['pid']==receipt['child_PID'] and exit['fullargv']==dispatch['fullargv']
    assert exit['plan_SHA']==pointer['plan_SHA'] and exit['fullargv'][1].endswith('/native_contract.py')
    result=json.loads(z.read('stdout.log'));assert result['pid']==exit['pid'] and result['fullargv']==exit['fullargv']
    assert result['status']=='SYNTHETIC_DIRECTIONAL_FLOW_AND_OOF_SELECTOR_CONTRACT_COMPLETE'
    assert result['synthetic_only'] and not result['actual_new_OOF_training'] and not result['new_VAL_or_TEST_scoring']
    target=d/'extracted';target.mkdir();z.extractall(target)
qualification=dict(actualclock_UTC=a.actualclock,node=a.node,archive_SHA=sha(archive),receipt_SHA=sha(d/'actual_capture_receipt.json'),
    source_plan_SHA=pointer['plan_SHA'],member_count=len(manifest)+1,all_original_SHA_CRC_unique=True,
    source_fullargv_PID_UUID_budget_space_runtime_verified=True,original_natural_exit=0,result=result,
    local_verification_is_not_remote_capture=True,real_crossfit_training_or_new_scores=False)
with (d/'actual_D_verification.json').open('x',encoding='utf8') as f:json.dump(qualification,f,ensure_ascii=False,indent=2)
print(json.dumps(qualification,ensure_ascii=False))
