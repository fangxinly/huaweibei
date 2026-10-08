import argparse,datetime,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path('work/innovation_v1').resolve()))
from contract import read,write,sha
p=argparse.ArgumentParser();p.add_argument('--clock',required=True);a=p.parse_args()
current=read('work/innovation_v1_current.json');bundle=Path(current['bundle']);D=Path('D:/CodexBackups/selective_flow_20261003_1105/innovation_v1_actual_20261008T014426Z')
assert sha(bundle/'protocol.json')==current['protocol_SHA'];plan=read(bundle/'protocol.json')
native={}
for node in ('A','B'):
    root=D/(node+'_native');v=read(root/'actual_D_verification.json');check=read(root/'extracted/out/actual_native_check.json')
    assert v['natural_exit']==0 and all(check['tests'].values()) and check['real_data']==False
    native[node]=dict(archive_SHA=v['archive_SHA'],D_verification_SHA=sha(root/'actual_D_verification.json'),native_tests=check['tests'])
c=D/'A_cache';v=read(c/'actual_D_verification.json');r=read(c/'extracted/out/actual_cache_receipt.json');cache=c/'extracted/out/public_frozen_features.npz'
assert v['natural_exit']==0 and sha(cache)==r['cache_SHA'] and r['parameters_and_RNG_unchanged'] and not r['label_scalars_indexed']
capacity=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),D_free_bytes=shutil.disk_usage('D:/').free,new_artifacts_upper_bytes=300000000,margin_bytes=1000000000,no_additional_deletion=True)
assert capacity['D_free_bytes']>capacity['new_artifacts_upper_bytes']+capacity['margin_bytes']
plan.update(status='SINGLE_MECHANISM_REAL_TRAIN_EXECUTION_FROZEN',actualclock_execution_freeze_UTC=a.clock,real_train_enabled=True,allowed_stages=['train'],native_D_and_other_CPU_reference=native,cache_D_reference=dict(D=str(c),archive_SHA=v['archive_SHA'],D_verification_SHA=sha(c/'actual_D_verification.json')),cache_reference=dict(SHA=r['cache_SHA'],remote_path='/data/coding/innovation_v1_cache_A_20261008T014426Z/out/public_frozen_features.npz'),local_storage_reservation=capacity,parent_protocol_SHA=current['protocol_SHA'])
out=bundle/'execution_protocol.json';assert not out.exists();write(out,plan);shutil.copy2(out,D/'execution_protocol.json');write(D/'execution_freeze_actual.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_SHA=sha(out),source_parent_SHA=current['protocol_SHA'],real_native_and_cache_D_gates_passed=True,capacity=capacity))
current.update(status='EXECUTION_FROZEN_NOT_YET_DISPATCHED',execution_plan=str(out),execution_plan_SHA=sha(out),D_actual=str(D));write('work/innovation_v1_current.json',current);print(current)
