"""Repair only the absent pointer after metadata keyword collision; frozen source is unchanged."""
import datetime,hashlib,json
from pathlib import Path
local=Path('work/weak_retention_train40_20261007T145201Z').resolve()
dest=Path('D:/CodexBackups/selective_flow_20261003_1105/weak_retention_train40_20261007T145201Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((local/'dispatch_plan.json').read_text(encoding='utf8'))
record=json.loads((local/'freeze_record.json').read_text(encoding='utf8'))
assert sha(local/'training_bundle.zip')==plan['archive_SHA']==record['archive_SHA']
assert sha(local/'bundle/training_execution_protocol.json')==plan['protocol_SHA']==record['protocol_SHA']
for p in local.rglob('*'):
    if p.is_file():assert sha(dest/p.relative_to(local))==sha(p)
current=dict(local=str(local),D=str(dest),local_bundle=str(local/'bundle'),stamp='20261007T145201Z',**plan)
target=Path('work/third_lease_weak_training_current.json');assert not target.exists()
target.write_text(json.dumps(current,indent=2)+'\n')
failure=dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='LOCAL_POINTER_FAILURE_REPAIRED_FROZEN_TRAIN_PLAN_AND_SOURCE_UNCHANGED',
    observed_original_exit=1,error='TypeError duplicate bundle keyword while writing final local pointer',
    original_freeze_AND_D_copy_already_complete=True,remote_train_started=False,
    repair='Write absent local pointer from unchanged actual freeze/archive/dispatch SHA; do not rerun freeze builder or change training source.')
(dest/'train_freeze_pointer_failure_and_repair.json').write_text(json.dumps(failure,indent=2)+'\n')
print(json.dumps(current))
