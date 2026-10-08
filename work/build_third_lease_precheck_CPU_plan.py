import hashlib,json,shutil
from pathlib import Path
local=Path('work/weak_retention_pilot40_20261007T143015Z')
base=Path('D:/CodexBackups/selective_flow_20261003_1105/weak_retention_pilot40_20261007T143015Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.loads((base/'actual_A_precheck_D_audit.json').read_text())
plan=dict(operator_SHA=sha('work/dispatch_third_lease_original_cpu.py'),
    bundle='/data/coding/weak_retention_pilot40_source_20261007T143015Z',
    python='/data/coding/multimodal_flow_public_20261007T141430Z/.venv/bin/python',
    protocol_name='precheck_execution_protocol.json',protocol_SHA=audit['protocol_SHA'],
    original_capsule='/data/coding/weak_retention_precheck_B_original_20261007T143015Z/A_complete_capture.zip',
    original_capsule_SHA=audit['capsule_SHA'],checkpoint_SHA=audit['whole_checkpoint_SHA'],
    original='/data/coding/weak_retention_precheck_B_original_20261007T143015Z',
    CPU_root='/data/coding/weak_retention_precheck_B_CPU_20261007T143015Z',
    CPU_execution='/data/coding/weak_retention_precheck_B_CPU_execution_20261007T143015Z',
    CPU_capture='/data/coding/weak_retention_precheck_B_CPU_capture_20261007T143015Z',
    B_UUID='GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f')
(local/'precheck_CPU_operator_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
shutil.copy2(local/'precheck_CPU_operator_plan.json',base/'precheck_CPU_operator_plan.json')
shutil.copy2('work/dispatch_third_lease_original_cpu.py',base/'dispatch_third_lease_original_cpu.py')
print(json.dumps(dict(operator_SHA=plan['operator_SHA'],plan_SHA=sha(local/'precheck_CPU_operator_plan.json'))))
