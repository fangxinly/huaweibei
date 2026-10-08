from pathlib import Path
import subprocess, sys, json
r=Path(__file__).resolve().parents[1]
cpu_record=r/'outputs/Jacobian三臂七文件异训练与组装节点CPU保存核验.json'
if not cpu_record.exists():
    subprocess.run([sys.executable, str(r/'work/audit_jacobian_cpu_receipts_v1.py')], check=True)
assert json.loads(cpu_record.read_text(encoding='utf-8'))['status']=='THREE_NEW_JACOBIAN_CPU_RECEIPTS_DOWNLOADED_AND_INDEPENDENTLY_AUDITED'
subprocess.run([sys.executable, str(r/'work/audit_soft_snapshot_v4.py'),
    '--directory','D:/CodexBackups/selective_flow_20261003_1105/jacobian_preservation_snapshots_20261005T1415Z',
    '--out',str(r/'outputs/Jacobian独立CPU保存后联合快照核验.json')], check=True)
