from pathlib import Path
import subprocess,sys
for script,args in [
 ('audit_soft_snapshot_v10.py',['--directory','D:/CodexBackups/selective_flow_20261003_1105/finite_risk_snapshots_20261005T1538Z','--out','work/new_p4_checks/finite_snapshot10_prior_independent_audit.json']),
 ('audit_finite_snapshot_v1.py',['--directory','D:/CodexBackups/selective_flow_20261003_1105/finite_risk_snapshots_20261005T1538Z','--out','outputs/有限任务风险完成与失败逐节点快照独立核验.json']),
 ('audit_finite_snapshot_full_extras_v2.py',[])
]:
 subprocess.run([sys.executable,str(Path('work')/script),*args],check=True)
