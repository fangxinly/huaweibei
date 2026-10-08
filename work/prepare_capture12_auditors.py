from pathlib import Path
w=Path(__file__).parent
s=(w/'audit_soft_snapshot_v11.py').read_text(encoding='utf-8').replace('capture_soft_vector_v11.py','capture_soft_vector_v12.py')
(w/'audit_soft_snapshot_v12.py').write_text(s,encoding='utf-8')
s=(w/'audit_finite_snapshot_full_extras_v3.py').read_text(encoding='utf-8').replace('finite_risk_snapshots_20261005T1541Z','finite_c2_snapshots_20261005T1612Z').replace('capture_soft_vector_v11.py','capture_soft_vector_v12.py').replace('有限任务风险保存与失败修正联合快照核验.json','有限任务风险C2启动前旧保存联合快照核验.json')
(w/'audit_finite_snapshot_full_extras_v4.py').write_text(s,encoding='utf-8')
