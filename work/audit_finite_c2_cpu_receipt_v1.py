from pathlib import Path
import json,hashlib,datetime
root=Path(__file__).parent.parent;d=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_c2_completed_20261005/c');local=json.loads((root/'outputs/有限任务风险C2完整权重本地独立核验.json').read_text(encoding='utf-8'));m=local['manifest'];r=json.loads((d/'cpu_preservation_receipt.json').read_text())
assert r['training_node']=='c' and r['assembly_node']=='a' and r['target_cpu_node']=='b' and r['target_gpu_uuid']=='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067' and not r['cuda_initialized'] and r['full_state_tensor_count']==461
assert r['required_seven_files']==m['required_seven_files'] and r['extra_files']==m['extra_files'] and len(r['required_seven_files'])==7 and len(r['extra_files'])==12
assert r['source_sha256']==hashlib.sha256((root/'work/verify_soft_preservation_cpu_v1.py').read_bytes()).hexdigest()
out=dict(status='C2_SEVEN_PLUS_TWELVE_ORIGINAL_CPU_RECEIPT_DOWNLOADED_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipt=r,full_checkpoint_sha256=local['receipt']['full_checkpoint_sha256'],scope='C2 trainedC, fullassembly/strictoriginalinputreplayA, CPU targetB distinct from BOTH. Original CPU receipt bytes downloaded, not terminal-only. No new training, originalC1failed preserved.')
(root/'outputs/有限任务风险C2七文件异训练与组装CPU保存独立核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print('ORIGINAL_C2_CPU_RECEIPT_INDEPENDENT_AUDIT_PASS')
