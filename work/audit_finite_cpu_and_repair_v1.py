from pathlib import Path
import datetime,hashlib,json
root=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_risk_completed_20261005');rows={};sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
targets={'a':('b','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'),'b':('c','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa')}
for n,(target,uuid) in targets.items():
 d=root/n;r=json.loads((d/'cpu_preservation_receipt.json').read_text());m=json.loads((d/'preservation_manifest.json').read_text());assert r['training_node']==n and r['assembly_node']=='a' and r['target_cpu_node']==target and target!=n and target!='a'
 assert r['target_gpu_uuid']==uuid and not r['cuda_initialized'] and r['full_state_tensor_count']==461
 assert r['source_sha256']==sha('work/verify_soft_preservation_cpu_v1.py')
 for g,count in [('required_seven_files',7),('extra_files',6)]:
  assert r[g]==m[g] and len(r[g])==count
  for filename,x in r[g].items():assert sha(d/filename)==x['sha256'] and (d/filename).stat().st_size==x['bytes']
 rows[n]=r
out=dict(status='FINITE_AB_SEVEN_PLUS_SIX_CPU_ORIGINAL_RECEIPTS_AUDITED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,scope='A-to-B and B-to-C; both targets outside training and common assembly newA. Two arms only, Cfailed not100. CPU Torch finite all461tensors/ZIP/donor-head correspondence proven by original independent verifier process.')
Path('outputs/有限任务风险两臂七文件异训练与组装CPU保存核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
c=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_risk_preflight_20261005T1434Z/c');r=json.loads((c/'single_token_mechanism_receipt.json').read_text());plan=json.loads((c/'single_token_precheck_plan.json').read_text());assert not r['test_requested'] and not r['dev_requested'] and r['train_rows']==1281 and r['pilot_steps']==200 and r['failed_witness_gradient_finite'] and r['maximum_all_train_prediction_error']<2e-5
assert len(r['all_train_shared10_gradient_batches'])==41 and all(x['all_gradients_finite'] for x in r['all_train_shared10_gradient_batches'])
assert r['source_sha256']==sha('work/check_finite_single_token_fix_v1.py')==plan['source_sha256']['check_finite_single_token_fix_v1.py'] and r['reader_source_sha256']==sha('work/finite_single_token_reader_v1.py')==plan['source_sha256']['finite_single_token_reader_v1.py']
out=dict(status='FINITE_SINGLE_TOKEN_MECHANISM_ORIGINAL_RECEIPT_INDEPENDENTLY_AUDITED_NOT_FORMAL100',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipt=r,plan=plan,remaining_dependencies=['full original-input classifier replay','initialization/complete orders/sharedwarmup equivalence','new formal amendment budget and source freeze'],scope='Original C v1 remains failed. Shared10/fresh Adam200pilot is not original optimizer recovery. No performance or semantic claim.')
Path('outputs/单token解析分支TRAIN机制独立核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print('FINITE_CPU_AND_SINGLE_TOKEN_MECHANISM_AUDITED',r['maximum_all_train_prediction_error'],r['elapsed_seconds'])
