from pathlib import Path
import argparse,json,hashlib,datetime,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--uuid',required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert gpu==a.uuid
manifest=json.loads((a.root/'preservation_manifest.json').read_text())
for n,m in manifest.items():
 f=a.root/n;assert f.stat().st_size==m['bytes'] and sha(f)==m['sha256']
argv=[sys.executable,str(a.root/'audit_group_teacher_mechanism_receipts_v1.py'),'--plan-dir',str(a.root/'plan'),'--receipts-dir',str(a.root/'original_receipts'),'--output',str(a.root/'independent_three_fold_audit.json')]
code=subprocess.call(argv);assert code==0
grads=[]
for fold,node in enumerate(['a','b','c']):
 r=json.loads((a.root/'original_receipts'/node/'all_gradient_receipt.json').read_text())
 assert r['fold']==fold and not r['missing_gradient_names'] and not r['parameters_updated']
 assert r['finite_gradient_tensors']==r['parameter_tensors']
 assert r['source_sha256']==sha(a.root/'check_group_teacher_all_gradients_v1.py')
 assert all(v['l1']>=0 for v in r['gradients'].values())
 assert not r['dev_requested'] and not r['test_requested'] and not r['outer_labels_read']
 e=json.loads((a.root/'original_receipts'/node/'all_gradient_exit.json').read_text());assert e['exit_code']==0
 pre=json.loads((a.root/'original_receipts'/node/'precheck_v1/receipt.json').read_text());assert r['initial_tensor_sha256']==pre['initial_full_tensor_sha256']
 grads.append(dict(fold=fold,parameter_tensors=r['parameter_tensors'],finite_gradient_tensors=r['finite_gradient_tensors'],parameters=r['parameters'],zero_gradient_names=r['zero_gradient_names'],receipt_sha256=sha(a.root/'original_receipts'/node/'all_gradient_receipt.json')))
r=dict(status='TEACHER_ORIGINAL_THREE_GPU_PRECHECK_ARRAYS_AND_ALL_PARAMETER_GRADIENT_RECEIPTS_CPU_SHA_AUDITED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_gpu_uuid=gpu,files=manifest,original_receipt_scope='A/B/C original GPU mechanisms inspected by NumPy CPU on B; B-origin receipt verification is same-host and does not establish B model preservation on a different host.',cuda_initialized=False,source_sha256=sha(Path(__file__)),independent_array_audit_sha256=sha(a.root/'independent_three_fold_audit.json'),all_gradient_receipts=grads,initial_fullweights_downloaded=False,formal_training_started=False)
target=a.root/'cpu_preservation_receipt.json';assert not target.exists();target.write_text(json.dumps(r,indent=2),encoding='utf-8');print('TEACHER_PRECHECK_CPU_PRESERVATION_COMPLETE',len(manifest))
