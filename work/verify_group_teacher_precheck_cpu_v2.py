from pathlib import Path
import argparse,json,hashlib,datetime,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--uuid',required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert gpu==a.uuid
manifest=json.loads((a.root/'preservation_manifest.json').read_text())
for n,m in manifest.items():
 f=a.root/n;assert f.stat().st_size==m['bytes'] and sha(f)==m['sha256']
argv=[sys.executable,str(a.root/'audit_group_teacher_mechanism_receipts_v2.py'),'--plan-dir',str(a.root/'plan'),'--receipts-dir',str(a.root/'original_receipts'),'--output',str(a.root/'independent_three_fold_audit.json')]
code=subprocess.call(argv);assert code==0
results=[]
for fold,node in enumerate(['a','b','c']):
 pre=json.loads((a.root/'original_receipts'/node/'precheck_v2/receipt.json').read_text())
 assert pre['fold']==fold and pre['all_retained_parameter_gradients_finite'] and pre['retained_parameter_tensors']==291
 assert pre['all_other_initial_tensors_equal_v1'] and pre['original_v1_initial_prediction_error']==0
 assert json.loads((a.root/'original_receipts'/node/'precheck_v2_exit.json').read_text())['exit_code']==0
 old=json.loads((a.root/'original_receipts'/node/'all_gradient_receipt.json').read_text())
 assert old['missing_gradient_names']==['dberta.pooler.dense.weight','dberta.pooler.dense.bias'] and old['finite_gradient_tensors']==291
 assert json.loads((a.root/'original_receipts'/node/'all_gradient_exit.json').read_text())['exit_code']==1
 results.append(dict(fold=fold,retained_parameter_tensors=291,trainable_parameters=pre['trainable_parameters'],all_gradients_finite=True,unused_pooler_prior_failure_retained=True))
origins={'a':'GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7','b':'GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
r=dict(status='TEACHER_V2_ORIGINAL_THREE_GPU_PRECHECK_ARRAYS_AND_RETAINED_V1_FAILURES_CPU_SHA_AUDITED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_gpu_uuid=gpu,files=manifest,original_receipt_scope='Original A/B/C GPU mechanism arrays audited by NumPy CPU; independent_from_original_gpu_host states the boundary per source. No full initial weight copied by this receipt.',independent_from_original_gpu_host={n:v!=gpu for n,v in origins.items()},cuda_initialized=False,source_sha256=sha(Path(__file__)),independent_array_audit_sha256=sha(a.root/'independent_three_fold_audit.json'),folds=results,initial_fullweights_downloaded=False,formal_training_started=False)
target=a.root/'cpu_preservation_receipt.json';assert not target.exists();target.write_text(json.dumps(r,indent=2),encoding='utf-8');print('TEACHER_PRECHECK_CPU_PRESERVATION_COMPLETE',len(manifest))
