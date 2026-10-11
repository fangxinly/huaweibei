from pathlib import Path
import hashlib,json,datetime,zipfile
w=Path(__file__).resolve().parent;d=Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_precheck_20261006T0210Z');sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((w.parent/'outputs/视频隔离教师v2预检CPU传输包冻结.json').read_text(encoding='utf-8'))['files'];reports={}
local=json.loads((w.parent/'outputs/视频隔离教师v2三GPU机制回执独立核验.json').read_text(encoding='utf-8'))
for node,suffix,uuid in [('b','','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'),('c','_c','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa')]:
 rpath=d/f'cpu_preservation_receipt_v2{suffix}.json';r=json.loads(rpath.read_text());a=d/f'independent_cpu_three_fold_audit{suffix}.json';remote=json.loads(a.read_text());log=(d/f'cpu_verify{suffix}.log').read_bytes()
 assert r['status']=='TEACHER_V2_ORIGINAL_THREE_GPU_PRECHECK_ARRAYS_AND_RETAINED_V1_FAILURES_CPU_SHA_AUDITED' and r['files']==manifest
 assert r['target_gpu_uuid']==uuid and not r['cuda_initialized'] and not r['initial_fullweights_downloaded'] and not r['formal_training_started']
 assert r['source_sha256']==sha((w/'verify_group_teacher_precheck_cpu_v2.py').read_bytes()) and r['independent_array_audit_sha256']==sha(a.read_bytes())
 assert b'TEACHER_PRECHECK_CPU_PRESERVATION_COMPLETE 41' in log and 'Traceback' not in log.decode()
 assert remote['plan_sha256']==local['plan_sha256'] and remote['common_non_normalization_initial_tensor_sha256']==local['common_non_normalization_initial_tensor_sha256']
 for x,y in zip(remote['folds'],local['folds']):assert x==y
 reports[node]={'original_receipt_sha256':sha(rpath.read_bytes()),'files':len(manifest),'independent_from_original_gpu_host':r['independent_from_original_gpu_host']}
assert reports['b']['independent_from_original_gpu_host']['a'] and reports['b']['independent_from_original_gpu_host']['c'] and reports['c']['independent_from_original_gpu_host']['b']
out=dict(status='THREE_GPU_ORIGINAL_V2_MECHANISMS_CPU_B_C_ORIGINAL_RECEIPTS_DOWNLOADED_AND_INDEPENDENTLY_AUDITED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),nodes=reports,full_initial_weights_downloaded=False,formal100_started=False,scope='Small original GPU arrays/source/failed gradient evidence only. Each origin has at least one CPU target on a different GPU host; not full initial model reload on that CPU.')
target=w.parent/'outputs/视频隔离教师v2异节点CPU原回执核验.json';assert not target.exists();target.write_text(json.dumps(out,indent=2),encoding='utf-8');print(out['status'])
