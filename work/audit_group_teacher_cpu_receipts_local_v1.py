from pathlib import Path
import argparse,json,datetime
from audit_group_teacher_completed_files_v1 import sha
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
nodes=['a','b','c'];targets=['b','c','a'];uuids=['GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa','GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7'];result={}
for fold,(node,target,uuid) in enumerate(zip(nodes,targets,uuids)):
 d=a.directory/node;c=json.loads((d/'cpu_preservation_receipt.json').read_text(encoding='utf-8'));m=json.loads((d/'preservation_manifest.json').read_text(encoding='utf-8'));completion=json.loads((d/'run_v1/completion.json').read_text(encoding='utf-8'));ex=json.loads((d/'cpu_verification_exit.json').read_text(encoding='utf-8'))
 assert ex['exit_code']==0
 assert c['status']=='INDEPENDENT_TRAINING_AND_ASSEMBLY_HOST_COMPLETED_TEACHER_SEVEN_FILES_EXTRAS_CPU_SHA_ZIP_TENSORS_ARRAYS_VERIFIED'
 assert c['training_node']==c['assembly_node']==node and c['target_cpu_node']==target and c['target_gpu_uuid']==uuid and c['fold']==fold and not c['cuda_initialized']
 assert c['source_sha256']==sha(d/'verify_group_teacher_completed_cpu_v1.py')==sha(Path(__file__).parent/'verify_group_teacher_completed_cpu_v1.py')
 assert c['manifest_sha256']==sha(d/'preservation_manifest.json') and c['full_state_tensor_count']==301 and c['selected_model_tensor_sha256']==completion['selected_model_tensor_sha256']
 assert c['required_seven_files']==m['required_seven_files'] and c['extra_files']==m['extra_files'] and len(m['required_seven_files'])==7 and len(m['extra_files'])==20
 for group in ['required_seven_files','extra_files']:
  for n,meta in m[group].items():assert (d/n).stat().st_size==meta['bytes'] and sha(d/n)==meta['sha256']
 assert c['independent_array_audit']['best_epoch']==completion['best_epoch'] and c['independent_array_audit']['full_file_sha_crc_checked']
 result[node]={'training_node':node,'assembly_node':node,'CPU_target_node':target,'target_gpu_uuid':uuid,'original_receipt_sha256':sha(d/'cpu_preservation_receipt.json'),'selected_full_tensor_sha256':c['selected_model_tensor_sha256'],'original_exit_code':ex['exit_code'],'all_seven_and_twenty_extra_files_match':True,'CPU_forward_claimed':False}
report={'status':'THREE_TEACHER_ORIGINAL_INDEPENDENT_CPU_RECEIPTS_DOWNLOADED_AND_D_SEVEN_TWENTY_EXTRA_HASHES_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'nodes':result,'whole_pipeline_crossfit':False}
assert not a.out.exists();a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(report['status'])
