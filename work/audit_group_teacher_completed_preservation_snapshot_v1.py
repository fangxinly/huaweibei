"""Join original atomic snapshot references with actual D originals and CPU receipts."""
from pathlib import Path
import argparse,datetime,json,zipfile,hashlib
from audit_group_teacher_completed_files_v1 import audit,sha
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--completed',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
nodes=['a','b','c'];targets=['b','c','a'];uuid={'a':'GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7','b':'GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
zips={};result={}
for node in nodes:
 d=a.directory/node;r=json.loads((d/'receipt.json').read_text(encoding='utf-8'));assert (d/'snapshot.zip').stat().st_size==r['bytes'] and sha(d/'snapshot.zip')==r['sha256'] and (d/'exit_code.txt').read_text().strip()=='0'
 z=zipfile.ZipFile(d/'snapshot.zip');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()));m=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
 for n,meta in m.items():assert len(z.read(n))==meta['bytes'] and hashlib.sha256(z.read(n)).hexdigest()==meta['sha256']
 zips[node]=(z,m,json.loads(z.read('large_file_manifest.json')),r)
for fold,node in enumerate(nodes):
 d=a.completed/node;local=audit(d,fold);manifest=json.loads((d/'preservation_manifest.json').read_text(encoding='utf-8'));cpu=json.loads((d/'cpu_preservation_receipt.json').read_text(encoding='utf-8'));target=targets[fold]
 assert cpu['training_node']==cpu['assembly_node']==node and cpu['target_cpu_node']==target and cpu['target_gpu_uuid']==uuid[target] and not cpu['cuda_initialized']
 assert cpu['status']=='INDEPENDENT_TRAINING_AND_ASSEMBLY_HOST_COMPLETED_TEACHER_SEVEN_FILES_EXTRAS_CPU_SHA_ZIP_TENSORS_ARRAYS_VERIFIED' and cpu['full_state_tensor_count']==301
 assert cpu['source_sha256']==sha(d/'verify_group_teacher_completed_cpu_v1.py') and cpu['manifest_sha256']==sha(d/'preservation_manifest.json')
 assert cpu['required_seven_files']==manifest['required_seven_files'] and cpu['extra_files']==manifest['extra_files']
 original_z,original_m,original_l,original_r=zips[node];target_z,target_m,target_l,target_r=zips[target]
 cp='group_teacher/cpu_preservation/completed/'+node+'/'
 for name in ['preservation_manifest.json','cpu_preservation_receipt.json']:
  assert target_z.read(cp+name)==(d/name).read_bytes()
 assert original_z.read('group_teacher/preservation_manifest.json')==(d/'preservation_manifest.json').read_bytes()
 completion=json.loads((d/'run_v1/completion.json').read_text(encoding='utf-8'));assert cpu['selected_model_tensor_sha256']==completion['selected_model_tensor_sha256']
 for group in ['required_seven_files','extra_files']:
  for name,meta in manifest[group].items():
   assert sha(d/name)==meta['sha256'] and (d/name).stat().st_size==meta['bytes']
   for small,large,prefix in [(original_m,original_l,'group_teacher/'),(target_m,target_l,cp)]:
    seen=large[prefix+name] if prefix+name in large else small[prefix+name]
    assert seen['sha256']==meta['sha256'] and seen['bytes']==meta['bytes']
 result[node]={'fold':fold,'best_epoch':completion['best_epoch'],'training_and_assembly_node':node,'independent_CPU_node':target,'D_full_checkpoint_sha256':completion['full_checkpoint_sha256'],'D_full_bytes':completion['full_checkpoint_bytes'],'original_capture_utc':original_r['utc'],'CPU_capture_utc':target_r['utc'],'original_CPU_receipt_sha256':sha(d/'cpu_preservation_receipt.json'),'original_seven_and_twenty_extra_files_checked':True,'CPU_301_tensor_SHA_verified_by_original_receipt':True,'D_full_file_SHA_CRC_verified_here':True,'CPU_full_model_forward_claimed':False}
for z,_,_,_ in zips.values():z.close()
report={'status':'THREE_COMPLETED100_TEACHERS_D_FULL_WEIGHT_ORIGINAL_SEVEN_TWENTY_EXTRA_INDEPENDENT_CPU_RECEIPTS_AND_ATOMIC_SNAPSHOTS_JOINED_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'nodes':result,'whole_pipeline_crossfit':False,'scope':'Actual D weights are hashed and CRC checked here, distinct from large SHA snapshot references. Original GPU strict model forward reload and independent host CPU file/tensor/array verification are separate recorded checks.'}
assert not a.out.exists();a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(report['status'])
