"""Create transport manifest only after actual natural exit0 and 100 completion."""
from pathlib import Path
import argparse,datetime,json,subprocess
from audit_group_teacher_completed_files_v1 import audit,sha,REQUIRED

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--node',choices=['a','b','c'],required=True);parser.add_argument('--uuid',required=True);args=parser.parse_args()
fold=['a','b','c'].index(args.node)
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==args.uuid
r=audit(args.root,fold)
extras=['train_group_teacher_v1.py','group_teacher_runtime_v2.py','check_group_teacher_v2.py','group_teacher_plan_v3.json','group_teacher_formal_plan_v4.json','three_fold_gpu_mechanism_audit.json','formal_space_authorization.json','formal_launch.json','formal_training.log','formal_wrapper.log','formal_wrapper_v1.py','precheck_v2/receipt.json',f'fit_{fold}.npy',f'inner_{fold}.npy',f'outer_{fold}.npy',f'orders_{fold}.npy','train_row_video_mapping.json','audit_group_teacher_completed_files_v1.py','manifest_group_teacher_completed_v1.py','verify_group_teacher_completed_cpu_v1.py']
manifest={'status':'COMPLETED100_TRANSPORT_MANIFEST_NOT_YET_LOCAL_OR_CPU_SAVE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_node':args.node,'training_gpu_uuid':uuid,'assembly_node':args.node,'fold':fold,'independent_array_audit':r,'required_seven_files':{},'extra_files':{}}
for group,names in [('required_seven_files',REQUIRED),('extra_files',extras)]:
 for n in names:
  p=args.root/n;assert p.is_file();manifest[group][n]={'bytes':p.stat().st_size,'sha256':sha(p)}
assert len(manifest['required_seven_files'])==7
out=args.root/'preservation_manifest.json';assert not out.exists();out.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('GROUP_TEACHER_COMPLETED100_TRANSPORT_MANIFEST_FROZEN',args.node)
