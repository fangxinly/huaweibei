"""Verify actual D whole files and capsule, then prepare exact CPU transfer args."""
from pathlib import Path,PurePosixPath
import sys,json,hashlib,zipfile
sys.path.insert(0,str(Path(__file__).parent/'paired_official_fulltrain_v2_20261007T011543Z'))
from paired_fulltrain_evidence_candidate_v1 import sha,read,write,zip_audit,capture_association,stage_association,plan_gate
from paired_fulltrain_saved_joint_candidate_v1 import extract
method=sys.argv[1];tag=sys.argv[2]
d=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_fulltrain100_complete_actual_20261007T030311Z')/method
extract(d/'a');c,m=capture_association(d/'a');planSHA=c['plan_sha256'];plan=plan_gate(d/'a/source',planSHA)
r,e=stage_association(d/'a/run',planSHA,method)
assert r['status']=='ACTUAL_PAIRED_METHOD_FULLTRAIN100_COMPLETE_PRESERVATION_PENDING_NO_FINAL_TEST' and e['exit_code']==0
whole={}
for n in ['selected_best_full','complete_resume_full']:
 p=d/'a/run/out'/Path(r[n]['path']).name;assert p.stat().st_size==r[n]['bytes'];whole[n]=zip_audit(p,r[n]['sha256'])
cpu=PurePosixPath('/data/coding')/('paired_fulltrain_cpu_train100_'+method+'_'+tag)
original=cpu/'original/run'
args=['--root',str(cpu),'--original-root',str(original),'--bundle','/data/coding/paired_official_fulltrain_v2_20261007T011543Z','--plan-sha',planSHA,'--original-receipt-sha',sha(d/'a/run/out/actual_stage_receipt.json'),'--original-exit-sha',sha(d/'a/run/natural_exit.json'),'--method',method,'--stage','train']
write(d/'actual_CPU_child_arguments.json',args)
write(d/'actual_D_GPU_whole_SHA_CRC_source_capture_receipt.json',{'status':'ACTUAL_GPU_COMPLETE_D_WHOLE_SHA_CRC_AND_CAPSULE_PASSED_CPU_PENDING','method':method,'original_natural_exit':e,'original_stage_receipt_SHA':sha(d/'a/run/out/actual_stage_receipt.json'),'GPU_capture':c,'whole_D_files':whole,'new_CPU_root':str(cpu),'CPU_argument_sha256':sha(d/'actual_CPU_child_arguments.json'),'CPU_model_forward':False,'final_TEST':False})
print(json.dumps({'method':method,'CPU_root':str(cpu),'original_copy_root':str(original),'snapshot_SHA':c['snapshot_sha256'],'GPU_receipt_SHA':sha(d/'a/run/out/actual_stage_receipt.json'),'GPU_exit_SHA':sha(d/'a/run/natural_exit.json'),'CPU_args_SHA':sha(d/'actual_CPU_child_arguments.json'),'whole':whole},ensure_ascii=False),flush=True)
