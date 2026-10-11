"""Independent host CPU full SHA/ZIP/tensors plus NumPy selection/arrays audit."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess
import torch
from audit_group_teacher_completed_files_v1 import audit,sha,REQUIRED

p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--target-node',choices=['a','b','c'],required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
m=json.loads((a.directory/'preservation_manifest.json').read_text(encoding='utf-8'))
assert m['training_node']!=a.target_node and m['assembly_node']==m['training_node']
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==a.expected_uuid
assert set(m['required_seven_files'])==set(REQUIRED) and len(m['required_seven_files'])==7
for group in ['required_seven_files','extra_files']:
 for name,meta in m[group].items():
  p=a.directory/name;assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256']
report=audit(a.directory,m['fold'])
torch.set_num_threads(2)
state=torch.load(a.directory/'run_v1/selected_full_checkpoint.pt',map_location='cpu',weights_only=True)
assert len(state)==301 and all(isinstance(v,torch.Tensor) and v.device.type=='cpu' and torch.isfinite(v).all() for v in state.values())
assert not any('pooler.' in n or any(s in n for s in ['.reflow_a','.reflow_v','.rf_a','.rf_v']) for n in state)
h=hashlib.sha256()
for n,v in sorted(state.items()):h.update(n.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
c=json.loads((a.directory/'run_v1/completion.json').read_text(encoding='utf-8'));assert h.hexdigest()==c['selected_model_tensor_sha256']
assert not torch.cuda.is_initialized()
out={'status':'INDEPENDENT_TRAINING_AND_ASSEMBLY_HOST_COMPLETED_TEACHER_SEVEN_FILES_EXTRAS_CPU_SHA_ZIP_TENSORS_ARRAYS_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_node':m['training_node'],'assembly_node':m['assembly_node'],'target_cpu_node':a.target_node,'target_gpu_uuid':uuid,'fold':m['fold'],'cuda_initialized':False,'full_state_tensor_count':len(state),'selected_model_tensor_sha256':h.hexdigest(),'source_sha256':sha(__file__),'manifest_sha256':sha(a.directory/'preservation_manifest.json'),'required_seven_files':m['required_seven_files'],'extra_files':m['extra_files'],'independent_array_audit':report,'scope':'CPU whole-file SHA/ZIP/finite full tensors/selection/original arrays. Original GPU strict reload is separate evidence; this is not another training or CPU full-model forward run.'}
dest=a.directory/'cpu_preservation_receipt.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2),encoding='utf-8');print('GROUP_TEACHER_COMPLETED_CPU_PRESERVATION_VERIFIED',m['training_node'],'TO',a.target_node)
