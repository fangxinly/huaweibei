"""Separate CPU verifier; training-node rotation and assembly location explicit."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,zipfile
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--target-node',required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();m=json.loads((a.directory/'preservation_manifest.json').read_text());assert m['training_node']!=a.target_node
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==a.expected_uuid
for group in ['required_seven_files','extra_files']:
 for n,x in m[group].items():assert (a.directory/n).stat().st_size==x['bytes'] and sha(a.directory/n)==x['sha256']
with zipfile.ZipFile(a.directory/'full_checkpoint.pt') as z:assert z.testzip() is None
state=torch.load(a.directory/'full_checkpoint.pt',map_location='cpu');addon=torch.load(a.directory/'best_addon.pt',map_location='cpu')
assert all(v.device.type=='cpu' and torch.isfinite(v).all() for v in state.values())
for key,v in addon['donor'].items():assert torch.equal(state['dberta.own_flow.donor_feedback.'+key],v)
for key,v in addon['feedback'].items():assert torch.equal(state['dberta.own_flow.vector_feedback.'+key],v)
assert not torch.cuda.is_initialized()
history=json.loads((a.directory/'history.json').read_text());sel=json.loads((a.directory/'selection.json').read_text());assert len(history)==100 and sel['best_epoch']==np.argmin([x['dev_author_batch_mean_mse'] for x in history])+1
with np.load(a.directory/'predictions.npz') as z:assert z['valid_pred'].shape==z['valid_y'].shape==(229,)
r={'status':'ROTATED_TRAINING_NODE_SEVEN_FILES_AND_EXTRAS_FULL_CPU_SHA_ZIP_TENSORS_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_node':m['training_node'],'assembly_node':m['assembly_node'],'target_cpu_node':a.target_node,'target_gpu_uuid':uuid,'cuda_initialized':False,'required_seven_files':m['required_seven_files'],'extra_files':m['extra_files'],'full_state_tensor_count':len(state),'source_sha256':sha(__file__),'directory':str(a.directory),'scope':'All full checkpoints were assembled/strictGPUreplayed on newA; rotated target is independent of original training node, not necessarily of assembly node. This separate CPU process validates all hashes and trained tensor correspondence; it is not another training run.'}
(a.directory/'cpu_preservation_receipt.json').write_text(json.dumps(r,indent=2));print('CPU_PRESERVATION_VERIFIED',m['training_node'],'TO',a.target_node,flush=True)
