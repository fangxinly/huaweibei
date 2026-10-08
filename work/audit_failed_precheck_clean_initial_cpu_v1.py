"""CPU full-state/public-encoder audit only, not a model forward or GPU pass."""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--public-weight',required=True);p.add_argument('--out',required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(a.checkpoint)=='38d589de501ddaf7caa9f5328db421f3406b017b1753bfef19ebb62d71adca57'
payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True);s=payload['model'];m=payload['metadata']
assert m['format']=='MINIMAL_FIXED_FULL_MODEL_STATE_V1' and m['runtime_plan_sha256']=='e4d0c71a73186ae9073dc0d88ccdd4fd753a4d81a967eb458a52a0669437fa63'
assert m['fold']==0 and m['seed']==91819 and m['scope']=='CLEAN_INITIAL_NO_PRECHECK' and m['optimizer_steps']==0
def state_digest(items):
 h=hashlib.sha256()
 for n,v in sorted(items.items()):
  assert v.device.type=='cpu' and torch.isfinite(v).all()
  x=v.contiguous();h.update(n.encode());h.update(str((tuple(x.shape),str(x.dtype))).encode());h.update(x.numpy().tobytes())
 return h.hexdigest()
assert state_digest(s)==m['state_sha256']
stats={n:v for n,v in s.items() if n.startswith(('dberta.v6_audio_','dberta.v6_visual_'))};assert state_digest(stats)==m['fit_statistics_sha256']
public=torch.load(a.public_weight,map_location='cpu',weights_only=True);matched=[]
for n,v in s.items():
 if not n.startswith('dberta.model.'):continue
 name=n[len('dberta.model.'):];key=name if name in public else 'deberta.'+name
 if key in public:assert torch.equal(v,public[key].to(dtype=v.dtype));matched.append(n)
assert len(matched)>=190
for i in range(6):
 for suffix in ('weight','bias'):assert torch.count_nonzero(s[f'dberta.own_flow.donor_feedback.{i}.3.{suffix}'])==0
receipt={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'OTHER_NODE_CPU_FULL_CLEAN_INITIAL_AND_PUBLIC_ENCODER_AUDIT_PASSED_NOT_MODEL_FORWARD','pid':os.getpid(),'argv':sys.argv,'auditor_source_sha256':sha(__file__),'checkpoint_file_sha256':sha(a.checkpoint),'checkpoint_bytes':Path(a.checkpoint).stat().st_size,'metadata':m,'complete_state_tensors':len(s),'complete_state_elements':sum(v.numel() for v in s.values()),'public_weight_file_sha256':sha(a.public_weight),'public_encoder_exact_matched_tensors':len(matched),'six_donor_last_layers_exact_zero':True,'GPU_used':False,'CPU_model_forward':False,'failed_GPU_precheck_now_passed':False,'post_two_steps_full_verified':False,'full_INNER_replay_verified':False,'formal100_complete':False}
Path(a.out).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
