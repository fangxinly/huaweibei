"""One independent CPU read of this new original; no inference, fitting or scoring."""
import argparse,datetime as dt,hashlib,json,os,pathlib
P=pathlib.Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--job',type=P,required=True);a=a.parse_args();r=a.job
 maximum=int(P('/sys/fs/cgroup/memory.max').read_text());current=int(P('/sys/fs/cgroup/memory.current').read_text());assert maximum-current>=6*1024**3
 original=json.loads((r/'out/checkpoint_receipt.json').read_bytes());cp=r/'out/complete_state.pt';assert sha(cp)==original['SHA']
 assert json.loads((r/'train_natural_exit.json').read_bytes())['natural_exit']==0
 os.environ['CUDA_VISIBLE_DEVICES']=''
 import torch
 from fixed_flow_components_candidate import tensor_sha
 torch.set_num_threads(2);state=torch.load(cp,map_location='cpu');m=state['metadata'];result=json.loads((r/'out/result.json').read_bytes())
 assert m['steps']==4000 and m['epochs_completed']==100 and len(state['history'])==100
 assert state['scheduler']['last_epoch']==4000 and state['orders'].shape==(100,1281)
 assert tensor_sha(state['selected_model'])==result['selected_state_SHA']
 assert m['best_epoch']==result['best_epoch'] and m['TEST_used_for_selection'] is False
 name_map=m['optimizer_index_to_name'];missing=[]
 for key,name in name_map.items():
  row=state['optimizer']['state'].get(int(key))
  if row is None:missing.append(name);continue
  assert float(row['step'])==4000 and torch.isfinite(row['exp_avg']).all() and torch.isfinite(row['exp_avg_sq']).all()
  assert row['exp_avg'].shape==state['model'][name].shape==row['exp_avg_sq'].shape
 allowed={'dberta.pooler.dense.weight','dberta.pooler.dense.bias'} if m['method']=='careflow' else set()
 assert set(missing)==allowed
 assert set(state['rng'])=={'python','numpy','torch','cuda'} and state['rng']['cuda']
 assert all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point())
 receipt=dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='NEW_ORIGINAL_FULL4000_CPU_AUDIT_COMPLETE',checkpoint_SHA=sha(cp),method=m['method'],seed=m['seed'],steps=4000,epochs=100,selected_state_SHA=result['selected_state_SHA'],optimizer_full_moments_steps_names_passed=True,allowed_original_unused_parameter_names=missing,RNG_inventory_passed=True,no_training_inference_or_scoring=True,CPU_CUDA_resume_equivalence_not_claimed=True)
 (r/'CPU_audit_original.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
