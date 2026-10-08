"""Build one evaluation-only diagnostic without editing frozen training/model source."""
from pathlib import Path
source=Path('work/weak_pilot_fixed_selected_diagnostic.py').read_text(encoding='utf8')
source=source.replace("bundle=parent/'source' if a.node=='B' else Path(plan['A_bundle'])\n if a.node=='B':parent=parent/'run'", "bundle=Path(plan['B_bundle'] if a.node=='B' else plan['A_bundle'])")
start=source.index(' def forward(role,dummy):');end=source.index(' scores={};digests={}',start)
source=source[:start]+''' current_mask=[1.]*6;cache=[]
 def cache_args(module,args):cache[:]=[args]
 def feedback_hook(index):
  def apply(module,args,output):return output*current_mask[index]
  return apply
 hooks=[flow.register_forward_pre_hook(cache_args)]+[module.register_forward_hook(feedback_hook(i)) for i,module in enumerate(flow.donor_feedback)]
 mask_names=['off','on']+[f'single_{i}' for i in range(6)]
 mask_values=[[0.]*6,[1.]*6]+[[float(j==i) for j in range(6)] for i in range(6)]
 def forward(role,dummy):
  all_values={k:[] for k in ('b','f','p')};counter={name:[] for name in mask_names};tensors=session.inputs[role].tensors
  with torch.no_grad():
   for start in range(0,len(tensors[0]),128):
    current_mask[:]=[1.]*6
    batch=[v[start:start+128].to(session.author.DEVICE) for v in tensors];batch[3]=torch.full_like(batch[3],dummy)
    p=prediction(session,'anchored_flow',tuple(batch))
    for k,v in [('b',flow.last_base),('f',flow.last_flow),('p',p)]:all_values[k].append(v.detach().cpu().numpy())
    args=cache[0];source_before=args[0].clone();mask_before=args[1].clone()
    if role in ('fit','inner'):
     for name,values in zip(mask_names,mask_values):
      current_mask[:]=values
      qp=flow(*args)[0]
      if name=='on':assert torch.equal(qp,p)
      assert torch.equal(args[0],source_before) and torch.equal(args[1],mask_before)
      counter[name].append(qp.detach().cpu().numpy())
    current_mask[:]=[1.]*6
  original={k:np.concatenate(v).astype(np.float32) for k,v in all_values.items()}
  masks={k:np.concatenate(v).astype(np.float32) for k,v in counter.items()} if role in ('fit','inner') else {}
  return original,masks
''' +source[end:]
source=source.replace('scores={};digests={}', 'scores={};digests={};mask_digests={};mask_scores={}')
source=source.replace('pred=forward(role,0);dummy=forward(role,7)', 'pred,masks=forward(role,0);dummy,dummy_masks=forward(role,7)\n  for name in masks:assert np.array_equal(masks[name],dummy_masks[name])')
source=source.replace("scores[role]=arithmetic(pred,labels,metrics)", "scores[role]=arithmetic(pred,labels,metrics)\n  if masks:\n   mask_path=a.root/(role+'_donor_masks.npz');np.savez(mask_path,**masks,row_ids=np.asarray(fold['row_ids'][role]));mask_digests[role]=sha(mask_path);mask_scores[role]=mask_arithmetic(masks,labels,metrics)")
source=source.replace('assert tensor_sha(session.model.state_dict())==before', 'for hook in hooks:hook.remove()\n assert tensor_sha(session.model.state_dict())==before',1)
source=source.replace('scores=scores,prediction_sha256=digests', 'scores=scores,prediction_sha256=digests,donor_mask_prediction_sha256=mask_digests,donor_mask_diagnostic=mask_scores')
source=source.replace("scope='Human requested fixed20 official VAL/TEST descriptive scores; pooled fold FIT/INNER overlap disclosed; no checkpoint or method selection'", "scope='Fixed weak-loss selected36; FIT/INNER donor-output masks only. p0 same Euler/readout with donor feedback off, role reader/head retained. Official VAL/TEST p only descriptive pooled overlap; no mask/readout/checkpoint selection. Cached inference encoder activations reused, no training or encoder-freezing claim.'")
source=source.replace("FIXED_WEAK_PILOT_BFP_GPU_COMPLETE_CPU_PENDING", "FIXED_WEAK_PILOT_DONOR_MASK_GPU_COMPLETE_CPU_PENDING")
point=source.index('def arithmetic(')
source=source[:point]+'''def mask_arithmetic(pred,labels,metric):
 import numpy as np
 y=np.asarray(labels,dtype=np.float64).reshape(-1);p0=np.asarray(pred['off'],dtype=np.float64);p1=np.asarray(pred['on'],dtype=np.float64)
 d=p1-p0;r=y-p0;U=r*r-(y-p1)**2;error=float(np.max(np.abs(U-(2*r*d-d*d))));assert error<1e-10
 result=dict(utility_identity_max_error=error,definition='Off/on same checkpoint Euler and decoder; only six donor feedback MLP outputs masked; reader and role head retained.',regions={})
 for region,mask in [('all',np.ones(len(y),bool)),('weak',abs(y)<=1),('strong',abs(y)>1)]:
  yy=y[mask];dd=d[mask];rr=r[mask]
  corr=float(np.corrcoef(dd,rr)[0,1]) if np.std(dd)>0 and np.std(rr)>0 else None
  result['regions'][region]=dict(rows=int(mask.sum()),p0=metric(p0[mask],yy),p1=metric(p1[mask],yy),delta_mean=float(dd.mean()),delta_std=float(dd.std()),delta_rms=float(np.sqrt(np.mean(dd*dd))),corr_delta_residual=corr,positive_utility_fraction=float(np.mean(U[mask]>0)),mean_utility=float(U[mask].mean()),binary_oracle_MSE=float(np.mean(np.minimum((p0[mask]-yy)**2,(p1[mask]-yy)**2))),binary_oracle_gain_vs_p1=float(np.mean((p1[mask]-yy)**2)-np.mean(np.minimum((p0[mask]-yy)**2,(p1[mask]-yy)**2))))
 result['single_direction_scores']={name:metric(value,y) for name,value in pred.items() if name.startswith('single_')}
 singles=np.stack([np.asarray(pred[f'single_{i}'],dtype=np.float64)-p0 for i in range(6)])
 closure=d-singles.sum(0);result['nonadditive_output_closure_rms']=float(np.sqrt(np.mean(closure*closure)))
 return result

''' +source[point:]
needle="assert s==raw['scores'][role]"
source=source.replace(needle,needle+"\n  if role in ('fit','inner'):\n   mp=original/(role+'_donor_masks.npz');assert sha(mp)==raw['donor_mask_prediction_sha256'][role]\n   with np.load(mp,allow_pickle=False) as z:masks={k:z[k] for k in z.files if k!='row_ids'}\n   assert np.array_equal(masks['on'],pred['p'])\n   assert mask_arithmetic(masks,np.load(lp,allow_pickle=False),metrics)==raw['donor_mask_diagnostic'][role]")
source=source.replace("FIXED_WEAK_PILOT_BFP_CPU_ORIGINAL_ARITHMETIC_COMPLETE", "FIXED_WEAK_PILOT_DONOR_MASK_CPU_ORIGINAL_ARITHMETIC_COMPLETE")
target=Path('work/fixed_weak_pilot_donor_mask_diagnostic.py');compile(source,str(target),'exec');target.write_text(source,encoding='utf8')
print(target)
