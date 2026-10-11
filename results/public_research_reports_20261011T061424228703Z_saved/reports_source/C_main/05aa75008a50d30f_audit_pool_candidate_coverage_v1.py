"""Post-freeze unlabeled coverage and exact duplicate-action accounting."""
from pathlib import Path
import argparse,hashlib,json,numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['precheck','execute'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
file=a.root/a.phase/'predictions_frozen.npz';z=np.load(file,allow_pickle=False)
pred=z['candidate_prediction'].astype(np.float64);pf=z['pf'].astype(np.float64);message=z['candidate_messages'];valid=z['candidate_valid'];rows=len(pf)
max_duplicate_error=0.;pool_stats=[]
for i in range(1,13):
    for j in range(i):
        same=(message[:,i]==message[:,j]).all((1,2))
        if same.any():max_duplicate_error=max(max_duplicate_error,float(np.max(np.abs(pred[same,i]-pred[same,j]))))
assert max_duplicate_error==0
pools=[[0,1,2,3,4],[0,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,9,10,11,12]]
for ids in pools:
    unique=np.ones((rows,len(ids)),dtype=bool)
    for i,ii in enumerate(ids):
        for j in ids[:i]:unique[:,i]&=~(message[:,ii]==message[:,j]).all((1,2))
    item=dict(nominal_candidates=len(ids),mean_unique_actions=float(unique.sum(1).mean()),mean_cap_valid_unique=float((unique&valid[:,ids]).sum(1).mean()))
    for name,target in [('old',z['target_old']),('cal',z['target_cal'])]:
        d=pred[:,ids]-pf[:,None];q=2*(pf-target)[:,None]*d+d*d;eligible=(q<0)&valid[:,ids]&unique
        if name=='cal':eligible[z['lambda_value']==0]=False
        item[name+'_rows_with_negative_proxy_candidate']=int(eligible.any(1).sum());item[name+'_mean_negative_proxy_unique']=float(eligible.sum(1).mean())
    pool_stats.append(item)
for i,lo,hi in [(0,1,2),(1,2,3),(2,3,4),(3,5,6),(4,6,7),(5,7,8),(6,9,10),(7,10,11),(8,11,12)]:
    small=message[:,lo]-message[:,0];big=message[:,hi]-message[:,0]
    assert np.max(np.abs(small-.5*big))<1e-6 # Actual messages, not output linearity.
proof=dict(status='UNLABELED_REAL_MESSAGE_DUPLICATES_HALF_STEPS_AND_COVERAGE_VERIFIED',phase=a.phase,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),array_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),maximum_exact_duplicate_prediction_difference=max_duplicate_error,pools=pool_stats,
    candidate_budget_scope='Both main pools nine selection opportunities and shared 13 candidate terminal calls. Teacher mu and three-step path generation add cost; independent single-arm timing not measured.',labels_requested=False,no_fit=True,no_model_forward=True)
a.out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
