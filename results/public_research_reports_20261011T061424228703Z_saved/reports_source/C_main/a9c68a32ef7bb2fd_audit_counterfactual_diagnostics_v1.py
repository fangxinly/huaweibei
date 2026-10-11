"""Independent CPU audit of the predefined 29 frozen DEV interventions."""
from pathlib import Path
import argparse,datetime,hashlib,io,json,zipfile
import numpy as np
from analyze_utility_completed_v1 import digest,rank,corr,smooth,NAMES
from audit_utility_diagnostics_v1 import measured,UUIDS,MODES
CONDITIONS=['default','off','forced_fixed','forced_predicted']+[f'disable_{i}' for i in range(6)]+[f'joint_disable_{i}' for i in range(3)]+[f'shift_donor_{i}' for i in range(6)]+['shift_donor_all']+[f'fixed_disable_{i}' for i in range(6)]+[f'fixed_joint_disable_{i}' for i in range(3)]
def calibration(u,g):
 t=np.tanh(g/.05);pos=g>1e-10;neg=g< -1e-10;nz=pos|neg
 pr=float((u[pos]>0).mean()) if pos.any() else None;nr=float((u[neg]<0).mean()) if neg.any() else None
 return {'mean_gain':float(g.mean()),'absolute_mean_gain':float(np.abs(g).mean()),'positive_fraction':float(pos.mean()),'negative_fraction':float(neg.mean()),'zero_count':int((~nz).sum()),'positive_recall':pr,'negative_recall':nr,'balanced_sign_recall':(pr+nr)/2 if pr is not None and nr is not None else None,'nonzero_sign_accuracy':float((np.sign(u[nz])==np.sign(g[nz])).mean()) if nz.any() else None,'u_positive_fraction':float((u>0).mean()),'spearman':corr(rank(u),rank(g)),'MSE_to_target':float(np.square(u-t).mean()),'zero_MSE_to_target':float(np.square(t).mean()),'smooth_l1':smooth(u,t),'zero_smooth_l1':smooth(np.zeros_like(u),t)}
def gain(base,other,y):
 d=other-base;e=base-y;g=np.square(other-y)-np.square(e);err=float(np.max(np.abs(g-d*(2*e+d))));assert err<1e-10
 return g,err
def main():
 a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True);a.add_argument('--completed-audit',type=Path,required=True);a.add_argument('--capture-source',type=Path,required=True);a.add_argument('--nodes',nargs='+',choices=list(MODES),required=True);a.add_argument('--out',type=Path,required=True);c=a.parse_args()
 complete=json.loads(c.completed_audit.read_text(encoding='utf-8'));assert complete['stage']=='complete' and complete['status']=='COUNTERFACTUAL_SNAPSHOT_AUDITED';selected={r['node']:r for r in complete['rows']};rows=[]
 for node in c.nodes:
  chosen=selected[node];assert chosen['epochs']==100 and chosen['full_checkpoint_verified'];folder=c.directory/node;proof=json.loads((folder/'proof.json').read_text(encoding='utf-8'))
  assert digest(folder/'snapshot.zip')==proof['archive_sha256'] and proof['capture_source_sha256']==digest(c.capture_source)
  with zipfile.ZipFile(folder/'snapshot.zip') as z:
   members=json.loads(z.read('member_manifest.json'));assert members==proof['members'] and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(members)|{'member_manifest.json'}
   for n,v in members.items():
    raw=z.read(n);assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
   res=json.loads(z.read('actual_resources.json'));assert all(res[k]['returncode']==0 for k in ['gpu','compute','processes']) and UUIDS[node] in res['gpu']['output'] and not res['compute']['output'].strip()
   prefix='counterfactual_diagnostics_20261005T0604Z/';mode=MODES[node];r=json.loads(z.read(prefix+'run_'+mode+'/results.json'));launch=json.loads(z.read(prefix+'launch.json'));log=z.read(prefix+'diagnostic.log').decode()
   assert 'Traceback' not in log and 'COUNTERFACTUAL_FROZEN_DEV_DIAGNOSTICS_COMPLETE' in log
   assert r['status']=='COUNTERFACTUAL_FROZEN_DEV_DIAGNOSTICS_COMPLETE' and r['mode']==launch['mode']==mode and r['gpu_uuid']==launch['gpu_uuid']==UUIDS[node]
   retired={str(launch['pid']),str({'a':7185,'b':7510,'c':11630}[node])}
   assert not any(line.split() and line.split()[0] in retired for line in res['processes']['output'].splitlines())
   assert r['source_sha256']==launch['source_sha256']==digest(Path(__file__).parent/'diagnose_counterfactual_v5_v1.py') and r['checkpoint_sha256']==chosen['checkpoint_sha256'] and r['protocol_sha256']==chosen['selection']['protocol_sha256']
   assert r['model_state_before_sha256']==r['model_state_after_sha256'] and r['optimizer_updates']==0 and r['test_accessed'] is False
   assert set(r['conditions'])==set(CONDITIONS) and set(r['original_method_equivalence_maxabs'])==set(CONDITIONS[:4]) and max(r['original_method_equivalence_maxabs'].values())<=2e-5
   raw=z.read(prefix+'run_'+mode+'/predictions.npz');assert hashlib.sha256(raw).hexdigest()==r['output_sha256']
   with np.load(io.BytesIO(raw),allow_pickle=False) as f:d={k:f[k] for k in f.files}
   widths={'own':3,'pair':6,'utility':6,'weights':6,'predicted_weights':6,'reference_prediction':None,'feedback_norm':6,'context_norm':3}
   assert set(d)=={'valid_y','donor_permutation'}|{k+'_'+n for n in CONDITIONS for k in ['pred',*widths]};assert all(np.isfinite(v).all() for v in d.values())
   with np.load(Path(chosen['checkpoint']).parent/'predictions.npz',allow_pickle=False) as saved:
    assert np.array_equal(d['valid_y'],saved['valid_y']);assert np.allclose(d['pred_default'],saved['valid_pred'],atol=2e-5,rtol=2e-5) and np.allclose(d['pred_off'],saved['condition_off_pred'],atol=2e-5,rtol=2e-5)
    for k in ['own','pair','utility','weights','predicted_weights','reference_prediction']:assert np.allclose(d[k+'_default'],saved[k],atol=2e-5,rtol=2e-5)
   assert np.array_equal(d['donor_permutation'],np.concatenate([np.roll(np.arange(128),1),np.roll(np.arange(128,229),1)]))
   y=d['valid_y'].astype(float);base=d['pred_default'].astype(float);fixed=d['pred_forced_fixed'].astype(float);risks=[]
   assert y.shape==(229,) and np.allclose(d['reference_prediction_default'].reshape(-1),fixed,atol=2e-5,rtol=2e-5)
   for n in CONDITIONS:
    p=d['pred_'+n];assert p.shape==(229,)
    for k,w in widths.items():assert d[k+'_'+n].shape==((229,w) if w else (229,))
    u=d['utility_'+n];pw=d['predicted_weights_'+n];w=d['weights_'+n];assert np.max(np.abs(u))<=1 and np.allclose(pw,1/(1+np.exp(-4*u)),atol=1e-7,rtol=1e-7)
    for k in ['own','pair','utility','predicted_weights','reference_prediction']:assert np.array_equal(d[k+'_'+n],d[k+'_default'])
    effective=r['selected_effective_mode']
    if n=='off':expect=np.zeros_like(w)
    elif n.startswith('fixed_') or n=='forced_fixed':expect=np.full_like(w,.5)
    elif n=='forced_predicted':expect=pw.copy()
    else:expect={'none':np.zeros_like(w),'fixed':np.full_like(w,.5),'predicted':pw.copy()}[effective]
    disabled=[]
    if n.startswith('disable_') or n.startswith('fixed_disable_'):disabled=[int(n[-1])]
    if n.startswith('joint_disable_') or n.startswith('fixed_joint_disable_'):disabled=[2*int(n[-1]),2*int(n[-1])+1]
    expect[:,disabled]=0;assert np.array_equal(w,expect)
    assert all(d[k+'_'+n].min()>=0 for k in ['feedback_norm','context_norm'])
    if 'shift_donor' not in n:assert np.array_equal(d['feedback_norm_'+n],d['feedback_norm_default'])
    if effective=='none' and (n=='default' or n.startswith(('disable_','joint_disable_','shift_donor'))):assert np.allclose(p,base,atol=2e-5,rtol=2e-5)
    m=measured(p,d['valid_y']);assert set(m)==set(r['conditions'][n]) and all(abs(v-r['conditions'][n][k])<1e-6 for k,v in m.items())
    g,err=gain(base,p.astype(float),y);risks.append({'condition':n,'metrics':m,'default_retention_benefit':float(g.mean()),'risk_identity_maxabs':err,'displacement_rms':float(np.sqrt(np.square(p-base).mean()))})
   channels=[];u=d['utility_default'].astype(float)
   for i,name in enumerate(NAMES):
    gf,_=gain(fixed,d['pred_fixed_disable_'+str(i)].astype(float),y);gn,_=gain(base,d['pred_disable_'+str(i)].astype(float),y);donor,_=gain(base,d['pred_shift_donor_'+str(i)].astype(float),y)
    channels.append({'direction':name,'fixed_reference_calibration':calibration(u[:,i],gf),'natural_policy_calibration':calibration(u[:,i],gn),'donor_replacement_mean_risk_increase':float(donor.mean())})
   joints=[]
   for m in range(3):
    for policy,b in [('natural',base),('fixed',fixed)]:
     pre='' if policy=='natural' else 'fixed_';ps=[d['pred_'+pre+'disable_'+str(i)].astype(float) for i in [2*m,2*m+1]];joint=d['pred_'+pre+'joint_disable_'+str(m)].astype(float)
     gi=[gain(b,p,y)[0] for p in ps];gj=gain(b,joint,y)[0];di=[p-b for p in ps];dj=joint-b;nonadd=dj-di[0]-di[1];interaction=gj-gi[0]-gi[1];identity=2*(b-y)*nonadd+dj*dj-di[0]*di[0]-di[1]*di[1];assert np.max(np.abs(identity-interaction))<1e-10
     joints.append({'receiver':m,'policy':policy,'joint_gain_mean':float(gj.mean()),'risk_interaction_mean':float(interaction.mean()),'prediction_nonadditivity_rms':float(np.sqrt(np.square(nonadd).mean()))})
   rows.append({'node':node,'mode':mode,'captured_at':res['captured_at'],'archive_sha256':proof['archive_sha256'],'members':len(members),'diagnostic':r,'risks':risks,'channels':channels,'joints':joints})
 report={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'COUNTERFACTUAL_29_CONDITION_FROZEN_DEV_ARRAYS_AUDITED','rows':rows,'auditor_sha256':digest(__file__),'limits':'Single exploratory seed; observed DEV; task utility is not semantic truth; inference perturbation is not retraining.'}
 with c.out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({'status':report['status'],'nodes':c.nodes}))
if __name__=='__main__':main()
