from pathlib import Path
import json,hashlib,zipfile,io,datetime
import numpy as np
w=Path(__file__).parent;root=w.parent;base=Path('D:/CodexBackups/selective_flow_20261003_1105');shots=base/'finite_c2_completed_snapshots_20261005T1627Z'
sha=lambda b:hashlib.sha256(b).hexdigest();rows={};c2=None
def corr(x,y):
 x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 return None if x.std()==0 or y.std()==0 else float(np.corrcoef(x,y)[0,1])
def ranking(a):
 # Tied ranks, deterministic within each direction across 229 samples.
 order=np.argsort(a,kind='stable');out=np.empty(len(a),float);v=np.asarray(a)[order];i=0
 while i<len(a):
  j=i+1
  while j<len(a) and v[j]==v[i]:j+=1
  out[order[i:j]]=(i+j-1)/2;i=j
 return out
for node in ['a','b','c']:
 d=shots/node;r=json.loads((d/'receipt.json').read_text());assert sha((d/'snapshot.zip').read_bytes())==r['sha256']
 with zipfile.ZipFile(d/'snapshot.zip') as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  m=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
  for n,v in m.items():assert sha(z.read(n))==v['sha256'] and len(z.read(n))==v['bytes']
  get=lambda n:json.loads(z.read(n))
  assert sha(z.read('source/capture_soft_vector_v13.py'))==sha((w/'capture_soft_vector_v13.py').read_bytes())==get('inventory.json')['capture_source_sha256']
  if node in ['a','b','c']:
   pre='finite_c2/diagnostics_v4/' if node=='c' else 'finite_diagnostics_v3/';rr=get(pre+'diagnostics_receipt.json');arr=np.load(io.BytesIO(z.read(pre+'diagnostics.npz')))
   formal='finite_c2/' if node=='c' else 'finite_formal_source/';version='v4' if node=='c' else 'v3';run='finite_c2/run/' if node=='c' else 'finite_run/'
   assert rr['source_sha256']==sha(z.read(formal+'diagnose_finite_task_risk_'+version+'.py'))==sha((w/('diagnose_finite_task_risk_'+version+'.py')).read_bytes())
   assert rr['plan_sha256']==sha(z.read(formal+'finite_diagnostics_plan_'+version+'.json'))==sha((w/('finite_diagnostics_plan_'+version+'.json')).read_bytes())
   assert rr['diagnostics_sha256']==sha(z.read(pre+'diagnostics.npz')) and rr['tensor_sha_before']==rr['tensor_sha_after'] and not rr['test_requested']
   assert rr['default_cached_replay_error']==rr['label_replacement_error']==rr['context_replay_error']==0 and rr['finite_squared_identity_error']<1e-12
   assert len(rr['conditions'])==20 and rr['selection']==get(run+'selection.json')
   assert rr['addon_sha256']==m[run+'best_addon.pt']['sha256']
   for n in arr.files:assert len(arr[n])==229 and np.isfinite(arr[n]).all()
   saved=np.load(io.BytesIO(z.read(run+'predictions.npz')));assert np.array_equal(arr['prediction_default'],saved['valid_pred']) and np.array_equal(arr['valid_y'],saved['valid_y'])
   assert np.array_equal(arr['prediction_alloff'],arr['reference_recomputed'])
   y=arr['valid_y'].astype('float64');p0=arr['reference_prediction'].astype('float64');rw=arr['raw_fixed_prediction'].astype('float64');ro=arr['raw_without_predictions'].astype('float64');rho=arr['estimated_residual'].astype('float64');delta=rw-p0;without=ro-p0[:,None]
   algebra=2*(p0-y)[:,None]*(without-delta[:,None])+without**2-delta[:,None]**2
   true=(ro-y[:,None])**2-(rw-y)[:,None]**2;err=float(np.max(np.abs(algebra-true)));assert err<1e-12
   proxy=2*rho[:,None]*(without-delta[:,None])+without**2-delta[:,None]**2
   assert np.max(np.abs(proxy-arr['estimated_raw_finite_utility']))<2e-6 and np.max(np.abs(true-arr['true_raw_finite_utility']))<4e-6
   final=np.stack([(arr['prediction_off'+str(i)].astype('float64')-y)**2-(arr['prediction_default'].astype('float64')-y)**2 for i in range(6)],1)
   assert np.max(np.abs(final-arr['true_final_transmission_gain']))<4e-6
   metrics={}
   for name in rr['conditions']:
    pred=arr['prediction_'+name].astype('float64');change=(pred-y)**2-(arr['prediction_default'].astype('float64')-y)**2
    metrics[name]={'mae':float(np.mean(np.abs(pred-y))),'mse':float(np.mean((pred-y)**2)),'risk_change_vs_default':float(change.mean()),'benefit_fraction_vs_default':float(np.mean(change<0))}
   channels=[]
   for i in range(6):
    channels.append(dict(direction=i,raw_proxy_mean=float(proxy[:,i].mean()),raw_true_mean=float(true[:,i].mean()),raw_sign_agreement=float(np.mean((proxy[:,i]>0)==(true[:,i]>0))),raw_spearman=corr(ranking(proxy[:,i]),ranking(true[:,i])),final_transmission_gain_mean=float(final[:,i].mean()),final_sign_agreement_with_raw_proxy=float(np.mean((proxy[:,i]>0)==(final[:,i]>0))),final_spearman_with_raw_proxy=corr(ranking(proxy[:,i]),ranking(final[:,i]))))
   rows[node]=dict(receipt=rr,metrics=metrics,channels=channels,independent_float64_identity_error=err,raw_best_direction_agreement=float(np.mean(np.argmax(proxy,axis=1)==np.argmax(true,axis=1))),final_best_direction_agreement=float(np.mean(np.argmax(proxy,axis=1)==np.argmax(final,axis=1))),scope='Raw fixed candidate risk versus final controlled transmitted-message removal are different estimands. DEV-only frozen exploratory diagnostics after DEV selection; no causal semantic truth or stable result.')
  if node=='c':
   inv=get('finite_c2_inventory.json');pr=get('finite_c2/run/protocol.json');plan=get('finite_c2/finite_formal_plan_v2.json');h=get('finite_c2/run/history.json')
   assert pr['seed']==91817 and pr['epochs']==100 and not pr['test_requested'] and pr['gpu_uuid']==inv['gpu'].split(',')[0]=='GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'
   assert pr['formal_plan_sha256']==sha(z.read('finite_c2/finite_formal_plan_v2.json'))==sha((w/'finite_formal_plan_v2.json').read_bytes())
   for n,v in plan['source_sha256'].items():assert sha(z.read('finite_c2/'+n))==v==pr['source_sha256'][n]
   assert [x['epoch'] for x in h]==list(range(1,len(h)+1)) and 10<=len(h)<=100
   assert all(x['effective_mode']==('fixed' if x['epoch']<=10 else 'finite_vector') for x in h)
   assert sha(z.read('finite_c2/run/shared_phase_addon.pt'))==sha(z.read('finite_run/shared_phase_addon.pt'))=='32fad83f206b32f4070fe2bbf0b3a588c1e8a5340c58b337a8e7052161c08088'
   assert sha(z.read('finite_c2/run/orders.npy'))==pr['orders_sha256']==plan['orders_sha256']
   sel=get('finite_c2/run/selection.json');assert len(h)==sel['epochs']==100 and sel['best_epoch']==np.argmin([x['dev_author_batch_mean_mse'] for x in h])+1
   assert inv['exit']['exit_code']==0 and inv['actual_process_argv'] in [None,[]]
   assert sel['effective_mode']==('fixed' if sel['best_epoch']<=10 else 'finite_vector') and sel['frozen_tensor_sha_before']==sel['frozen_tensor_sha_after'] and sel['selected_prediction_replay_max_error']==0
   assert m['finite_c2/run/best_addon.pt']['sha256']==sel['addon_sha256'] and m['finite_c2/run/predictions.npz']['sha256']==sel['prediction_sha256']
   c2=dict(state='C2_ACTUAL_100_EXIT0_COMPLETE',epochs=len(h),selection=sel,last_epoch=h[-1],capture_inventory=inv,plan_sha256=pr['formal_plan_sha256'])
out=dict(status='FINITE_AB_AND_C2_FROZEN20_ARRAYS_AND_C2_ACTUAL100_SNAPSHOT_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,c2=c2)
(root/'outputs/有限任务风险两对照及C2三组20条件诊断与100完成独立核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'arms':{n:dict(default=r['metrics']['default'],channels=r['channels'],alloff=r['metrics']['alloff']) for n,r in rows.items()},'c2_epochs':c2['epochs']}))
