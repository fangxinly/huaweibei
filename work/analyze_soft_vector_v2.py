from pathlib import Path
import datetime,hashlib,io,json,zipfile,numpy as np
r=Path(__file__).resolve().parents[1];d=Path('D:/CodexBackups/selective_flow_20261003_1105/jacobian_completed_20261005');snap=Path('D:/CodexBackups/selective_flow_20261003_1105/jacobian_snapshots_atomic_20261005T1333Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((r/'outputs/Jacobian20条件诊断冻结计划.json').read_text());scale=np.load(Path('D:/CodexBackups/selective_flow_20261003_1105/soft_teacher_cache_20261005T1246Z/teacher_cache_v1/train_gradient_rms.npy')).astype(float)
def corr(x,y):
 x=np.asarray(x,float).reshape(-1);y=np.asarray(y,float).reshape(-1)
 return None if min(np.std(x),np.std(y))<1e-14 else float(np.corrcoef(x,y)[0,1])
results={};arrays={}
for node in 'abc':
 with zipfile.ZipFile(snap/node/'snapshot.zip') as z:
  selected=json.loads(z.read('run/selection.json'));source=json.loads(z.read('run/protocol.json'));saved=np.load(io.BytesIO(z.read('run/predictions.npz')))
  # Export original run artifacts to permanent destination once.
  for name in z.namelist():
   if name.startswith('run/'):
    p=d/node/name[4:]
    if p.exists():assert p.read_bytes()==z.read(name)
    else:p.write_bytes(z.read(name))
 receipt=json.loads((d/node/'diagnostics_receipt.json').read_text());assert receipt['source_sha256']==plan['source_sha256']==sha(r/'work/diagnose_soft_vector_v2.py')
 assert receipt['conditions']==plan['conditions'] and receipt['utc']>plan['utc'] and receipt['rows']==229
 assert receipt['default_cached_replay_max_error']==receipt['label_replacement_max_error']==0
 assert receipt['tensor_sha_before']==receipt['tensor_sha_after'] and receipt['addon_sha256']==selected['addon_sha256']
 assert receipt['diagnostics_sha256']==sha(d/node/'diagnostics.npz') and not receipt['test_requested']
 a={k:v for k,v in np.load(d/node/'diagnostics.npz').items()};arrays[node]=a
 assert all(np.isfinite(v).all() and v.shape[0]==229 for v in a.values())
 assert np.array_equal(a['prediction_default'],saved['valid_pred']) and np.array_equal(a['valid_y'],saved['valid_y'])
 assert np.max(np.abs(a['prediction_alloff']-a['reference_prediction']))<2e-5
 pred=a['predicted_gradient'];truth=a['true_reference_gradient'];message=a['message'];used=np.repeat(pred,2,axis=1);dots=np.sum(used*message,-1);lam=np.repeat(.1*100*scale**2,2)[None,:]
 norm2=np.sum(used**2,-1);ret=lam/(norm2+lam)
 assert np.allclose(a['estimated_raw_message_dot'],dots,atol=1e-8,rtol=2e-5)
 assert np.allclose(a['positive_parallel_retention_ratio'].squeeze(-1),ret,atol=2e-6)
 weights=1/(1+np.exp(.125*dots/.05));assert np.allclose(a['scalar_weights'],weights,atol=1e-6)
 mode=selected['effective_mode'];expected=.5*dots if mode=='fixed' else weights*dots if mode=='scalar' else .5*np.where(dots>0,dots*ret,dots)
 assert np.allclose(a['estimated_transformed_dot'],expected,atol=1e-8,rtol=2e-5)
 y=a['valid_y'];base=(a['prediction_default']-y)**2;benefit=np.stack([(a['prediction_off'+str(i)]-y)**2-base for i in range(6)],1);proxy=-.25*a['estimated_transformed_dot']
 cosine=np.sum(pred*truth,-1)/(np.linalg.norm(pred,axis=-1)*np.linalg.norm(truth,axis=-1)+1e-20)
 learned=np.mean(((pred-truth)/scale[None,:,None])**2,axis=(0,2));zero=np.mean((truth/scale[None,:,None])**2,axis=(0,2))
 result={'best_epoch':selected['best_epoch'],'effective_mode':mode,'mae':float(np.mean(np.abs(a['prediction_default']-y))),'mse':float(np.mean(base)),'gradient_normalized_mse_by_receiver':learned.tolist(),'zero_gradient_baseline_mse_by_receiver':zero.tolist(),'mean_gradient_cosine_by_receiver':np.mean(cosine,0).tolist(),'estimated_to_true_gradient_rms_ratio_by_receiver':(np.sqrt(np.mean(pred**2,axis=(0,2)))/np.sqrt(np.mean(truth**2,axis=(0,2)))).tolist(),'mean_scalar_weight_by_channel':np.mean(a['scalar_weights'],0).tolist(),'mean_parallel_retention_by_channel':np.mean(ret,0).tolist(),'actual_single_channel_benefit_mean':np.mean(benefit,0).tolist(),'actual_single_channel_positive_fraction':np.mean(benefit>0,0).tolist(),'estimated_proxy_vs_actual_benefit_pearson_by_channel':[corr(proxy[:,i],benefit[:,i]) for i in range(6)],'estimated_proxy_vs_actual_sign_agreement_by_channel':[float(np.mean((proxy[:,i]>0)==(benefit[:,i]>0))) for i in range(6)],'all_feedback_benefit_mean':float(np.mean((a['prediction_alloff']-y)**2-base)),'donor_swap_risk_change_mean': [float(np.mean((a['prediction_swap'+str(i)]-y)**2-base)) for i in range(6)],'receiver_shutdown_benefit_mean':[float(np.mean((a['prediction_receiver_off'+str(i)]-y)**2-base)) for i in range(3)],'same_checkpoint_strategy_mse':{m:float(np.mean((a['prediction_'+m]-y)**2)) for m in ['fixed','scalar','soft_projected']},'addon_sha256':selected['addon_sha256'],'diagnostics_sha256':receipt['diagnostics_sha256']}
 results[node]=result
for n in 'bc':assert np.array_equal(arrays['a']['valid_y'],arrays[n]['valid_y']) and np.array_equal(arrays['a']['true_reference_gradient'],arrays[n]['true_reference_gradient'])
comparison={}
for n in 'bc':
 a=arrays['a'];b=arrays[n];diff=np.abs(b['prediction_default']-b['valid_y'])-np.abs(a['prediction_default']-a['valid_y']);rng=np.random.RandomState(620);boot=np.mean(diff[rng.randint(0,229,size=(10000,229))],axis=1)
 comparison[n]={'mae_delta_vs_fixed':float(np.mean(diff)),'mse_delta_vs_fixed':results[n]['mse']-results['a']['mse'],'descriptive_paired_row_bootstrap_mae95':np.percentile(boot,[2.5,97.5]).tolist(),'scope':'Same DEV-selected models and shared teacher; exploratory descriptive uncertainty only, not independent test or multi-seed inference.'}
out={'status':'TWENTY_FROZEN_CONDITIONS_ALL_ARRAYS_AND_PROXY_FORMULAS_INDEPENDENTLY_AUDITED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'diagnostic_plan':plan,'results':results,'comparison':comparison,'scope':'OfficialTRAIN1281/dev229 only; cached frozen teacher coordinates. No TEST; no stable improvement, SOTA or semantic ground-truth claim.'}
(r/'outputs/Jacobian20条件诊断与效用校准独立核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out['results']))
