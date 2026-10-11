from pathlib import Path
import datetime,hashlib,json,numpy as np
r=Path(__file__).resolve().parents[1];w=r/'work';d=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_risk_preflight_20261005T1434Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((w/'finite_mechanism_plan_v1.json').read_text(encoding='utf-8'))
cache=np.load('D:/CodexBackups/selective_flow_20261003_1105/soft_teacher_cache_20261005T1246Z/teacher_cache_v1/train_cache.npz',allow_pickle=False)
residual=cache['reference_prediction']-cache['y'];reports={};common={}
expected={'a':'GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7','b':'GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
for node,uuid in expected.items():
    b=d/node/'checks';p=json.loads((b/'preflight.json').read_text())
    assert p['status']=='FINITE_RISK_REAL_TRAIN_MECHANISMS_CHECKED_NOT_FORMAL_TRAINING'
    assert p['gpu_uuid']==uuid and not p['initial_inventory']['compute'] and uuid in p['inventory']['compute']
    assert not p['dev_requested'] and not p['test_requested'] and p['trainable_parameters']==520506 and p['head_parameters']==214506
    assert p['frozen_tensor_sha_before']==p['frozen_tensor_sha_after']
    for name,h in p['source_sha256'].items():assert sha(w/name)==h==plan['source_sha256'][name]
    for name,h in p['base_source_sha256'].items():assert sha(w/name)==h
    for name,key in [('train_scales.json','scales_sha256'),('head_fit_rows.npy','head_fit_rows_sha256'),('head_heldout_rows.npy','head_heldout_rows_sha256'),('orders.npy','orders_sha256'),('after20_addon.pt','after20_addon_sha256')]:assert sha(b/name)==p[key]
    for key in ['initial_tensor_sha256','after20_tensor_sha256','scales_sha256','orders_sha256','after20_addon_sha256','head_fit_rows_sha256','head_heldout_rows_sha256']:common.setdefault(key,[]).append(p[key])
    fit=np.load(b/'head_fit_rows.npy');held=np.load(b/'head_heldout_rows.npy');order=np.random.RandomState(91990).permutation(1281)
    assert np.array_equal(fit,np.sort(order[:1153])) and np.array_equal(held,np.sort(order[1153:]))
    orders=np.load(b/'orders.npy');assert np.array_equal(orders,np.stack([np.random.RandomState(91817+1327+i).permutation(1281) for i in range(100)]))
    scales=json.loads((b/'train_scales.json').read_text());assert scales==p['scales']
    rms=float(np.sqrt(np.mean(residual[fit].astype(np.float64)**2)))
    assert rms==scales['residual_rms'] and np.all(np.array(scales['message_rms'])>0)
    assert abs(scales['beta']-rms**2/(2*scales['median_reference_normalized_message_sensitivity_norm2']))<1e-8
    assert p['elapsed_seconds']<600 and p['peak_allocated_bytes']<4*1024**3
    for mode,t in p['mode_checks'].items():
        assert t['label_replacement_max_error']==0 and t['train_vs_no_grad_max_error']<2e-6 and t['main_head_gradients_none'] and t['frozen_gradients_none']
    assert p['mode_checks']['finite_vector']['trust_max']<=.25001
    fd=p['unrolled_real_decoder_double_finite_difference'];assert abs(fd['analytic'])>1e-10 and fd['absolute_error']<max(2e-6,.002*abs(fd['analytic']))
    assert p['exact_TRAIN_squared_risk_identity_max_error']<2e-6
    hr=json.loads((b/'TRAIN_head_holdout_receipt.json').read_text())
    assert hr['source_sha256']==sha(w/'inspect_finite_head_holdout_v2.py') and hr['aux_loss_error']==hr['aux_gradient_error']==0
    assert sha(b/'TRAIN_head_holdout_arrays.npz')==hr['arrays_sha256']
    z=np.load(b/'TRAIN_head_holdout_arrays.npz');mask=z['fit_mask'];pred=z['predicted_residual'];y=z['residual_target']
    assert np.array_equal(z['row_id'],np.arange(1281)) and np.array_equal(np.where(mask)[0],fit) and np.array_equal(y,residual)
    metrics={}
    for label,m in [('head_fit',mask),('head_label_holdout',~mask)]:
        observed=dict(rows=int(m.sum()),mse=float(np.mean((pred[m]-y[m])**2)),zero_baseline_mse=float(np.mean(y[m]**2)),sign_agreement=float(np.mean((pred[m]>0)==(y[m]>0))))
        assert observed==hr['metrics'][label]
        # A sign baseline is needed before interpreting a seemingly high agreement.
        sign=bool(np.mean(y[mask]>0)>=.5)
        observed['fit_majority_sign_baseline_agreement']=float(np.mean((y[m]>0)==sign))
        observed['head_prediction_mean']=float(pred[m].mean());metrics[label]=observed
    reports[node]=dict(gpu_uuid=uuid,twenty_seconds=p['twenty_fixed_updates_seconds'],peak_bytes=p['peak_allocated_bytes'],mode_checks=p['mode_checks'],finite_difference=fd,head_metrics=metrics)
for key,values in common.items():assert len(set(values))==1,key
full=json.loads((d/'a/checks/full_inference_check.json').read_text())
assert full['status']=='FINITE_RISK_FULL_RAW_TRAIN32_THREE_MODE_REPLAY_CHECKED_NOT_FORMAL_TRAINING' and full['rows']==32 and full['split']=='train'
assert not full['dev_requested'] and not full['test_requested'] and full['source_sha256']==sha(w/'check_finite_full_inference_v1.py')
assert full['addon_sha256']==common['after20_addon_sha256'][0] and full['scales_sha256']==common['scales_sha256'][0]
for mode,x in full['modes'].items():assert x['full_vs_cached_prediction_max_error']<2e-5 and x['label_replacement_max_error']==0
out=dict(status='THREE_REAL_FINITE_RISK_TRAIN_PREFLIGHTS_AND_FULL_RAW_REPLAY_INDEPENDENTLY_AUDITED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reports=reports,common={k:v[0] for k,v in common.items()},full_inference=full,
    source_sha256=sha(__file__),limits='20fixedpilot and three-mode mechanics only. Head holdout majority baseline shown; not independent validation or a100epoch efficacy result. Inspectionv1 lacked CUBLAS process env and failed before arrays;v1 sources/logs retained, onlyv2 actual arrays/receipts accepted. Formal training not started.')
with (r/'outputs/有限任务风险三GPU机制与原输入独立核验.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=out['status'],common=out['common'],head_metrics=reports['a']['head_metrics'],full_modes=full['modes'])))
