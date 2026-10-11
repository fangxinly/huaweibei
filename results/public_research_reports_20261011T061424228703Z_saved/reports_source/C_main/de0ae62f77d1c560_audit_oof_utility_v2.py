"""Independent NumPy audit. No Torch; old arrays reused without GPU replay."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--old-arrays',type=Path,required=True);p.add_argument('--oof',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-uuid')
    a=p.parse_args();r=a.directory;plan=json.loads((r/'plan.json').read_text(encoding='utf-8'))
    assert sha(a.old_arrays)=='92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658'
    assert sha(a.oof)=='cff1a52f7e828c96cacf070ae4e051a5c34042ebb9435f44f5b3242cad91a7ea'
    assert sha(r/'diagnose_oof_teacher_utility_v1.py')==plan['source_sha256']
    rec=json.loads((r/'execute/receipt.json').read_text());gate=json.loads((r/'precheck/receipt.json').read_text())
    assert gate['passed'] and rec['passed'] and rec['plan_sha256']==sha(r/'plan.json')
    assert all(json.loads((r/(phase+'_exit.json')).read_text())['exit_code']==0 for phase in ['precheck','execute'])
    assert rec['model_state_before']==rec['model_state_after']==gate['model_state_before']==gate['model_state_after']
    assert rec['optimizer_steps']==0 and rec['no_parameter_grads']
    assert rec['peak_allocated_bytes']<=plan['maximum_peak_allocated_bytes'] and rec['seconds']<=plan['maximum_seconds']
    assert rec['rows']==1281 and rec['videos']==52 and rec['batches']==41
    freeze=json.loads((r/'execute/prediction_freeze.json').read_text())
    assert not freeze['real_labels_read'] and freeze['sha256']==rec['prediction_sha256']==sha(r/'execute/predictions_frozen.npz')
    assert sha(r/'execute/metric_labels.npz')==rec['labels_sha256']
    labels=np.load(r/'execute/metric_labels.npz',allow_pickle=False)
    z=np.load(r/'execute/predictions_frozen.npz',allow_pickle=False)
    old=np.load(a.old_arrays,allow_pickle=False);oof=np.load(a.oof,allow_pickle=False)
    n=1281;rows=np.arange(n);assert np.array_equal(z['row'],rows) and np.array_equal(labels['row'],rows)
    assert np.array_equal(z['video'],old['video']) and len(np.unique(z['video']))==52
    assert np.array_equal(z['fold'],oof['fold']) and np.array_equal(z['mu'],oof['mu'])
    assert np.array_equal(z['p0'],old['p0']) and np.array_equal(labels['y'],old['y'])
    assert np.array_equal(labels['y'],oof['y'].astype(labels['y'].dtype))
    for k in z.files:
        if np.issubdtype(z[k].dtype,np.number):assert np.isfinite(z[k]).all(),k
    pred=z['prediction_path'].astype('float64');prox=z['prox_path'].astype('float64')
    assert pred.shape==prox.shape==(n,4)
    assert np.max(np.abs(pred[:,0]-old['raw_prediction']))<=1e-6
    assert np.max(z['relative_path'])<=.25001 and np.all(z['relative_path'][:,0]==0)
    assert np.all(prox>=0) and np.all(prox[:,0]==0)
    assert z['message_distance_path'].shape==z['message_cosine_path'].shape==(n,4,6)
    target=z['mu'].astype('float32').astype('float64')
    assert np.array_equal(target,z['mu'])  # Teacher originals were float32 predictions.
    risk=(pred-target[:,None])**2
    assert np.array_equal(risk,z['mu_risk_path'])
    index=risk.argmin(1);assert np.array_equal(index,z['selected_step'])
    selected=pred[rows,index];assert np.array_equal(selected,z['selected_prediction'])
    assert np.all(risk[rows,index]<=risk[:,0])
    scale=np.float64(np.float32(.22068938092649817))**2
    beta=np.float64(np.float32(plan['beta']))
    p0=z['p0'].astype('float64');rho=(z['p0'].astype('float32')-z['mu'].astype('float32')).astype('float64')
    delta=pred-p0[:,None]
    reconstructed=prox+beta*(2*rho[:,None]*delta+delta**2)/scale
    objective=z['objective_path'].astype('float64')
    relative_error=float(np.max(np.abs(reconstructed-objective)/(1+np.abs(reconstructed)+np.abs(objective))))
    assert relative_error<5e-5
    y=labels['y'].astype('float64');raw=pred[:,0];videos=z['video'];metrics={}
    for name,value in [('raw',raw),('oof_last',pred[:,-1]),('oof_mu_best',selected),('old_learned',old['original_prediction'].astype('float64'))]:
        e=value-y;q=e**2-(raw-y)**2;changed=np.abs(value-raw)>1e-8;d=value-raw
        values=dict(mse=float(np.mean(e**2)),mae=float(np.mean(np.abs(e))),
            video_equal_mse=float(np.mean([np.mean(e[videos==v]**2) for v in np.unique(videos)])),
            changed_rows=int(changed.sum()),harmful_fraction_changed=float(np.mean(q[changed]>1e-12)) if changed.any() else None,
            improved_fraction_all=float(np.mean(q< -1e-12)),mean_risk_change=float(q.mean()))
        for k,x in values.items():
            actual=rec['metrics'][name][k]
            assert (x is None and actual is None) or (x is not None and abs(actual-x)<1e-12),(name,k)
        qhat=2*(raw-z['mu'])*d+d**2
        identity=float(np.max(np.abs(q-qhat-2*(z['mu']-y)*d)))
        assert identity<1e-12
        values.update(relative_mse_gain_vs_original_F=float(1-values['mse']/rec['metrics']['raw']['mse']),
            proxy_risk_mean=float(qhat.mean()),proxy_observed_identity_error=identity,
            video_improved_count=int(sum(np.mean(q[videos==v])<0 for v in np.unique(videos))))
        metrics[name]=values
    assert rec['selected_step_counts']==np.bincount(index,minlength=4).tolist()
    cpu=None
    if a.expected_uuid:
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used','--format=csv,noheader,nounits'],text=True).strip()
        assert gpu.split(',')[0]==a.expected_uuid
        assert 'torch' not in sys.modules
        cpu=dict(gpu_inventory=gpu,torch_imported=False,cuda_forward_executed=False)
    report=dict(status='NEW_OOF_MU_PATH_ORIGINAL_ARRAYS_SELECTOR_RISK_AND_METRICS_INDEPENDENTLY_RECONSTRUCTED',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=n,videos=52,
        selected_step_counts=rec['selected_step_counts'],objective_reconstruction_relative_error=relative_error,
        metrics=metrics,files={str(x.relative_to(r)):dict(bytes=x.stat().st_size,sha256=sha(x)) for x in sorted(r.rglob('*')) if x.is_file() and x.suffix in ['.json','.npz','.py','.log']},
        preserved_old_array_sha256=sha(a.old_arrays),OOF_sha256=sha(a.oof),independent_cpu=cpu,
        scope=plan['limits'],generator_and_selector_labels_used=False,parameter_update=False)
    assert not a.out.exists();a.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],selected_steps=report['selected_step_counts'],metrics=metrics)))

if __name__=='__main__':main()
