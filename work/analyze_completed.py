from pathlib import Path
import datetime,hashlib,io,json,zipfile
import numpy as np
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105/inflow_conditions_20261005022433Z')
OUT=Path('outputs')
diagnostics={}
arrays={}
for node in ('a','b','c'):
    folder=Path('work/diagnostics')/node
    proof=json.loads((folder/'proof.json').read_text())
    raw=(folder/'snapshot.zip').read_bytes()
    assert len(raw)==proof['archive_bytes'] and hashlib.sha256(raw).hexdigest()==proof['archive_sha256']
    with zipfile.ZipFile(folder/'snapshot.zip') as z:
        members=json.loads(z.read('member_manifest.json'))
        assert members==proof['members'] and set(z.namelist())==set(members)|{'member_manifest.json'}
        for name,entry in members.items():
            data=z.read(name)
            assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
        result=json.loads(z.read('results/results.json'))
        assert result['status']=='FROZEN_DEV_DIAGNOSTICS_COMPLETE'
        assert result['model_state_before_sha256']==result['model_state_after_sha256']
        assert result['default_replay_maxabs']==result['off_replay_maxabs']==0
        assert result['test_accessed'] is False and result['optimizer_updates']==0
        assert result['source_sha256']==hashlib.sha256(Path('work/diagnose_conditions.py').read_bytes()).hexdigest()
        data=z.read('results/predictions.npz')
        assert hashlib.sha256(data).hexdigest()==result['output_sha256']
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:
            a={k:archive[k] for k in archive.files}
        resources=json.loads(z.read('actual_resources.json'))
        assert not resources['compute'].strip() and result['gpu_uuid'] in resources['gpu']
        with np.load(BASE/node/'run/predictions.npz',allow_pickle=False) as saved:
            assert np.array_equal(a['valid_y'],saved['valid_y'])
            assert np.array_equal(a['pred_default'],saved['valid_pred'])
            assert np.array_equal(a['pred_off'],saved['condition_off_pred'])
        y=a['valid_y'].astype('float64')
        for name,m in result['conditions'].items():
            pred=a['pred_'+name].astype('float64')
            assert abs(np.abs(pred-y).mean()-m['MAE'])<3e-7
            assert abs(((pred-y)**2).mean()-m['MSE'])<3e-7
        diagnostics[node]=result;arrays[node]=a
basepred=arrays['a']['pred_default'].astype('float64')
rows=[]
for node in ('a','b','c'):
    audit=json.loads((BASE/node/'completion_audit.json').read_text())
    a=arrays[node];y=a['valid_y'].astype('float64')
    on=a['pred_default'].astype('float64');off=a['pred_off'].astype('float64')
    delta=on-off;residual=off-y
    change=(on-y)**2-residual**2
    linear=2*residual*delta;quadratic=delta**2
    assert np.allclose(change,linear+quadratic,atol=1e-12,rtol=1e-12)
    diff=on-basepred;trajectory=off-basepred
    assert np.allclose(diff,delta+trajectory,atol=1e-12)
    maechange=np.abs(on-y)-np.abs(off-y)
    worst=np.argsort(change)[-10:][::-1]
    rows.append({'node':node,'mode':audit['mode'],'selection':audit['selection'],'metrics':audit['metrics'],
        'mean_uniform_mse_change_on_vs_off':float(change.mean()),'linear_residual_alignment':float(linear.mean()),
        'quadratic_perturbation_cost':float(quadratic.mean()),'mean_prediction_shift':float(delta.mean()),
        'squared_error_improved':int((change<-1e-10).sum()),'squared_error_worsened':int((change>1e-10).sum()),
        'absolute_error_improved':int((maechange<-1e-10).sum()),'absolute_error_worsened':int((maechange>1e-10).sum()),
        'inference_condition_shift_rms':float(np.sqrt(np.mean(delta**2))),
        'trained_off_vs_independent_none_rms':float(np.sqrt(np.mean(trajectory**2))),
        'total_vs_independent_none_rms':float(np.sqrt(np.mean(diff**2))),
        'top_ten_error_increases':[{'dev_index':int(i),'label':float(y[i]),'default_pred':float(on[i]),'off_pred':float(off[i]),'squared_error_increase':float(change[i])} for i in worst],
        'diagnostics':diagnostics[node]})
report={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'COMPLETION_AND_FROZEN_DIAGNOSTIC_ARRAYS_INDEPENDENTLY_VERIFIED','rows':rows,
        'limits':['Exploratory 229 DEV cases, not independent confirmation; one seed91811 and condition-node binding.',
            'Squared risk decomposition is an identity, not causal attribution of training effects.',
            'Uniform sample MSE differs from author unweighted mean of 128/101 batch MSE used for selection.',
            'Weak unimodal heads do not establish absence of complementary multimodal information.']}
(OUT/'条件流实验核验与诊断.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'verified_at':report['verified_at'],'rows':[{k:r[k] for k in ['mode','mean_uniform_mse_change_on_vs_off','linear_residual_alignment','quadratic_perturbation_cost','mean_prediction_shift','squared_error_improved','squared_error_worsened','inference_condition_shift_rms','trained_off_vs_independent_none_rms']} for r in rows]},indent=2))
