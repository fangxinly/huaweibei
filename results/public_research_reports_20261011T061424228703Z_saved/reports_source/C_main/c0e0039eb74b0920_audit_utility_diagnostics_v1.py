"""Independent proof/member/array/identity checks of the fixed 24 DEV conditions."""
from pathlib import Path
import argparse,datetime,hashlib,io,json,zipfile
import numpy as np
from analyze_utility_completed_v1 import calibration,corr,digest,rank
MODES={'a':'none','b':'fixed','c':'predicted'}
UUIDS={'a':'GPU-5902bbd4-2328-0822-777c-1311d53ee5a3','b':'GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
PIDS={'a':6508,'b':6830,'c':10862}
HERE=Path(__file__).resolve().parent
CONDITIONS=['default','off','forced_fixed','forced_predicted']+[f'disable_{i}' for i in range(6)]+[f'shift_feedback_{i}' for i in range(6)]+['shift_feedback_all']+[f'shift_full_{i}' for i in range(6)]+['shift_full_all']
def measured(pred,y):
    truth=y!=0
    return {'MAE':float(np.abs(pred-y).mean()),'MSE':float(np.square(pred-y).mean()),'author_batch_mse':float(np.mean([np.square(pred[i:i+128]-y[i:i+128]).mean() for i in range(0,len(y),128)])),
        'Non0_acc2':float(((pred[truth]>=0)==(y[truth]>=0)).mean()),'bias':float((pred-y).mean()),'Corr':float(np.corrcoef(pred,y)[0,1])}
def main():
    a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True);a.add_argument('--completed-audit',type=Path,required=True);a.add_argument('--out',type=Path,required=True);c=a.parse_args()
    assert not c.out.exists();complete=json.loads(c.completed_audit.read_text(encoding='utf-8'))
    assert complete['status']=='UTILITY_SELECTED_NODES_COMPLETED_WHOLE_SHA_100_SELECTION_OFFICIAL_DEV_VERIFIED'
    selected={r['node']:r for r in complete['rows']};assert set(selected)==set(MODES)
    rows=[]
    for node,mode in MODES.items():
        folder=c.directory/node;proof=json.loads((folder/'proof.json').read_text(encoding='utf-8'))
        assert digest(folder/'snapshot.zip')==proof['archive_sha256'] and (folder/'snapshot.zip').stat().st_size==proof['archive_bytes']
        assert proof['capture_source_sha256']==digest(HERE/'capture_utility_followup_v1.py')
        with zipfile.ZipFile(folder/'snapshot.zip') as z:
            members=json.loads(z.read('member_manifest.json'));assert members==proof['members']
            assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(members)|{'member_manifest.json'}
            for name,expect in members.items():
                raw=z.read(name);assert len(raw)==expect['bytes'] and hashlib.sha256(raw).hexdigest()==expect['sha256'],name
            resources=json.loads(z.read('actual_resources.json'))
            assert all(resources[k]['returncode']==0 for k in ['gpu','compute','processes'])
            assert UUIDS[node] in resources['gpu']['output'] and not resources['compute']['output'].strip()
            retired={PIDS[node],selected[node]['pid']}
            assert not any(line.split() and line.split()[0] in {str(p) for p in retired} for line in resources['processes']['output'].splitlines())
            prefix='utility_diagnostics_20261005T0439Z/'
            r=json.loads(z.read(prefix+'run_'+mode+'/results.json'));launch=json.loads(z.read(prefix+'launch.json'))
            assert 'Traceback' not in z.read(prefix+'diagnostic.log').decode() and 'UTILITY_FROZEN_DEV_DIAGNOSTICS_COMPLETE' in z.read(prefix+'diagnostic.log').decode()
            assert launch['pid']==PIDS[node] and launch['mode']==r['mode']==mode and launch['gpu_uuid']==r['gpu_uuid']==UUIDS[node]
            assert r['status']=='UTILITY_FROZEN_DEV_DIAGNOSTICS_COMPLETE' and r['optimizer_updates']==0 and r['test_accessed'] is False
            assert r['source_sha256']==launch['source_sha256']==digest(HERE/'diagnose_utility_v4_v1.py')
            assert r['checkpoint_sha256']==selected[node]['checkpoint_sha256'] and r['protocol_sha256']==selected[node]['selection']['protocol_sha256']
            assert r['model_state_before_sha256']==r['model_state_after_sha256']
            assert set(r['conditions'])==set(CONDITIONS) and all(v<=2e-5 for v in r['original_method_equivalence_maxabs'].values())
            assert set(r['original_method_equivalence_maxabs'])==set(CONDITIONS[:4])
            raw=z.read(prefix+'run_'+mode+'/predictions.npz');assert hashlib.sha256(raw).hexdigest()==r['output_sha256']
            with np.load(io.BytesIO(raw),allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
            expected={'valid_y','donor_permutation'}|{key+'_'+name for name in CONDITIONS for key in ['pred','own','pair','utility','weights','predicted_weights','feedback_norm','context_norm','state_norm']}
            assert set(data)==expected and all(np.isfinite(v).all() for v in data.values())
            with np.load(Path(selected[node]['checkpoint']).parent/'predictions.npz',allow_pickle=False) as saved:
                assert np.array_equal(data['valid_y'],saved['valid_y'])
                for key in ['own','pair','utility','predicted_weights']:assert np.allclose(data[key+'_default'],saved[key],atol=2e-5,rtol=2e-5)
                assert np.allclose(data['pred_default'],saved['valid_pred'],atol=2e-5,rtol=2e-5)
                assert np.allclose(data['pred_off'],saved['condition_off_pred'],atol=2e-5,rtol=2e-5)
            permutation=np.concatenate([np.roll(np.arange(128),1),np.roll(np.arange(128,229),1)])
            assert np.array_equal(data['donor_permutation'],permutation) and np.all(permutation!=np.arange(229))
            y=data['valid_y'];base=data['pred_default'].astype(float);risk=[]
            for name in CONDITIONS:
                pred=data['pred_'+name];assert pred.shape==(229,)
                for key,width in [('own',3),('pair',6),('utility',6),('weights',6),('predicted_weights',6),('feedback_norm',6),('context_norm',3),('state_norm',3)]:assert data[key+'_'+name].shape==(229,width)
                assert np.max(np.abs(data['utility_'+name]))<=1
                assert np.allclose(data['predicted_weights_'+name],1/(1+np.exp(-4*data['utility_'+name])),atol=1e-7,rtol=1e-7)
                for key in ['weights','predicted_weights']:assert data[key+'_'+name].min()>=0 and data[key+'_'+name].max()<=1
                for key in ['feedback_norm','context_norm','state_norm']:assert data[key+'_'+name].min()>=0
                if name=='off' or (mode=='none' and name not in ['forced_fixed','forced_predicted']):assert np.max(np.abs(data['weights_'+name]))==0
                if name=='forced_fixed' or (mode=='fixed' and name not in ['off','forced_predicted']):assert np.array_equal(data['weights_'+name],np.full((229,6),.5))
                if name=='forced_predicted' or (mode=='predicted' and name not in ['off','forced_fixed']):assert np.array_equal(data['weights_'+name],data['predicted_weights_'+name])
                assert np.array_equal(data['own_'+name],data['own_default'])
                if not name.startswith('shift_full'):
                    for key in ['pair','utility','predicted_weights']:assert np.array_equal(data[key+'_'+name],data[key+'_default'])
                if name.startswith('disable_'):assert np.max(np.abs(data['feedback_norm_'+name][:,int(name[-1])]))==0
                if mode=='none':assert np.allclose(pred,base,atol=2e-5,rtol=2e-5)
                measure=measured(pred,y);assert set(measure)==set(r['conditions'][name])
                assert all(abs(value-r['conditions'][name][k])<1e-6 for k,value in measure.items())
                delta=base-pred.astype(float);e=pred.astype(float)-y.astype(float)
                benefit=e*e-np.square(base-y.astype(float));residual=float(np.max(np.abs(benefit-(-2*e*delta-delta*delta))));assert residual<1e-10
                risk.append({'condition':name,'metrics':measure,'default_retention_benefit':float(benefit.mean()),'positive_benefit_fraction':float((benefit>1e-10).mean()),'negative_benefit_fraction':float((benefit< -1e-10).mean()),'prediction_delta_rms':float(np.sqrt(np.square(delta).mean())),'risk_identity_maxabs':residual})
            channels=[];u=data['utility_default'].astype(float);own=data['own_default'].astype(float);pair=data['pair_default'].astype(float)
            for i,m in enumerate([0,0,1,1,2,2]):
                pred=data['pred_disable_'+str(i)].astype(float);benefit=np.square(pred-y)-np.square(base-y)
                q=np.square(own[:,m]-y)-np.square(pair[:,i]-y)
                channels.append({'channel':i,'u_vs_actual_retention':calibration(u[:,i],benefit),'teacher_q_vs_actual_spearman':corr(rank(q),rank(benefit)),
                    'mean_actual_retention_benefit':float(benefit.mean()),'u_mean':float(u[:,i].mean()),'q_mean':float(q.mean())})
            rows.append({'node':node,'mode':mode,'captured_at':resources['captured_at'],'gpu':resources['gpu']['output'],'disk_free_bytes':resources['disk_free_bytes'],
                'archive_sha256':proof['archive_sha256'],'members':len(members),'diagnostic':r,'risks':risk,'channels':channels})
    report={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ALL_THREE_24_CONDITION_GPU_DIAGNOSTICS_ARRAYS_AND_RISK_IDENTITIES_VERIFIED','rows':rows,'completed_audit_sha256':digest(c.completed_audit),'auditor_sha256':digest(__file__),
        'limits':'Official DEV previously used for selection; inference perturbation is not retraining or semantic truth. Single seed with mode-node binding. CPU independent full-weight copies checked separately.'}
    with c.out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':report['status'],'nodes':[r['node'] for r in rows]}))
if __name__=='__main__':main()
