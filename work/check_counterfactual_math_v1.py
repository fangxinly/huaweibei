from pathlib import Path
import ast,datetime,hashlib,json
import numpy as np
w=Path(__file__).resolve().parent;rng=np.random.default_rng(91814);p=rng.normal(size=229);y=rng.normal(size=229);d=rng.normal(size=(229,6))*.1
direct=(p[:,None]+d-y[:,None])**2-(p[:,None]-y[:,None])**2
identity=d*(2*(p-y)[:,None]+d);assert np.allclose(direct,identity,atol=1e-14,rtol=1e-14)
di,dj=d[:,0],d[:,1];joint=di+dj;gij=(p+joint-y)**2-(p-y)**2;res=gij-direct[:,0]-direct[:,1];assert np.allclose(res,2*di*dj,atol=1e-14,rtol=1e-14)
yp=y+.37;gp=(p[:,None]+d-yp[:,None])**2-(p[:,None]-yp[:,None])**2;assert np.allclose(gp-direct,-2*d*.37,atol=1e-14,rtol=1e-14)
audit=json.loads((w.parent/'outputs/主任务反事实效用运行核验_202610050531Z.json').read_text(encoding='utf-8'));keys=['train_total','task_mse','utility_calibration','valid_mse','best_epoch','best_valid_mse','last_dev_batch_utility_weights'];histories=[r['history_prefix'] for r in audit['rows']]
for h in histories:assert [{k:e[k] for k in keys} for e in h[:10]]==[{k:e[k] for k in keys} for e in histories[0][:10]]
scripts=['audit_counterfactual_checks_v1.py','audit_counterfactual_snapshot_v1.py','audit_lease_captures_v1.py','run_lease_capture_pair_v1.py','capture_provisional_weight_v1.py','finalize_counterfactual_node_v1.py','save_metadata_pack_v1.py','verify_metadata_zip_v1.py']
for name in scripts:ast.parse((w/name).read_text(encoding='utf-8'))
report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'COUNTERFACTUAL_MATH_IDENTITIES_AND_COMMON_TEN_EPOCH_VALUES_VERIFIED','risk_identity_maxabs':float(np.max(np.abs(direct-identity))),'additive_prediction_risk_interaction_maxabs':float(np.max(np.abs(res-2*di*dj))),'label_dependency_identity_verified':True,'common_ten_epoch_train_dev_values_equal':True,'shared_phase_sha256':[r['shared_phase']['model_sha256'] for r in audit['rows']],'parsed_sources':{n:hashlib.sha256((w/n).read_bytes()).hexdigest() for n in scripts},'limits':'CPU synthetic identity check and saved actual TRAIN/dev records only; no new data/GPU/performance experiment or TEST.'}
p=w.parent/'outputs/主任务反事实效用数学与保存工具核验.json';assert not p.exists();p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report))
