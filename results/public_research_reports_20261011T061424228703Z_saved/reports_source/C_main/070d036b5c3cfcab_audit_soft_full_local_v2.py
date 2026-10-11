from pathlib import Path
import datetime,hashlib,json,zipfile,numpy as np
r=Path(__file__).resolve().parents[1];d=Path('D:/CodexBackups/selective_flow_20261003_1105/jacobian_completed_20261005');rows={};sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for node in 'abc':
 folder=d/node;x=json.loads((folder/'full_checkpoint_receipt.json').read_text());sel=json.loads((folder/'selection.json').read_text());history=json.loads((folder/'history.json').read_text())
 assert x['status']=='FULL_CHECKPOINT_STRICT_DISK_RELOAD_AND_OFFICIAL_DEV229_REPLAY_COMPLETE' and x['node']==node
 assert x['full_checkpoint_bytes']==(folder/'full_checkpoint.pt').stat().st_size and x['full_checkpoint_sha256']==sha(folder/'full_checkpoint.pt')
 assert x['best_epoch']==sel['best_epoch'] and x['addon_sha256']==sha(folder/'best_addon.pt')==sel['addon_sha256']
 assert x['full_replay_predictions_sha256']==sha(folder/'full_replay_predictions.npz') and x['full_raw_vs_cached_prediction_max_error']<2e-5 and not x['test_requested']
 with zipfile.ZipFile(folder/'full_checkpoint.pt') as z:assert z.testzip() is None
 a=np.load(folder/'full_replay_predictions.npz');b=np.load(folder/'predictions.npz');assert np.array_equal(a['valid_y'],b['valid_y']);error=float(np.max(np.abs(a['valid_pred']-b['valid_pred'])));assert abs(error-x['full_raw_vs_cached_prediction_max_error'])<1e-12
 assert abs(float(np.mean(np.abs(a['valid_pred']-a['valid_y'])))-x['metrics']['MAE'])<1e-6
 names=['full_checkpoint.pt','full_checkpoint_receipt.json','protocol.json','history.json','selection.json','predictions.npz','orders.npy']
 files={n:{'sha256':sha(folder/n),'bytes':(folder/n).stat().st_size} for n in names}
 extra={n:{'sha256':sha(folder/n),'bytes':(folder/n).stat().st_size} for n in ['best_addon.pt','shared_phase_addon.pt','diagnostics.npz','diagnostics_receipt.json','full_replay_predictions.npz']}
 manifest={'node':node,'required_seven_files':files,'extra_files':extra,'assembly_node':'newA','training_node':node,'full_checkpoint_receipt':x}
 (folder/'preservation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');rows[node]=manifest
out={'status':'THREE_FULL_WEIGHTS_LOCAL_SHA_ZIPCRC_AND_229_REPLAY_AUDITED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'directory':str(d),'rows':rows,'independent_cpu_preservation_pending':True}
(r/'outputs/Jacobian三完整权重本地核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({n:x['full_checkpoint_receipt']['metrics'] for n,x in rows.items()}))
