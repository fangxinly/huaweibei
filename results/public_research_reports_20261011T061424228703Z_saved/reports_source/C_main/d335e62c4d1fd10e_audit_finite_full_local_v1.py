from pathlib import Path
import datetime,hashlib,json,zipfile
import numpy as np
root=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_risk_completed_20261005');rows={};sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for n in ['a','b']:
 d=root/n;m=json.loads((d/'preservation_manifest.json').read_text());r=json.loads((d/'full_checkpoint_receipt.json').read_text());h=json.loads((d/'history.json').read_text());s=json.loads((d/'selection.json').read_text());pr=json.loads((d/'protocol.json').read_text())
 for group in ['required_seven_files','extra_files']:
  for name,x in m[group].items():assert sha(d/name)==x['sha256'] and (d/name).stat().st_size==x['bytes']
 with zipfile.ZipFile(d/'full_checkpoint.pt') as z:assert z.testzip() is None
 assert len(h)==100 and s['best_epoch']==r['best_epoch']==np.argmin([x['dev_author_batch_mean_mse'] for x in h])+1
 assert r['full_checkpoint_bytes']==746207712 and r['full_state_tensor_count']==461 and r['assembly_node']=='a' and r['training_node']==n
 assert r['full_checkpoint_sha256']==sha(d/'full_checkpoint.pt') and r['source_sha256']==sha('work/assemble_finite_full_checkpoint_v1.py')
 assert r['formal_plan_sha256']==sha('work/finite_formal_plan_v1.json')==pr['formal_plan_sha256']
 with np.load(d/'predictions.npz') as a,np.load(d/'full_replay_predictions.npz') as b:
  assert a['valid_y'].shape==a['valid_pred'].shape==b['valid_pred'].shape==(229,) and np.array_equal(a['valid_y'],b['valid_y'])
  err=float(np.max(np.abs(a['valid_pred']-b['valid_pred'])));assert err==r['full_raw_vs_cached_prediction_max_error']<2e-5
  assert abs(float(np.mean(np.abs(b['valid_pred']-b['valid_y'])))-r['metrics']['MAE'])<1e-7
 rows[n]=dict(full_checkpoint_receipt=r,manifest=m,local_path=str(d),local_sha_and_zip_crc_pass=True)
out=dict(status='FINITE_AB_TWO_COMPLETE_FULL_MODELS_LOCAL_SHA_ZIP_SOURCE_AND_DEV229_ARRAYS_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,scope='Only A/B completed. Original C v1 failed after10 complete epochs. Strict full Torch reload executed on assembly newA; this local NumPy process independently checks original receipts, file SHA, ZIP CRC,100minimum selection and replay arrays. No TEST.')
Path('outputs/有限任务风险两完整权重本地核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({n:x['full_checkpoint_receipt']['metrics'] for n,x in rows.items()}))
