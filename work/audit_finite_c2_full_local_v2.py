from pathlib import Path
import json,hashlib,zipfile,tarfile,datetime
import numpy as np
w=Path(__file__).parent;root=w.parent;base=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_c2_completed_20261005');d=base/'c';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
archive=base/'c2_full_preservation.tar';ar=json.loads((base/'c2_full_archive_receipt.json').read_text());assert sha(archive)==ar['sha256'] and archive.stat().st_size==ar['bytes']
if not d.exists():
 d.mkdir()
 with tarfile.open(archive) as t:
  for member in t.getmembers():assert member.isfile() and Path(member.name).name==member.name
  t.extractall(d,filter='data')
m=json.loads((d/'preservation_manifest.json').read_text());assert m==ar['manifest']
for group in ['required_seven_files','extra_files']:
 for n,x in m[group].items():assert sha(d/n)==x['sha256'] and (d/n).stat().st_size==x['bytes']
assert len(m['required_seven_files'])==7 and len(m['extra_files'])==12
with zipfile.ZipFile(d/'full_checkpoint.pt') as z:assert z.testzip() is None
r=json.loads((d/'full_checkpoint_receipt.json').read_text());h=json.loads((d/'history.json').read_text());s=json.loads((d/'selection.json').read_text());pr=json.loads((d/'protocol.json').read_text());plan=json.loads((d/'finite_formal_plan_v2.json').read_text())
assert len(h)==s['epochs']==100 and s['best_epoch']==r['best_epoch']==np.argmin([x['dev_author_batch_mean_mse'] for x in h])+1
assert r['assembly_node']==m['assembly_node']=='a' and r['training_node']==m['training_node']=='c' and r['full_state_tensor_count']==461 and not r['test_requested']
assert r['full_checkpoint_sha256']==sha(d/'full_checkpoint.pt') and r['full_checkpoint_bytes']==(d/'full_checkpoint.pt').stat().st_size
assert r['source_sha256']==sha(d/'assemble_finite_full_checkpoint_v2.py')==sha(w/'assemble_finite_full_checkpoint_v2.py')
assert r['formal_plan_sha256']==pr['formal_plan_sha256']==sha(d/'finite_formal_plan_v2.json')==sha(w/'finite_formal_plan_v2.json')
for n,v in plan['source_sha256'].items():assert sha(d/n)==v==pr['source_sha256'][n]==sha(w/n)
assert sha(d/'best_addon.pt')==s['addon_sha256'] and sha(d/'predictions.npz')==s['prediction_sha256'] and sha(d/'shared_phase_addon.pt')==s['shared_phase_sha256']=='32fad83f206b32f4070fe2bbf0b3a588c1e8a5340c58b337a8e7052161c08088'
with np.load(d/'predictions.npz') as a,np.load(d/'full_replay_predictions.npz') as b:
 assert a['valid_y'].shape==a['valid_pred'].shape==b['valid_pred'].shape==(229,) and np.array_equal(a['valid_y'],b['valid_y']) and all(np.isfinite(x).all() for x in [a['valid_pred'],a['valid_y'],b['valid_pred']])
 error=float(np.max(np.abs(a['valid_pred']-b['valid_pred'])));assert error==r['full_raw_vs_cached_prediction_max_error']<2e-5
 assert abs(float(np.mean((b['valid_pred']-b['valid_y'])**2))-r['metrics']['MSE229'])<1e-7
out=dict(status='C2_ACTUAL100_FULL_LOCAL_SHA_ZIP_SOURCE_AND_229_REPLAY_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipt=r,manifest=m,local_directory=str(d),archive_receipt=ar,scope='OriginalC1failed remains failed; C2 new protocol. Strict Torch full reload executed on newA; this independent local NumPy process audits raw receipts and arrays. CPU independent target receipt still separate.')
(root/'outputs/有限任务风险C2完整权重本地独立核验.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(r))
