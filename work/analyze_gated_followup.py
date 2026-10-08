from pathlib import Path
import datetime,hashlib,io,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'work/diagnose_gated_conditions_v2.py'
CAPTURE=ROOT/'work/capture_gated_followup_v1.py'
FOLDERS={'a':ROOT/'work/gated_v3_completed_20261005/a','b':Path('D:/CodexBackups/selective_flow_20261003_1105/gated_v3_completed_20261005/b'),'c':ROOT/'work/gated_v3_completed_20261005/c'}
AUDITS={'a':ROOT/'outputs/门控完整审核_A_20261005032825Z.json','b':ROOT/'outputs/门控完整审核_B_20261005032625Z.json','c':ROOT/'outputs/门控完整审核_C_20261005032825Z.json'}
def digest(b):return hashlib.sha256(b).hexdigest()
def metrics(p,y):
 n=y!=0
 return {'MAE':float(np.abs(p-y).mean()),'MSE':float(np.square(p-y).mean()),'author_batch_mse':float(np.mean([np.square(p[i:i+128]-y[i:i+128]).mean() for i in range(0,len(y),128)])),'Non0_acc2':float(((p[n]>=0)==(y[n]>=0)).mean()),'bias':float((p-y).mean()),'Corr':float(np.corrcoef(p,y)[0,1])}
arrays={};results={};audits={};receipts={}
for node,folder in FOLDERS.items():
 audit=json.loads(AUDITS[node].read_text(encoding='utf-8'));r=audit['rows'][0]
 assert r['full_checkpoint_verified'] and r['epochs']==100
 audits[node]=r
 directory=ROOT/'work/gated_followup_202610050335Z'/node
 proof=json.loads((directory/'proof.json').read_text())
 raw=(directory/'snapshot.zip').read_bytes()
 assert len(raw)==proof['archive_bytes'] and digest(raw)==proof['archive_sha256'] and proof['capture_source_sha256']==digest(CAPTURE.read_bytes())
 with zipfile.ZipFile(directory/'snapshot.zip') as z:
  members=json.loads(z.read('member_manifest.json'));assert members==proof['members']
  assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(members)|{'member_manifest.json'}
  for name,expected in members.items():
   b=z.read(name);assert len(b)==expected['bytes'] and digest(b)==expected['sha256']
  prefix='gated_diagnostics_20261005T0332Z/'
  d=json.loads(z.read(prefix+'run_'+r['mode']+'/results.json'))
  assert d['status']=='FROZEN_DEV_DIAGNOSTICS_COMPLETE' and d['test_accessed'] is False and d['optimizer_updates']==0
  assert d['checkpoint_sha256']==r['checkpoint_sha256'] and d['protocol_sha256']==r['selection']['protocol_sha256']
  assert d['source_sha256']==digest(SOURCE.read_bytes()) and d['model_state_before_sha256']==d['model_state_after_sha256']
  assert d['default_replay_maxabs']==d['off_replay_maxabs']==0
  log=z.read(prefix+'diagnostic.log').decode();assert 'Traceback' not in log and 'FROZEN_DEV_DIAGNOSTICS_COMPLETE' in log
  launch=json.loads(z.read(prefix+'launch.json'));resources=json.loads(z.read('actual_resources.json'))
  assert d['gpu_uuid'] in resources['gpu']['output'] and not resources['compute']['output'].strip()
  assert not any(line.split() and line.split()[0]==str(launch['pid']) for line in resources['processes']['output'].splitlines())
  b=z.read(prefix+'run_'+r['mode']+'/predictions.npz');assert digest(b)==d['output_sha256']
  with np.load(io.BytesIO(b),allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
  with np.load(folder/'run/predictions.npz',allow_pickle=False) as reference:
   assert np.array_equal(a['valid_y'],reference['valid_y']) and np.array_equal(a['pred_default'],reference['valid_pred']) and np.array_equal(a['pred_off'],reference['condition_off_pred'])
  assert a['valid_y'].shape==(229,)
  for name,v in d['conditions'].items():
   p=a['pred_'+name];assert p.shape==(229,) and np.isfinite(p).all()
   measured=metrics(p,a['valid_y']);assert set(measured)==set(v)
   assert all(abs(measured[k]-v[k])<1e-8 for k in measured)
   assert a['heads_'+name].shape==(229,3) and np.isfinite(a['heads_'+name]).all()
   assert np.array_equal(a['heads_'+name],a['heads_default'])
   assert a['context_norm_'+name].shape==(229,3) and np.isfinite(a['context_norm_'+name]).all()
  assert np.allclose(np.tanh(d['feedback_gates_raw']),d['feedback_gates_tanh'],atol=1e-8)
  arrays[node]=a;results[node]=d
  # Earlier failed path checks were pre-inference; preserve their actual traceback as history.
  failed=z.read('gated_diagnostics_20261005T0330Z/diagnostic.log').decode()
  assert 'FileNotFoundError' in failed and '/training.log' in failed
  m=json.loads((folder/'independent_manifest.json').read_text())
  target=json.loads((folder/'destination_verification.json').read_text())
  assert target['status']=='NEW_INFLOW_INDEPENDENT_LEASED_COPY_ALL_SEVEN_FILES_SHA_VERIFIED'
  assert target['manifest_sha256']==digest((folder/'independent_manifest.json').read_bytes())
  assert target['files']==m['files'] and target['selection']==r['selection']==m['selection']
  assert target['source_gpu_uuid']==m['source_gpu_uuid']!=target['target_gpu_uuid']
  assert target['local_audit_sha256']==m['local_audit_sha256']==digest(AUDITS[node].read_bytes())
  assert target['files']['best.pt']['sha256']==r['checkpoint_sha256'] and target['files']['best.pt']['bytes']==r['checkpoint_bytes']
  receipts[node]=target
base=arrays['a']['pred_default'].astype('float64');rows=[]
for node,a in arrays.items():
 y=a['valid_y'].astype('float64');on=a['pred_default'].astype('float64');off=a['pred_off'].astype('float64')
 delta=on-off;residual=off-y;change=(on-y)**2-residual**2;linear=2*residual*delta;quadratic=delta**2
 assert np.allclose(change,linear+quadratic,atol=1e-12,rtol=1e-12)
 assert np.allclose(on-base,delta+(off-base),atol=1e-12,rtol=1e-12)
 rows.append({'node':node,'mode':audits[node]['mode'],'selection':audits[node]['selection'],'metrics':audits[node]['metrics'],'mean_uniform_mse_change_on_vs_off':float(change.mean()),'linear_residual_alignment':float(linear.mean()),'quadratic_perturbation_cost':float(quadratic.mean()),'inference_condition_shift_rms':float(np.sqrt(quadratic.mean())),'inference_condition_shift_maxabs':float(np.abs(delta).max()),'trained_off_vs_matched_none_rms':float(np.sqrt(np.square(off-base).mean())),'squared_error_improved':int((change<-1e-10).sum()),'squared_error_worsened':int((change>1e-10).sum()),'diagnostics':results[node],'independent_preservation':receipts[node]})
report={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'GATED_100_COMPLETION_AND_INDEPENDENT_COPIES_AND_FROZEN_ARRAYS_ALL_VERIFIED','rows':rows,'limits':['Single exploratory seed91812 and mode-node binding; no TEST/model selection or stable five-seed claim.','Risk decomposition is an identity, not attribution of training-path differences.','Global scalar gates do not identify per-example shared/complementary/interfering evidence.','First three diagnostic attempts failed the log-path check before GPU inference; corrected v2 results independently checked.']}
p=ROOT/'outputs/门控实验完整核验与诊断.json'
with p.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({'status':report['status'],'rows':[{k:r[k] for k in ('mode','mean_uniform_mse_change_on_vs_off','linear_residual_alignment','quadratic_perturbation_cost','inference_condition_shift_rms','trained_off_vs_matched_none_rms')} for r in rows]},indent=2))
