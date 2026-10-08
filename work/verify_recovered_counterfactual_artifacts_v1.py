"""CPU recovery consistency only: does not certify live execution or replay weights."""
from pathlib import Path
import argparse,datetime,hashlib,io,json,sys,zipfile
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--node',choices=['a','b','c'],default='c');a.add_argument('--run',type=Path,required=True);a.add_argument('--out',type=Path,required=True);c=a.parse_args()
 root=Path(__file__).resolve().parents[1];dep='inflow_counterfactual_v5_deployment_20261005T0520Z';mode={'a':'none','b':'fixed','c':'predicted'}[c.node]
 mapping={'a':root/'work/counterfactual_live_202610050609Z/a','b':root/'work/counterfactual_live_202610050605Z/b','c':root/'work/lease_202610050610Z/c/new'}
 anchor=mapping[c.node];proof=json.loads((anchor/'proof.json').read_text(encoding='utf-8'));assert sha(anchor/'snapshot.zip')==proof['archive_sha256']
 with zipfile.ZipFile(anchor/'snapshot.zip') as z:
  prefix=dep+'/run_'+mode+'/';names=['protocol.json','batch_orders.npy','shared_phase.json','history.json'];raws={n:z.read(prefix+n) for n in names}
  for n,b in raws.items():assert len(b)==proof['members'][prefix+n]['bytes'] and hashlib.sha256(b).hexdigest()==proof['members'][prefix+n]['sha256']
  original_history=json.loads(raws['history.json']);protocol=json.loads(raws['protocol.json']);phase=json.loads(raws['shared_phase.json'])
 files=['best.pt','protocol.json','batch_orders.npy','history.json','selection.json','results.json','predictions.npz'];assert all((c.run/n).is_file() for n in files+['shared_phase.json'])
 for n in ['protocol.json','batch_orders.npy','shared_phase.json']:assert (c.run/n).read_bytes()==raws[n]
 assert protocol['name']=='inflow_counterfactual_v5' and protocol['mode']==mode and protocol['seed']==91814 and protocol['epochs']==100 and protocol['train_samples']==1281 and protocol['valid_samples']==229
 for name,digest in protocol['own_source_sha256'].items():assert sha(root/'work'/dep/name)==digest
 history=json.loads((c.run/'history.json').read_text(encoding='utf-8'));assert len(history)==100 and history[:len(original_history)]==original_history
 assert [r['epoch'] for r in history]==list(range(1,101))
 for j,r in enumerate(history):
  assert np.isfinite(r['valid_mse']);best=min(range(j+1),key=lambda i:history[i]['valid_mse']);assert r['best_epoch']==best+1 and r['best_valid_mse']==history[best]['valid_mse']
 best=min(range(100),key=lambda i:history[i]['valid_mse']);selection=json.loads((c.run/'selection.json').read_text(encoding='utf-8'));result=json.loads((c.run/'results.json').read_text(encoding='utf-8'))
 assert selection['epochs']==100 and selection['best_epoch']==best+1 and selection['valid_mse']==history[best]['valid_mse'] and selection['protocol_sha256']==sha(c.run/'protocol.json')
 assert selection['selected_effective_mode']==('fixed' if best<10 else mode) and result['mode']==mode and result['selection']==selection and result['model_state_unchanged'] is True and result['test_accessed'] is False
 cp=c.run/'best.pt';assert cp.stat().st_size>700_000_000 and sha(cp)==selection['checkpoint_sha256']
 with zipfile.ZipFile(cp) as z:assert len(z.namelist())>100 and z.testzip() is None
 orders=np.load(c.run/'batch_orders.npy',allow_pickle=False);assert orders.shape==(100,1280) and hashlib.sha256(orders.tobytes()).hexdigest()==protocol['batch_order_sha256']
 assert phase['epoch']==10 and phase['batch_order_sha256']==protocol['batch_order_sha256'] and phase['model_sha256']=='0f0da6e231ba350096f4a132416c15771b530fabb720afad4337410eaa2f89f3'
 with np.load(c.run/'predictions.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 expected={'valid_pred','valid_y','condition_off_pred','own','pair','utility','predicted_weights','weights','reference_prediction'};assert set(d)==expected and all(np.isfinite(v).all() for v in d.values())
 with np.load(root/'work/utility_completed_20261005/a/run/predictions.npz',allow_pickle=False) as z:assert np.array_equal(d['valid_y'],z['valid_y'])
 for k in ['valid_pred','valid_y','condition_off_pred','reference_prediction']:assert d[k].shape==(229,)
 for k in ['pair','utility','predicted_weights','weights']:assert d[k].shape==(229,6)
 assert d['own'].shape==(229,3) and np.max(np.abs(d['utility']))<=1 and np.allclose(d['predicted_weights'],1/(1+np.exp(-4*d['utility'])),atol=1e-7,rtol=1e-7)
 effective=selection['selected_effective_mode'];expected_weights=0 if effective=='none' else .5 if effective=='fixed' else d['predicted_weights'];assert np.allclose(d['weights'],expected_weights,atol=1e-7,rtol=1e-7)
 sys.path.insert(0,'C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen/outputs/monitoring_tools');from verify_careflow_five_seeds_v1 import metrics
 for name,pred in [('valid','valid_pred'),('frozen_condition_off','condition_off_pred')]:
  measured=metrics(d[pred],d['valid_y']);assert set(measured)==set(result[name]) and all(np.isclose(v,result[name][k],atol=1e-6) for k,v in measured.items())
 assert np.isclose(result['valid']['author_batch_mse'],selection['valid_mse'],atol=1e-6)
 if effective=='none':assert np.allclose(d['valid_pred'],d['condition_off_pred'],atol=1e-6)
 report={'status':'RECOVERED_ARTIFACT_SEVEN_FILES_CPU_CONSISTENCY_VERIFIED_NOT_EXECUTION_CERTIFICATE','verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':c.node,'mode':mode,'run':str(c.run.resolve()),'history_epochs':100,'history_anchor_epochs':len(original_history),'history_anchor_captured_at':proof['captured_at'],'history_anchor_archive_sha256':proof['archive_sha256'],'selection':selection,'files':{n:{'bytes':(c.run/n).stat().st_size,'sha256':sha(c.run/n)} for n in files},'shared_phase':{'bytes':(c.run/'shared_phase.json').stat().st_size,'sha256':sha(c.run/'shared_phase.json')},'checkpoint_zip_crc_verified':True,'actual_gpu_replay_verified':False,'original_run_process_completion_verified':False,'independent_target_copy_verified':False,'auditor_sha256':sha(Path(__file__)),'limits':'CPU artifact consistency binds recorded weight hash to selection and saved arrays, but does not replay PyTorch weights, verify current GPU UUID or establish original completion timing. No TEST access, model import or training.'}
 with c.out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({'status':report['status'],'node':c.node,'history_anchor_epochs':len(original_history)}))
if __name__=='__main__':main()
