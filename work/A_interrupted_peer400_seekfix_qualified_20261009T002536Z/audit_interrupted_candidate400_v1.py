"""Download preserved interrupted original, verify all bytes and complete saved400 on peer CPU."""
import argparse,datetime,hashlib,json,os,subprocess,sys,urllib.request,zipfile
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('publication','capture','out'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--retained-archive',type=Path,required=True);p.add_argument('--uuid',required=True);p.add_argument('--peer',choices=['A','B'],required=True);a=p.parse_args()
def digest(f):
 h=hashlib.sha256()
 with f.open('rb') as z:
  for b in iter(lambda:z.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
cap=json.loads(a.capture.read_text());pub=json.loads(a.publication.read_text());assert cap['capture_process_natural_exit']==0 and cap['prior_training_natural_exit_unknown'] and not cap['prior_complete100'] and pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED'
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==a.uuid
a.out.mkdir();archive=a.retained_archive
assert archive.resolve().is_relative_to(Path('/data/coding'))
assert digest(archive)==cap['archive_SHA'] and archive.stat().st_size==cap['archive_bytes']
import torch,numpy as np
torch.set_num_threads(2)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in json.loads(z.read('member_SHA.json')).items():
  q=hashlib.sha256()
  with z.open(n) as f:
   for b in iter(lambda:f.read(8*1024**2),b''):q.update(b)
  assert q.hexdigest()==h
 # Preserve the complete archive without a second full extraction on constrained disk.
 source=a.out/'original_source';source.mkdir()
 for n in z.namelist():
  if n.startswith('original_source/'):
   target=source/n[len('original_source/'):];assert target.resolve().is_relative_to(source.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
 sys.path.insert(0,str(source));from fixed_flow_components_candidate import tensor_sha
 temporary=a.out/'_temporary_saved400.pt'
 with z.open('prior/out/complete_resume_step400.pt') as inp,temporary.open('xb') as out:
  for block in iter(lambda:inp.read(8*1024**2),b''):out.write(block)
 assert temporary.stat().st_size==cap['warm400_checkpoint_bytes'] and digest(temporary)==cap['warm400_checkpoint_SHA']
 state=torch.load(temporary,map_location='cpu')
 m=state['metadata'];assert m['steps']==400 and m['next_epoch']==10 and m['next_batch']==0 and tensor_sha(state['model'])==m['final_state_SHA']
 assert state['scheduler']['last_epoch']==400 and len(state['rng']['cuda'])==1 and state['rng']['torch'].dtype==torch.uint8
 mapping=m['optimizer_index_to_name'];info=m['parameter_info'];assert len(state['optimizer']['state'])==len(info)==len(mapping)==345
 for i,s in state['optimizer']['state'].items():
  assert s['step'].item()==400;n=mapping[str(i)]
  for k in ('exp_avg','exp_avg_sq'):assert list(s[k].shape)==info[n]['shape'] and torch.isfinite(s[k]).all()
  assert (s['exp_avg_sq']>=0).all()
 history=state['history'];assert len(history)==10 and all(r['epoch']==i+1 and r['steps']==(i+1)*40 for i,r in enumerate(history))
 best=min(range(10),key=lambda i:history[i]['DEV_batch_MSE']);assert m['best_epoch']==best+1 and m['best_MSE']==history[best]['DEV_batch_MSE'] and tensor_sha(state['selected_model'])==history[best]['state_SHA']
 for r in history:assert hashlib.sha256(z.read('prior/out/DEV_epoch_%03d_prediction_only.npz'%r['epoch'])).hexdigest()==r['prediction_SHA']
 assert hashlib.sha256(z.read('prior/out/original_common_orders.npy')).hexdigest()==m['orders_SHA']
 from common_budget_selection_candidate import train_batches
 import io
 orders=np.load(io.BytesIO(z.read('prior/out/original_common_orders.npy')),allow_pickle=False)
 for i in range(10):assert history[i]['dropped_rows']==list(train_batches(list(map(int,orders[i])))[1])
r=dict(status='INTERRUPTED_ORIGINAL_PEER_COMPLETE_400_MODEL_ADAM_SCHED_RNG_ORDER_SELECTION_CPU_PASSED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,UUID=uuid,peer=a.peer,archive_SHA=cap['archive_SHA'],archive_bytes=cap['archive_bytes'],warm400_checkpoint_SHA=cap['warm400_checkpoint_SHA'],model_state_SHA=m['final_state_SHA'],mode=m['candidate_mode'],all_Adam_steps=400,Adam_parameter_states=345,source_SHA=digest(Path(__file__)),complete_original_retained=str(archive),prior_training_natural_exit_unknown=True,CPU_encoder_forward=False,CPU_flow_forward=False,no_new_VAL_TEST_scoring=True,public_Release_download=False,retained_public_Release_origin_archive_reuse=True,temporary_PT_bytes_SHA_verified=True,all_member_SHA_CRC_unique=True)
# This task-owned temporary extraction is explicitly disposable; immutable original ZIP retained.
assert temporary.resolve().parent==a.out.resolve() and temporary.name=='_temporary_saved400.pt' and digest(archive)==cap['archive_SHA']
temporary.unlink();r['temporary_PT_removed_after_success']=True
(a.out/'audit_result.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r),flush=True)
