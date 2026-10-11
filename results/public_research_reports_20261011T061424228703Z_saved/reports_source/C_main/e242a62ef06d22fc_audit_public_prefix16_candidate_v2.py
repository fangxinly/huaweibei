"""Verify a completed peer original and replay actual CandidateTail on CPU."""
import argparse,datetime,hashlib,json,os,subprocess,sys,urllib.request,zipfile
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def blockhash(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def run(a):
 r=json.loads(a.publication.read_text());cap=json.loads(a.capture.read_text());assert r['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED' and cap['natural_exit']==0 and cap['ZIP_CRC_unique_all_members']
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==a.expected_uuid
 a.out.mkdir(exist_ok=False);parts=sorted([x for x in r['assets'] if x['source_sha256']==cap['archive_SHA']],key=lambda x:x['source_offset']);assert parts
 archive=a.out/'peer_complete_original.zip'
 if a.received_archive:
  assert a.received_archive.stat().st_size==cap['archive_bytes'] and digest(a.received_archive)==cap['archive_SHA'];os.link(a.received_archive,archive)
 else:
  with archive.open('xb') as target:
   for row in parts:
    assert target.tell()==row['source_offset'];h=hashlib.sha256();n=0
    with urllib.request.urlopen(row['url'],timeout=180) as resp:
     for b in iter(lambda:resp.read(8*1024**2),b''):target.write(b);h.update(b);n+=len(b)
    assert n==row['bytes'] and h.hexdigest()==row['sha256']
 assert archive.stat().st_size==cap['archive_bytes'] and digest(archive)==cap['archive_SHA']
 original=a.out/'original';original.mkdir()
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for name in z.namelist():assert (original/name).resolve().is_relative_to(original.resolve())
  members=json.loads(z.read('member_SHA.json'))
  for name,h in members.items():
   with z.open(name) as f:assert blockhash(f)==h
  z.extractall(original)
 sys.path.insert(0,str(original/'original_source'))
 import numpy as np
 import torch
 from torch import nn
 from types import SimpleNamespace
 from candidate_adapter import make_candidate,CandidateTail
 from fixed_flow_components_candidate import tensor_sha
 from common_budget_selection_candidate import train_batches
 sourceplan=json.loads((original/'original_source/qualified_precheck_plan.json').read_text());peer=sourceplan['execution_node'];assert sourceplan['GPU_UUID'][peer]!=uuid and sourceplan['candidate_mode']==a.mode
 cp=original/'out/complete_resume_step16.pt';actual=json.loads((original/'out/precheck_result.json').read_text());assert digest(cp)==actual['checkpoint_SHA']
 torch.set_num_threads(2);state=torch.load(cp,map_location='cpu');m=state['metadata'];assert m==actual['metadata'] and m['candidate_mode']==a.mode and m['steps']==16 and m['next_epoch']==0 and m['next_batch']==16 and not m['validation_labels_used'] and m['TEST_entry_not_indexed']
 assert tensor_sha(state['model'])==m['final_state_SHA'];info=m['parameter_info'];mapping=m['optimizer_index_to_name'];opt=state['optimizer'];assert len(opt['state'])==len(info)==len(mapping)
 for i,q in opt['state'].items():
  name=mapping[str(i)];assert q['step'].item()==16
  for k in ('exp_avg','exp_avg_sq'):assert list(q[k].shape)==info[name]['shape'] and torch.isfinite(q[k]).all()
  assert (q['exp_avg_sq']>=0).all()
 assert state['scheduler']['last_epoch']==16 and len(state['rng']['cuda'])==1 and state['rng']['torch'].dtype==torch.uint8
 order=np.load(original/'out/original_common_orders.npy',allow_pickle=False);assert digest(original/'out/original_common_orders.npy')==m['orders_SHA'];batches,omitted=train_batches(list(map(int,order[0])))
 assert state['omitted_rows']==list(omitted)
 records=[json.loads(line) for line in (original/'out/TRAIN_health.jsonl').read_text().splitlines()];assert records==state['prefix_records'] and len(records)==16
 for i,row in enumerate(records):assert row['step']==i+1 and row['rows']==list(batches[i])
 assert not list((original/'out').glob('DEV*'))
 core=SimpleNamespace(own_flow=make_candidate(a.mode),fusion=nn.Sequential(nn.Linear(300,150),nn.ReLU(),nn.Linear(150,100)),predictor=nn.Sequential(nn.Linear(100,150),nn.ReLU(),nn.Linear(150,1)))
 tail=CandidateTail(core);export=torch.load(original/'out/prefix16_tail.pt',map_location='cpu');assert export['mode']==a.mode;tail.load_state_dict(export['state'],strict=True)
 for n,v in tail.state_dict().items():
  key='dberta.own_flow.'+n[len('flow.'):] if n.startswith('flow.') else 'dberta.'+n;assert torch.equal(v,state['model'][key]),n
 cache=dict(np.load(original/'out/prefix16_source.npz',allow_pickle=False));before=tensor_sha(tail.state_dict());rng=torch.get_rng_state().clone()
 with torch.no_grad():on,off=tail(torch.from_numpy(cache['source']),torch.from_numpy(cache['mask']))
 err=max(float(np.max(abs(on.numpy()-cache['on']))),float(np.max(abs(off.numpy()-cache['off']))));assert err<1e-4
 assert tensor_sha(tail.state_dict())==before and torch.equal(rng,torch.get_rng_state())
 result=dict(status='CANDIDATE_OTHER_NODE_PREFIX16_COMPLETE_MODEL_ADAM_RNG_ORDER_CPU_ON_OFF_PASSED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,UUID=uuid,source_SHA=digest(__file__),peer=peer,mode=a.mode,archive_SHA=cap['archive_SHA'],checkpoint_SHA=digest(cp),model_state_SHA=m['final_state_SHA'],Adam_parameter_states=len(info),all_Adam_steps=16,scheduler_last_epoch=16,all_source_and_member_SHA=True,ZIP_CRC_unique=True,full_original_retained=str(original),CPU_actual_same_flow_ON_OFF_maxerror=err,CPU_encoder_forward=False,parameters_buffers_RNG_unchanged=True,VAL_labels_used=False,TEST_entry_not_indexed=True,original_transport='explicit_received_original' if a.received_archive else 'public_Release_download')
 (a.out/'audit_result.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('publication','capture','out'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--mode',choices=['factorized_aux','regression_aux'],required=True);p.add_argument('--expected-uuid',required=True);p.add_argument('--received-archive',type=Path);run(p.parse_args())
