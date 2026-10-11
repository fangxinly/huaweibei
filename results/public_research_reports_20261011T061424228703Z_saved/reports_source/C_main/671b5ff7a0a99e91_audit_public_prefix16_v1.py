"""B downloads new completed original once, verifies all bytes and complete optimizer state."""
import argparse,datetime,hashlib,json,os,subprocess,sys,urllib.request,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--publication',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else digest(p)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def blockhash(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
r=json.loads(a.publication.read_text());cap=json.loads(a.capture.read_text());assert r['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED' and cap['natural_exit']==0 and cap['ZIP_CRC_unique_all_members']
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-609b23d6-282d-8a2b-23f5-9433b824d212'
a.out.mkdir(exist_ok=False);parts=sorted([x for x in r['assets'] if x['source_sha256']==cap['archive_SHA']],key=lambda x:x['source_offset']);assert parts
archive=a.out/'A_complete_original.zip'
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
from fixed_flow_components_candidate import tensor_sha
from common_budget_selection_candidate import train_batches
cp=original/'out/complete_resume_step16.pt';actual=json.loads((original/'out/precheck_result.json').read_text());assert digest(cp)==actual['checkpoint_SHA']
torch.set_num_threads(2);state=torch.load(cp,map_location='cpu');m=state['metadata'];assert m==actual['metadata'] and m['steps']==16 and m['next_epoch']==0 and m['next_batch']==16 and not m['validation_labels_used'] and m['TEST_entry_not_indexed']
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
probe=np.load(original/'out/fixed_TRAIN_probe.npz',allow_pickle=False);assert probe['labels_not_used'].item() and probe['ids'].tolist()==[m['official_train_dev_IDs']['train'][i] for i in batches[0]]
assert not list((original/'out').glob('DEV*'))
result=dict(status='B_ORIGINAL_PREFIX16_COMPLETE_MODEL_ADAM_RNG_ORDER_SHA_CRC_PASSED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,UUID=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),source_SHA=digest(__file__),A_archive_SHA=cap['archive_SHA'],checkpoint_SHA=digest(cp),model_state_SHA=m['final_state_SHA'],Adam_parameter_states=len(info),all_Adam_steps=16,scheduler_last_epoch=16,all_source_and_member_SHA=True,ZIP_CRC_unique=True,full_original_retained=str(original),CPU_encoder_forward=False,VAL_labels_used=False,TEST_entry_not_indexed=True,processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True))
(a.out/'audit_result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='processes'}),flush=True)
