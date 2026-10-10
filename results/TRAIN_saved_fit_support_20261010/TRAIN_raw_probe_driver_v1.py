"""TRAIN-only raw diagnostic driver; no official VAL/TEST scoring entry point."""
import argparse,datetime,hashlib,io,json,os,pathlib,pickle,pickletools,subprocess,sys,shutil,tempfile,zipfile
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import TRAIN_raw_conditional_ridge_core_v1 as core
P=pathlib.Path
DATA_SHA='5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b'
class OpaqueDtype:
 def __init__(self,code,*args):self.code=code;self.endian='='
 def __setstate__(self,state):self.endian=state[1]
class OpaqueArray:
 def __init__(self,*args):pass
 def __setstate__(self,state):
  assert isinstance(state,tuple) and len(state)==5
  self.shape=tuple(state[1]);self.dtype=state[2];self.fortran=state[3];self.payload=state[4]
def reconstruct(*args):return OpaqueArray()
class MetadataOnly(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={('numpy.core.multiarray','_reconstruct'):reconstruct,('numpy','ndarray'):OpaqueArray,('numpy','dtype'):OpaqueDtype}
  if (module,name) not in allowed:raise pickle.UnpicklingError('Unapproved global')
  return allowed[(module,name)]
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def write(p,v):p.write_bytes(raw(v))
def read(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_sha(p,expected):assert sha(p)==expected
def claim_once(p,receipt):
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw(receipt))
def identity_sha(ids):return hashlib.sha256(raw(ids)).hexdigest()
def decode(array,role,ledger):
 assert role=='train' and isinstance(array,OpaqueArray) and isinstance(array.payload,bytes)
 dtype=np.dtype(array.dtype.code).newbyteorder(array.dtype.endian)
 assert dtype.kind in 'fiu' and dtype.itemsize in (1,2,4,8)
 assert len(array.payload)==int(np.prod(array.shape))*dtype.itemsize
 ledger['decoded_arrays']+=1;ledger['decoded_roles'].add(role)
 return np.frombuffer(array.payload,dtype=dtype).reshape(array.shape,order='F' if array.fortran else 'C').astype(np.float64)
def features_from_train(ds,manifest):
 # Only this explicit TRAIN lookup is legal. Other split arrays remain opaque.
 train=ds['train'];assert len(train)==len(manifest)
 features={'T':[],'A':[],'V':[]};labels=[];ledger=dict(decoded_arrays=0,decoded_roles=set(),decoded_TRAIN_labels=0)
 for record,row in zip(train,manifest):
  assert row['split']=='train' and isinstance(record,tuple) and len(record)==3 and record[2]==row['row_id']
  words,visual,audio=record[0];assert isinstance(words,list) and words
  # Check all row identities before decoding this row's numerical buffers.
  assert visual.shape==(len(words),47) and audio.shape==(len(words),74)
  v=decode(visual,'train',ledger);a=decode(audio,'train',ledger);y=decode(record[1],'train',ledger)
  assert y.size==1 and np.isfinite(y).all();ledger['decoded_TRAIN_labels']+=1
  features['T'].append(core.text_features(words));features['A'].append(core.numeric_features(a,74));features['V'].append(core.numeric_features(v,47));labels.append(float(y.reshape(-1)[0]))
 ledger['decoded_roles']=sorted(ledger['decoded_roles']);assert ledger['decoded_arrays']==3*len(manifest) and ledger['decoded_roles']==['train']
 return {k:np.stack(v) for k,v in features.items()},np.array(labels),ledger
def validate_manifest(rows):
 assert len({r['row_id'] for r in rows})==len(rows) and [r['row_index'] for r in rows]==list(range(len(rows)))
 videos={}
 for r in rows:
  assert r['split']=='train' and type(r['held_out_fold']) is int and 0<=r['held_out_fold']<5
  if r['video_id'] in videos:assert videos[r['video_id']]==r['held_out_fold']
  videos[r['video_id']]=r['held_out_fold']
 assert set(videos.values())==set(range(5))
def make_opaque(x):
 x=np.asarray(x,dtype=np.float64);a=OpaqueArray();d=OpaqueDtype('f8');d.endian='<'
 a.__setstate__((1,x.shape,d,False,x.astype('<f8').tobytes()));return a
def qualify():
 class TrainOnly(dict):
  def __getitem__(self,k):
   assert k=='train','Forbidden split indexed';return super().__getitem__(k)
 rows=[dict(row_index=i,row_id='synthetic'+str(i),video_id='video'+str(i),split='train',held_out_fold=i%5) for i in range(10)]
 records=[((['a','b'],make_opaque(np.ones((2,47))),make_opaque(np.ones((2,74)))),make_opaque([i*.1]),r['row_id']) for i,r in enumerate(rows)]
 ds=TrainOnly(train=records);x,y,ledger=features_from_train(ds,rows);assert ledger['decoded_arrays']==30 and x['T'].shape==(10,257) and x['A'].shape==(10,149) and x['V'].shape==(10,95)
 validate_manifest(rows)
 refused={}
 for name,action in [('dev_numeric_decode',lambda:decode(records[0][1],'dev',dict(decoded_arrays=0,decoded_roles=set()))),('wrong_ID',lambda:features_from_train(ds,[dict(rows[0],row_id='wrong')]+rows[1:])),('wrong_split',lambda:features_from_train(ds,[dict(rows[0],split='test')]+rows[1:])),('duplicate_row',lambda:validate_manifest([dict(rows[0],row_id=rows[1]['row_id'])]+rows[1:]))]:
  try:action()
  except AssertionError:refused[name]=True
  else:raise AssertionError(name+' accepted')
 with tempfile.TemporaryDirectory() as td:
  path=P(td)/'token.json';claim_once(path,{'synthetic':True})
  try:claim_once(path,{'synthetic':True})
  except FileExistsError:refused['once_token_reuse']=True
  else:raise AssertionError('Once token reused')
  try:verify_sha(path,'0'*64)
  except AssertionError:refused['wrong_source_SHA']=True
  else:raise AssertionError('Wrong SHA accepted')
 return dict(status='SYNTHETIC_TRAIN_DECODER_AND_DRIVER_QUALIFIED',numpy=np.__version__,no_other_split_indexed=True,numeric_decoder_role_rejections=refused,TRAIN_decode_ledger=ledger,core=core.qualification(),real_data_loaded=False,real_probe_executed=False)
def archive(root,name):
 target=root/name;members=[]
 with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or p==target:continue
   b=p.read_bytes();n=p.relative_to(root).as_posix();z.writestr(n,b);members.append(dict(name=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
  z.writestr('member_manifest.json',raw(members))
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())=={v['name'] for v in members}|{'member_manifest.json'}
  for v in members:
   b=z.read(v['name']);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 return dict(path=str(target),sha256=sha(target),all_member_SHA_CRC_unique_exact_set_passed=True,members=len(members))
def run(plan_path,plan_sha):
 assert sha(plan_path)==plan_sha;plan=read(plan_path);protocol=P(plan['protocol']);fold=P(plan['fold_manifest']);data=P(plan['data']);root=P(plan['root'])
 assert not root.exists() and not P(plan['once_token']).exists()
 assert sha(protocol)==plan['protocol_SHA'] and sha(fold)==plan['fold_SHA'] and sha(data)==DATA_SHA
 for name,h in plan['source_SHA'].items():assert sha(P(__file__).parent/name)==h
 assert np.__version__=='1.26.4' and str(P(sys.executable).resolve())==str(P(plan['interpreter']).resolve())
 now=datetime.datetime.now(datetime.timezone.utc);deadline=datetime.datetime.fromisoformat(plan['human_conservative_deadline']);assert (deadline-now).total_seconds()>3*3600
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==plan['UUID']
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not compute
 mem=dict((k,int(v.split()[0])*1024) for k,v in (line.split(':',1) for line in P('/proc/meminfo').read_text().splitlines()));assert mem['MemAvailable']>=6*1024**3
 assert shutil.disk_usage(root.parent).free>256*1024**2
 rows=read(fold);validate_manifest(rows);assert len(rows)==1281 and len({r['video_id'] for r in rows})==52
 payload=data.read_bytes();assert not any('FLOAT' in op.name for op,arg,pos in pickletools.genops(payload))
 ds=MetadataOnly(io.BytesIO(payload)).load();features,labels,ledger=features_from_train(ds,rows);del ds,payload
 root.mkdir();write(root/'fresh_preflight.json',dict(actual_UTC=now.isoformat(),identity=plan['execution_identity'],UUID=uuid,compute=compute,RAM_available=mem['MemAvailable'],space=shutil.disk_usage(root.parent)._asdict(),runtime=dict(python=sys.version,numpy=np.__version__,interpreter=sys.executable),fullargv=sys.argv,processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True),source_SHA=plan['source_SHA'],data_SHA=DATA_SHA,protocol_SHA=sha(protocol),fold_SHA=sha(fold),remaining_seconds=(deadline-now).total_seconds(),no_old_node_contract_reused=True))
 assert (datetime.datetime.now(datetime.timezone.utc)-now).total_seconds()<300
 claim_once(plan['once_token'],dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_SHA=plan_sha,root=str(root),identity=plan['execution_identity']))
 paths={'T':features['T'],'T+A':np.concatenate([features['T'],features['A']],1),'T+V':np.concatenate([features['T'],features['V']],1),'T+A+V':np.concatenate([features['T'],features['A'],features['V']],1)}
 for name,x in paths.items():assert x.shape==(1281,read(protocol)['dimensions'][name])
 assignments=np.array([r['held_out_fold'] for r in rows]);pred={name:np.full(1281,np.nan) for name in paths};states={};fits=[]
 for f in range(5):
  fit=np.flatnonzero(assignments!=f);held=np.flatnonzero(assignments==f);assert not {rows[i]['video_id'] for i in fit}&{rows[i]['video_id'] for i in held}
  for name,x in paths.items():
   prediction,params=core.fit_predict(x[fit],labels[fit],x[held]);pred[name][held]=prediction
   key='fold'+str(f)+'_'+name.replace('+','_')
   for k in ['mean','scale','active','coefficient']:states[key+'_'+k]=params[k]
   fits.append(dict(fold=f,path=name,fit_rows=len(fit),held_rows=len(held),fit_ID_SHA=identity_sha([rows[i]['row_id'] for i in fit]),held_ID_SHA=identity_sha([rows[i]['row_id'] for i in held]),intercept=params['intercept'],lambda_mean_loss=1.))
  print(json.dumps(dict(phase='raw_TRAIN_out_of_video_prediction',fold=f,held_rows=len(held))),flush=True)
 assert all(np.isfinite(v).all() for v in pred.values())
 np.savez_compressed(root/'TRAIN_fixed_fivefold_prediction.npz',row_id=np.array([r['row_id'] for r in rows]),video_id=np.array([r['video_id'] for r in rows]),fold=assignments,label=labels,**{'prediction_'+k.replace('+','_'):v for k,v in pred.items()})
 np.savez_compressed(root/'fit_fold_states.npz',**states);write(root/'fit_identity_records.json',fits);write(root/'TRAIN_decoder_ledger.json',ledger)
 for p in [protocol,fold,plan_path,P(__file__),P(core.__file__)]:shutil.copyfile(p,root/p.name)
 write(root/'prediction_result.json',dict(status='RAW_TRAIN_FIXED_FIVEFOLD_PREDICTIONS_COMPLETE_METRICS_NOT_YET_COMPUTED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),prediction_SHA=sha(root/'TRAIN_fixed_fivefold_prediction.npz'),states_SHA=sha(root/'fit_fold_states.npz'),protocol_SHA=sha(protocol),TRAIN_rows=1281,video_groups=52,decoder_ledger=ledger,VAL_TEST_numeric_decode=False,official_AB_tokens_untouched=True,no_AB_model_forward=True,no_performance_or_mechanism_claim=True))
 saved=archive(root,'complete_raw_TRAIN_prediction_original.zip');write(root/'capture_receipt.json',saved);print(json.dumps(saved),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--qualify',action='store_true');p.add_argument('--plan',type=P);p.add_argument('--plan-sha');a=p.parse_args()
 if a.qualify:assert a.plan is None;print(json.dumps(qualify(),allow_nan=False))
 else:assert a.plan and a.plan_sha;run(a.plan,a.plan_sha)
