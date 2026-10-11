"""Separate TRAIN-only support description. Never solve, fit, read labels or score."""
import argparse,ast,datetime,hashlib,io,json,os,pathlib,pickletools,shutil,subprocess,sys,tempfile
import numpy as np
P=pathlib.Path
sys.path.insert(0,str(P(__file__).resolve().parent))
import TRAIN_raw_probe_driver_v1 as original
core=original.core
DATA_SHA=original.DATA_SHA
def raw(v):return original.raw(v)
def sha(p):return original.sha(p)
def write(p,v):p.write_bytes(raw(v))
def describe(x,mean,scale,active,coef,fit,held):
 assert x.ndim==2 and all(a.shape==(x.shape[1],) for a in [mean,scale,active,coef])
 assert active.dtype.kind=='b' and np.isfinite(x).all() and all(np.isfinite(a).all() for a in [mean,scale,coef]) and (scale>0).all()
 assert len(set(fit)&set(held))==0 and sorted(list(fit)+list(held))==list(range(len(x)))
 # Saved statistics must bind the same feature construction and fit rows.
 mu=x[fit].mean(0);sd=x[fit].std(0,ddof=0);ac=sd>=1e-12
 assert np.array_equal(mu,mean) and np.array_equal(ac,active) and np.array_equal(np.where(ac,sd,1.),scale)
 z=(x-mean)/scale*active;terms=z*coef
 assert np.isfinite(z).all() and np.isfinite(terms).all()
 channel=[]
 for k in range(x.shape[1]):
  channel.append(dict(index=k,active=bool(active[k]),fit_mean=float(mean[k]),saved_scale=float(scale[k]),coefficient=float(coef[k]),fit_min=float(x[fit,k].min()),fit_max=float(x[fit,k].max()),held_min=float(x[held,k].min()),held_max=float(x[held,k].max()),fit_max_abs_z=float(np.abs(z[fit,k]).max()),held_max_abs_z=float(np.abs(z[held,k]).max()),held_max_abs_contribution=float(np.abs(terms[held,k]).max())))
 return z,terms,channel
def features_only(ds,rows):
 original.validate_manifest(rows);train=ds['train'];assert len(train)==len(rows)
 feats={k:[] for k in ['T','A','V']};ledger=dict(decoded_arrays=0,decoded_roles=set(),decoded_TRAIN_labels=0)
 missing=[]
 for record,row in zip(train,rows):
  assert isinstance(record,tuple) and len(record)==3 and record[2]==row['row_id'] and row['split']=='train'
  words,visual,audio=record[0];assert visual.shape==(len(words),47) and audio.shape==(len(words),74)
  # record[1] (label) deliberately never indexed or decoded.
  feats['T'].append(core.text_features(words));detail=dict(row_id=row['row_id'],video_id=row['video_id'],word_rows=len(words))
  for name,a,width in [('A',audio,74),('V',visual,47)]:
   value=original.decode(a,'train',ledger);bad=~np.isfinite(value);finite=np.isfinite(value)
   feats[name].append(core.numeric_features(value,width))
   detail[name]=dict(nonfinite_counts=bad.sum(0).tolist(),finite_zero_counts=((value==0)&finite).sum(0).tolist(),elements_per_channel=len(value))
  missing.append(detail)
 ledger['decoded_roles']=sorted(ledger['decoded_roles']);assert ledger['decoded_arrays']==2*len(rows) and ledger['decoded_TRAIN_labels']==0
 return {k:np.stack(v) for k,v in feats.items()},missing,ledger
def qualify():
 x=np.array([[1e-10,1.],[2e-10,1.],[3e-10,1.],[1.,1.]])
 fit=np.array([0,1,2]);held=np.array([3]);mu=x[fit].mean(0);sd=x[fit].std(0);ac=sd>=1e-12;scale=np.where(ac,sd,1.);coef=np.array([.2,0.])
 z,terms,ch=describe(x,mu,scale,ac,coef,fit,held);assert z[3,0]>1e9 and not ch[1]['active'] and terms[3,1]==0
 rejects={}
 for n,args in [('wrong_saved_mean',(x,mu+1.,scale,ac,coef,fit,held)),('nonfinite_input',(x*np.nan,mu,scale,ac,coef,fit,held)),('overlapping_rows',(x,mu,scale,ac,coef,fit,fit)),('invalid_scale',(x,mu,np.zeros(2),ac,coef,fit,held))]:
  try:describe(*args)
  except AssertionError:rejects[n]=True
  else:raise AssertionError(n)
 class NoOtherSplit(dict):
  def __getitem__(self,k):assert k=='train';return super().__getitem__(k)
 class NoLabel(tuple):
  def __getitem__(self,k):assert k!=1,'Label accessed';return super().__getitem__(k)
 rows=[dict(row_index=i,row_id=str(i),video_id=str(i),split='train',held_out_fold=i%5) for i in range(10)]
 v=np.ones((2,47));v[0,0]=np.nan;a=np.ones((2,74));a[1,1]=np.inf
 ds=NoOtherSplit(train=[NoLabel(((['a','b'],original.make_opaque(v),original.make_opaque(a)),object(),str(i))) for i in range(10)])
 feats,miss,ledger=features_only(ds,rows);assert ledger['decoded_arrays']==20 and ledger['decoded_TRAIN_labels']==0 and miss[0]['V']['nonfinite_counts'][0]==1
 for n,action in [('wrong_ID',lambda:features_only(ds,[dict(rows[0],row_id='bad')]+rows[1:])),('wrong_split',lambda:features_only(ds,[dict(rows[0],split='test')]+rows[1:]))]:
  try:action()
  except AssertionError:rejects[n]=True
  else:raise AssertionError(n)
 with tempfile.TemporaryDirectory() as td:
  p=P(td)/'once';original.claim_once(p,{'synthetic':True})
  try:original.claim_once(p,{})
  except FileExistsError:rejects['duplicate_once']=True
  else:raise AssertionError('Duplicate once')
 # Diagnostic cannot invoke fitting, even when imported originals expose it.
 calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(P(__file__).read_text())) if isinstance(n,ast.Call)]
 assert 'np.linalg.solve' not in calls and 'core.fit_predict' not in calls
 return dict(status='SYNTHETIC_TRAIN_SAVED_SUPPORT_QUALIFIED',numpy=np.__version__,tiny_fit_scale_amplification_detected=True,inactive_channel_preserved=True,labels_and_nonTRAIN_not_indexed=True,rejections=rejects,new_fit_or_score=False)
def run(planpath,expected):
 assert sha(planpath)==expected;p=json.loads(planpath.read_bytes());r=P(p['root']);assert not r.exists() and not P(p['once_token']).exists()
 for n,h in p['source_SHA'].items():assert sha(P(__file__).parent/n)==h
 assert np.__version__=='1.26.4' and P(sys.executable).resolve()==P(p['interpreter']).resolve()
 for key in ['data','rows','states','fits','predictions','protocol']:assert sha(P(p[key]))==p[key+'_SHA']
 assert p['data_SHA']==DATA_SHA
 now=datetime.datetime.now(datetime.timezone.utc);remaining=(datetime.datetime.fromisoformat(p['human_conservative_deadline'])-now).total_seconds();assert remaining>3*3600
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==p['UUID']
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not compute
 mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in P('/proc/meminfo').read_text().splitlines())};assert mem['MemAvailable']>=6*1024**3 and shutil.disk_usage(r.parent).free>256*1024**2
 rows=json.loads(P(p['rows']).read_bytes());original.validate_manifest(rows);assert len(rows)==1281 and len({x['video_id'] for x in rows})==52
 payload=P(p['data']).read_bytes();assert not any('FLOAT' in op.name for op,arg,pos in pickletools.genops(payload))
 ds=original.MetadataOnly(io.BytesIO(payload)).load()
 # Freshness and once are checked before the first real numerical decode.
 r.mkdir();write(r/'fresh_preflight.json',dict(actual_UTC=now.isoformat(),identity=p['execution_identity'],UUID=uuid,RAM_available=mem['MemAvailable'],space=shutil.disk_usage(r.parent)._asdict(),compute=compute,fullargv=sys.argv,processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True),source_SHA=p['source_SHA'],runtime=dict(python=sys.version,numpy=np.__version__,interpreter=sys.executable),remaining_seconds=remaining))
 assert (datetime.datetime.now(datetime.timezone.utc)-now).total_seconds()<300
 original.claim_once(p['once_token'],dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_SHA=expected,role='TRAIN_FEATURE_SUPPORT_ONLY_NO_LABELS_NO_FIT_NO_SCORE'))
 features,missing,ledger=features_only(ds,rows);del ds,payload
 paths={'T':features['T'],'T_A':np.column_stack([features['T'],features['A']]),'T_V':np.column_stack([features['T'],features['V']]),'T_A_V':np.column_stack([features['T'],features['A'],features['V']])}
 states=np.load(p['states'],allow_pickle=False);preds=np.load(p['predictions'],allow_pickle=False);fits=json.loads(P(p['fits']).read_bytes())
 assert preds['row_id'].tolist()==[x['row_id'] for x in rows] and preds['video_id'].tolist()==[x['video_id'] for x in rows] and preds['fold'].tolist()==[x['held_out_fold'] for x in rows]
 fold=np.array([x['held_out_fold'] for x in rows]);channels=[];video=[];max_error=0.
 for f in range(5):
  fit=np.flatnonzero(fold!=f);held=np.flatnonzero(fold==f)
  for name,x in paths.items():
   key='fold'+str(f)+'_'+name;rec=next(v for v in fits if v['fold']==f and v['path'].replace('+','_')==name)
   assert rec['fit_ID_SHA']==original.identity_sha([rows[i]['row_id'] for i in fit]) and rec['held_ID_SHA']==original.identity_sha([rows[i]['row_id'] for i in held])
   z,t,ch=describe(x,states[key+'_mean'],states[key+'_scale'],states[key+'_active'],states[key+'_coefficient'],fit,held)
   reconstruction=z[held]@states[key+'_coefficient']+rec['intercept'];saved=preds['prediction_'+name][held]
   error=float(np.abs(reconstruction-saved).max());assert error<1e-10;max_error=max(max_error,error)
   channels.append(dict(fold=f,path=name,channels=ch))
   for vid in sorted({rows[i]['video_id'] for i in held}):
    ix=np.array([i for i in held if rows[i]['video_id']==vid]);az=np.abs(z[ix]);at=np.abs(t[ix]);rr,cc=np.unravel_index(np.argmax(at),at.shape);i=int(ix[rr]);k=int(cc)
    video.append(dict(video_id=vid,fold=f,path=name,rows=len(ix),max_abs_standardized_feature=float(az.max()),max_abs_coefficient_term=float(at.max()),standardized_elements_gt10=int((az>10).sum()),standardized_elements_gt100=int((az>100).sum()),standardized_elements_gt1000=int((az>1000).sum()),largest_term=dict(row_id=rows[i]['row_id'],channel_index=k,raw_summary=float(x[i,k]),fit_mean=ch[k]['fit_mean'],saved_scale=ch[k]['saved_scale'],fit_min=ch[k]['fit_min'],fit_max=ch[k]['fit_max'],z=float(z[i,k]),coefficient=float(states[key+'_coefficient'][k]),term=float(t[i,k]))))
 write(r/'per_channel_saved_fit_support.json',channels);write(r/'per_video_saved_fit_support.json',video);write(r/'TRAIN_raw_nonfinite_and_zero_counts.json',missing);write(r/'decoder_ledger.json',ledger)
 write(r/'support_result.json',dict(status='TRAIN_SAVED_FIT_SUPPORT_DESCRIPTION_COMPLETE',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=1281,videos=52,paths=4,fold_paths=20,max_original_prediction_reconstruction_error=max_error,TRAIN_labels_decoded=0,VAL_TEST_numeric_decode=False,new_solve_fit_score=False,primary_results_changed=False,source_units_and_missing_sentinel_unknown=True,no_causal_or_information_claim=True,input_SHA={k:p[k+'_SHA'] for k in ['data','rows','states','fits','predictions','protocol']}))
 for n in p['source_SHA']:shutil.copyfile(P(__file__).parent/n,r/n)
 shutil.copyfile(planpath,r/planpath.name);print(json.dumps(json.loads((r/'support_result.json').read_bytes())),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--qualify',action='store_true');a.add_argument('--plan',type=P);a.add_argument('--sha');v=a.parse_args()
 if v.qualify:assert not v.plan;print(json.dumps(qualify(),allow_nan=False))
 else:assert v.plan and v.sha;run(v.plan,v.sha)
