"""Audit saved TRAIN-only finite fit states, then report the already saved predictions."""
import datetime,hashlib,io,json,os,pathlib,subprocess,sys,zipfile
import numpy as np
P=pathlib.Path
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_fit(x,y,held,pred,state,intercept):
 mean=x.mean(0);s=x.std(0,ddof=0);active=s>=1e-12;s=np.where(active,s,1.)
 assert np.array_equal(mean,state['mean']) and np.array_equal(s,state['scale']) and np.array_equal(active,state['active'])
 assert intercept==float(y.mean())
 xf=(x-mean)/s*active;xh=(held-mean)/s*active;coef=state['coefficient']
 residual=(xf.T@xf+len(x)*np.eye(x.shape[1]))@coef-xf.T@(y-intercept)
 relative=float(np.max(np.abs(residual))/max(1.,np.max(np.abs(xf.T@(y-intercept)))))
 err=float(np.max(np.abs(xh@coef+intercept-pred)))
 assert relative<1e-10 and err<1e-10 and np.isfinite(coef).all()
 return dict(relative_normal_equation_residual=relative,prediction_max_abs_error=err)
def qualify(bundle):
 sys.path.insert(0,str(bundle));import TRAIN_raw_conditional_ridge_core_v1 as core
 rng=np.random.default_rng(128);x=rng.normal(size=(20,3));y=rng.normal(size=20);held=rng.normal(size=(4,3));pred,state=core.fit_predict(x,y,held)
 check_fit(x,y,held,pred,state,state['intercept']);wrong=dict(state,coefficient=state['coefficient']+1.)
 try:check_fit(x,y,held,pred,wrong,state['intercept'])
 except AssertionError:pass
 else:raise AssertionError('Incorrect saved coefficients accepted')
 print(json.dumps(dict(status='SYNTHETIC_SAVED_RIDGE_AUDIT_QUALIFIED',wrong_coefficients_rejected=True,real_data_loaded=False)))
def run(base,release_receipt,expected_receipt_SHA):
 sys.path.insert(0,str(base/'payload'));import TRAIN_raw_probe_driver_v1 as driver;import TRAIN_raw_probe_report_v1 as reporter
 assert sha(release_receipt)==expected_receipt_SHA
 pub=json.loads(release_receipt.read_bytes());assert pub['remote_digest_verified'] and pub['source_SHA']==sha(base/'complete_actual_raw_TRAIN_capture.zip')
 plan=json.loads((base/'payload/execution_plan.json').read_bytes())
 assert str(P(sys.executable).resolve())==str(P(plan['interpreter']).resolve()) and np.__version__=='1.26.4'
 for n,h in plan['source_SHA'].items():assert sha(base/'payload'/n)==h
 now=datetime.datetime.now(datetime.timezone.utc);assert (datetime.datetime.fromisoformat(plan['human_conservative_deadline'])-now).total_seconds()>3*3600
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==plan['UUID']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 mem=dict((k,int(v.split()[0])*1024) for k,v in (line.split(':',1) for line in P('/proc/meminfo').read_text().splitlines()));assert mem['MemAvailable']>=6*1024**3
 out=base/'analysis';assert not out.exists() and not (base/'once_raw_TRAIN_analysis.json').exists()
 rows=json.loads(P(plan['fold_manifest']).read_bytes());driver.validate_manifest(rows)
 pr=base/'prediction';result=json.loads((pr/'prediction_result.json').read_bytes());npz=pr/'TRAIN_fixed_fivefold_prediction.npz';statespath=pr/'fit_fold_states.npz'
 assert sha(npz)==result['prediction_SHA'] and sha(statespath)==result['states_SHA'] and not result['VAL_TEST_numeric_decode']
 with np.load(npz,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
 assert arrays['row_id'].tolist()==[r['row_id'] for r in rows] and arrays['video_id'].tolist()==[r['video_id'] for r in rows] and arrays['fold'].tolist()==[r['held_out_fold'] for r in rows]
 assert sha(P(plan['data']))==driver.DATA_SHA
 ds=driver.MetadataOnly(io.BytesIO(P(plan['data']).read_bytes())).load();features,y,ledger=driver.features_from_train(ds,rows);del ds
 assert np.array_equal(y,arrays['label'])
 paths={'T':features['T'],'T_A':np.concatenate([features['T'],features['A']],1),'T_V':np.concatenate([features['T'],features['V']],1),'T_A_V':np.concatenate([features['T'],features['A'],features['V']],1)}
 fits=json.loads((pr/'fit_identity_records.json').read_bytes());audit=[]
 with np.load(statespath,allow_pickle=False) as stored:
  for f in range(5):
   fit=np.flatnonzero(arrays['fold']!=f);held=np.flatnonzero(arrays['fold']==f)
   for name,x in paths.items():
    item=next(v for v in fits if v['fold']==f and v['path'].replace('+','_')==name)
    assert item['fit_ID_SHA']==driver.identity_sha([rows[i]['row_id'] for i in fit]) and item['held_ID_SHA']==driver.identity_sha([rows[i]['row_id'] for i in held]) and item['lambda_mean_loss']==1.
    key='fold'+str(f)+'_'+name;state={k:stored[key+'_'+k] for k in ['mean','scale','active','coefficient']}
    check=check_fit(x[fit],y[fit],x[held],arrays['prediction_'+name][held],state,item['intercept']);audit.append(dict(fold=f,path=name,**check))
 assert (datetime.datetime.now(datetime.timezone.utc)-now).total_seconds()<300
 out.mkdir();driver.claim_once(base/'once_raw_TRAIN_analysis.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),prediction_SHA=result['prediction_SHA'],entry_source_SHA=sha(P(__file__))))
 (out/'saved_fit_CPU_audit.json').write_bytes(raw(dict(status='REAL_TRAIN_SAVED_FIT_CPU_AUDIT_PASSED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),prediction_SHA=result['prediction_SHA'],checks=audit,TRAIN_decoder_ledger=ledger,fit_states_retrained=False,VAL_TEST_numeric_decode=False,UUID=uuid,RAM_available=mem['MemAvailable'],entry_source_SHA=sha(P(__file__),),original_sources=plan['source_SHA'])))
 report=reporter.report(arrays['row_id'].tolist(),arrays['video_id'].tolist(),arrays['fold'].tolist(),y,{k:arrays['prediction_'+k] for k in reporter.PATHS})
 report.update(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),prediction_SHA=result['prediction_SHA'],protocol_SHA=result['protocol_SHA'],saved_fit_CPU_audit_passed=True,official_AB_tokens_untouched=True)
 (out/'TRAIN_finite_prediction_report.json').write_bytes(raw(report));(out/'Release_prediction_preserved_before_metrics.json').write_bytes(release_receipt.read_bytes())
 for p in [P(__file__),base/'payload/TRAIN_raw_probe_report_v1.py']: (out/p.name).write_bytes(p.read_bytes())
 capture=driver.archive(out,'complete_actual_TRAIN_audit_report.zip');(out/'capture_receipt.json').write_bytes(raw(capture));print(json.dumps(dict(report_MSE=report['pooled_row_weighted_MSE'],pairs=report['paired_comparisons'],capture=capture)),flush=True)
if __name__=='__main__':
 if sys.argv[1]=='--qualify':qualify(P(sys.argv[2]))
 else:assert sys.argv[1]=='--run';run(P(sys.argv[2]),P(sys.argv[3]),sys.argv[4])
