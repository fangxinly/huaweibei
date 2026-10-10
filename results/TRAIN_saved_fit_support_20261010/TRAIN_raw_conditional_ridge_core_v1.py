"""Fixed raw-input TRAIN diagnostic. Synthetic qualification entry point only.

Predictive gains of this finite linear estimator are neither conditional MI,
PID, causality, nor a mechanism test for the previously trained A/B models.
"""
import hashlib,json,sys
import numpy as np

def text_features(words):
 assert isinstance(words,list) and words and all(isinstance(w,str) for w in words)
 out=np.zeros(257,dtype=np.float64)
 for word in words:
  digest=hashlib.sha256(('TRAIN_probe_seed128:'+word.casefold()).encode('utf8')).digest()
  out[int.from_bytes(digest[:8],'little')%256]+=(1. if digest[8]&1 else -1.)/len(words)
 out[-1]=np.log1p(len(words))
 return out

def numeric_features(value,channels):
 value=np.asarray(value,dtype=np.float64)
 assert value.ndim==2 and value.shape[0]>0 and value.shape[1]==channels
 missing=~np.isfinite(value);clean=np.where(missing,0.,value)
 return np.concatenate([clean.mean(0),clean.std(0,ddof=0),[missing.mean()]])

def fit_predict(x_fit,y_fit,x_held):
 x_fit=np.asarray(x_fit,dtype=np.float64);x_held=np.asarray(x_held,dtype=np.float64);y_fit=np.asarray(y_fit,dtype=np.float64)
 assert x_fit.ndim==x_held.ndim==2 and x_fit.shape[1]==x_held.shape[1] and y_fit.shape==(len(x_fit),)
 assert np.isfinite(x_fit).all() and np.isfinite(x_held).all() and np.isfinite(y_fit).all()
 mean=x_fit.mean(0);scale=x_fit.std(0,ddof=0);active=scale>=1e-12;scale=np.where(active,scale,1.)
 fit=(x_fit-mean)/scale*active;held=(x_held-mean)/scale*active;center=float(y_fit.mean())
 # Mean squared fitting error + fixed lambda1 L2 penalty; intercept unpenalized.
 coef=np.linalg.solve(fit.T@fit+len(fit)*np.eye(fit.shape[1]),fit.T@(y_fit-center))
 prediction=held@coef+center
 assert np.isfinite(prediction).all()
 return prediction,dict(mean=mean,scale=scale,active=active,coefficient=coef,intercept=center,fit_rows=len(fit),lambda_mean_loss=1.)

def qualification():
 rng=np.random.default_rng(128);x=rng.normal(size=(240,3));y=4*x[:,1]+.05*rng.normal(size=240)
 baseline,_=fit_predict(x[:160,:1],y[:160],x[160:,:1]);joint,fit=fit_predict(x[:160,:2],y[:160],x[160:,:2])
 assert np.mean((joint-y[160:])**2)<.6*np.mean((baseline-y[160:])**2)
 shifted=x[160:,:2]+1000.;_,changed=fit_predict(x[:160,:2],y[:160],shifted)
 assert np.array_equal(fit['mean'],changed['mean']) and np.array_equal(fit['coefficient'],changed['coefficient'])
 constant=np.column_stack([x[:160,0],np.ones(160)]);held=np.column_stack([x[160:,0],np.ones(80)])
 pred,state=fit_predict(constant,y[:160],held);assert not state['active'][1] and state['coefficient'][1]==0
 words=['Hello','world'];assert np.array_equal(text_features(words),text_features(words)) and text_features(words).shape==(257,)
 audio=np.ones((2,74));audio[0,0]=np.nan;assert np.isfinite(numeric_features(audio,74)).all() and numeric_features(audio,74).shape==(149,)
 rejected=False
 try:fit_predict(np.array([[np.nan],[1.]]),np.array([1.,2.]),np.array([[0.]]))
 except AssertionError:rejected=True
 assert rejected
 return dict(status='SYNTHETIC_FIXED_RIDGE_AND_FEATURE_QUALIFICATION_PASSED',numpy_version=np.__version__,conditional_signal_detected=True,held_feature_shift_cannot_change_fit_statistics_or_weights=True,constant_channel_rejected_from_fit=True,nonfinite_fit_rejected=True,raw_modality_missingness_handling_finite=True,real_data_loaded=False,real_probe_executed=False)

if __name__=='__main__':
 assert sys.argv[1:]==['--qualify'],'Only synthetic qualification is callable in this source.'
 print(json.dumps(qualification(),allow_nan=False))
