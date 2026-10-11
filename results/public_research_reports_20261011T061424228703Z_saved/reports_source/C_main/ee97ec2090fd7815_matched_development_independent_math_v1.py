"""Independent arithmetic for audits. No official asset or model forward."""
import numpy as np
def readouts(m,f,c):
 f=np.asarray(f,dtype=np.float64);c=np.asarray(c,dtype=np.float64);d=c-f;raw=np.column_stack([f,d]);sd=np.asarray(m['std']);mu=np.asarray(m['mean']);z=np.zeros_like(raw)
 for j in range(2):
  if sd[j]>0:z[:,j]=(raw[:,j]-mu[j])/sd[j]
 X=np.concatenate([np.ones((len(f),1)),z],axis=1);out={'F':f,'C':c,'constant_free':f+float(m['constant'])}
 for n in ('U','W'):
  h=X@np.asarray(m[n]);out[n+'_free']=f+h;out[n+'_interval']=f+np.minimum(np.maximum(h,np.minimum(d,0)),np.maximum(d,0));out[n+'_discrete']=np.where(d*d-2*d*h<0,c,f)
 return out
def independent_metrics(p,y):
 p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 result={'Acc7':float(np.count_nonzero(np.rint(np.clip(p,-3,3))==np.rint(np.clip(y,-3,3)))/len(y)),'MAE':float(sum(np.abs(p-y))/len(y)),'MSE':float(sum((p-y)**2)/len(y))}
 nonzero=y!=0
 for mask,prefix in [(nonzero,''),(np.ones(len(y),dtype=bool),'Has0_')]:
  yy=y[mask]>=0;pp=p[mask]>=0;N=len(yy);conf=np.zeros((2,2),dtype=np.int64)
  for t,b in zip(yy,pp):conf[int(t),int(b)]+=1
  support=conf.sum(axis=1);f1=0.
  for k in range(2):
   denom=int(conf[k].sum()+conf[:,k].sum())
   if denom:f1+=float(support[k])*(2*int(conf[k,k])/denom)
  result[prefix+'Acc2']=None if N==0 else float(int(np.trace(conf))/N);result[prefix+'F1']=None if N==0 else f1/N
  result['nonzero_confusion_matrix' if prefix=='' else 'haszero_confusion_matrix']=conf.tolist()
 pc=p-float(sum(p)/len(p));yc=y-float(sum(y)/len(y));denom=np.sqrt(float(sum(pc*pc))*float(sum(yc*yc)))
 result.update(Corr=None if denom==0 or len(y)<2 else float(sum(pc*yc)/denom),samples=len(y),nonzero_samples=int(nonzero.sum()),zero_label_samples=int((~nonzero).sum()),exact_zero_predictions=int(np.sum(p==0)),correlation_defined=bool(denom>0 and len(y)>1))
 return result
def compare_metrics(original,independent):
 errors=[]
 for k,val in independent.items():
  if isinstance(val,(float,np.floating)):
   error=abs(float(original[k])-val);assert error<1e-12,(k,error);errors.append(error)
  else:assert original[k]==val,k
 return max(errors,default=0.)
