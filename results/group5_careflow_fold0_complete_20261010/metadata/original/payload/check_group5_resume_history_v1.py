"""Small protocol counterexamples, no scientific arrays or fitting."""
import ast,copy,json,pathlib
from group5_epoch_resume_v1 import history_state

def main():
 rows=[];best=float('inf');which=0
 for epoch,mse in enumerate([1.,.5,.5,.7],1):
  if mse<best:best,which=mse,epoch
  rows.append(dict(epoch=epoch,updates=epoch*47,inner_MSE=mse,best_epoch=which,best_MSE=best))
 assert history_state(rows,47)==dict(completed_epochs=4,updates=188,best_MSE=.5,best_epoch=2)
 bad=[]
 for field,value in [('epoch',9),('updates',48),('inner_MSE',float('nan')),('inner_MSE',-1),('best_epoch',3),('best_MSE',.4)]:
  x=copy.deepcopy(rows);x[2][field]=value;bad.append(x)
 bad.extend([[],rows+rows])
 for x in bad:
  try:history_state(x,47)
  except ValueError:pass
  else:raise AssertionError('Invalid recovery history accepted')
 root=pathlib.Path(__file__).parent
 for name in ['group5_epoch_resume_v1.py','group5_direct_runtime_v2.py']:ast.parse((root/name).read_text(encoding='utf8'))
 print(json.dumps(dict(status='EPOCH_RESUME_HISTORY_PROTOCOL_SYNTHETIC_PASS',invalid_histories_rejected=len(bad),scientific_imports_or_arrays_or_training=False,native_tensor_RNG_restoration_qualified=False)))
if __name__=='__main__':main()
