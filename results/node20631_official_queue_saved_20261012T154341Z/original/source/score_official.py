"""Once fixed prediction has genuine D preservation, all five paired scores."""
import argparse,csv,os,pickle,sys
from pathlib import Path
import numpy as np
from common import sha,read,write,utc,verify
from sentiment_metrics_careflow_v1 import metrics,SEMANTICS

def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets);assert p['prediction_D_gate'];assert sha(a.prediction)==p['prediction_SHA'];a.out.mkdir()
    fd=os.open(p['score_once_token'],os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,a.plan_sha.encode());os.close(fd)
    pred=dict(np.load(a.prediction,allow_pickle=False))
    with (a.assets/'assets/mosi.pkl').open('rb') as f:data=pickle.load(f)
    baseline=p['fixed_CaReFlow_five'];roles={}
    for name,key in [('VAL','dev'),('TEST','test')]:
        records=data[key];ids=[r[2].decode() if isinstance(r[2],bytes) else r[2] for r in records];assert ids==p['official_role_IDs'][name]==pred[name+'_ids'].tolist();y=np.asarray([np.asarray(r[1]).reshape(-1)[0] for r in records],np.float32).astype(float)
        new=metrics(pred[name+'_prediction'],y);off=metrics(pred[name+'_p0'],y);b=baseline[name]
        margins={k:new[k]-b[k] for k in ('Acc7','Acc2','F1','MAE','Corr')};allwin=all(margins[k]>0 for k in ('Acc7','Acc2','F1','Corr')) and margins['MAE']<0
        roles[name]=dict(rows=len(y),new=new,messages_off=off,careflow=b,new_minus_careflow=margins,all_five_strict_point_improvement=allwin,groups={g:metrics(pred[name+'_prediction'][m],y[m]) for g,m in [('weak',abs(y)<=1),('strong',abs(y)>1)]})
        with (a.out/(name+'_actual_predictions.csv')).open('w',encoding='utf8',newline='') as f:
            w=csv.writer(f);w.writerow(['row_id','video','y','new','messages_off'])
            for i,s in enumerate(ids):w.writerow([s,s.split('[')[0],y[i],pred[name+'_prediction'][i],pred[name+'_p0'][i]])
    write(a.out/'official_aligned_five_result.json',dict(status='OFFICIAL_TRAIN_VAL_TEST_ALIGNED_UPGRADE_CAREFLOW_FIVE_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,roles=roles,prediction_SHA=p['prediction_SHA'],selected_state_SHA=p['selected_state_SHA'],baseline_source_reference=p['CaReFlow_baseline_original_reference'],same_official_roles=True,TRAIN1281_only=True,VAL229_selection_only=True,TEST685_not_used_for_training_or_selection=True,seed128_100epochs_4000updates_common_orders_common_selection=True,metric_semantics=SEMANTICS,historical_TEST_access_disclosed=True,not_a_fresh_blinded_dataset_or_multiseed=True))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','out','prediction'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())
