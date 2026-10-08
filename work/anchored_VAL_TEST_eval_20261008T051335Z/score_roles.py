"""Single final VAL/TEST five-metric evaluation after original D/CPU preservation."""
import argparse,os,sys,pickle,csv
from pathlib import Path
import numpy as np
from common import sha,read,write,utc,verify
from sentiment_metrics_careflow_v1 import metrics,SEMANTICS
def independent(v,y):
    keep=y!=0;yt=y[keep]>=0;pt=v[keep]>=0;f=0.
    for label in (False,True):
        support=int(np.sum(yt==label));count=int(np.sum(pt==label));tp=int(np.sum((yt==label)&(pt==label)));f+=support*2*tp/(support+count) if support+count else 0.
    return dict(Acc7=float(np.mean(np.rint(np.clip(v,-3,3))==np.rint(np.clip(y,-3,3)))),Acc2=float(np.mean(yt==pt)),F1=f/len(yt),MAE=float(np.mean(abs(v-y))),Corr=float(np.corrcoef(v,y)[0,1]))
def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.bundle,a.assets);assert p['original_D_CPU_score_gate'];assert sha(a.prediction)==p['role_prediction_reference']['SHA'];a.out.mkdir()
    fd=os.open(p['score_once_token'],os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,a.plan_sha.encode());os.close(fd)
    pred=dict(np.load(a.prediction,allow_pickle=False))
    with (a.assets/'assets/mosi.pkl').open('rb') as f:whole=pickle.load(f)
    result={};errors={};labels={}
    for name,key in [('VAL','dev'),('TEST','test')]:
        records=whole[key];ids=[r[2].decode() if isinstance(r[2],bytes) else r[2] for r in records];assert ids==pred[name+'_ids'].tolist()==p['official_role_IDs'][name]
        y=np.asarray([np.asarray(r[1]).reshape(-1)[0] for r in records],dtype=np.float32).astype(float);assert np.isfinite(y).all() and (abs(y)<=3).all();labels[name+'_ids']=np.asarray(ids);labels[name+'_y']=y
        table={};errors[name]={}
        for model,col in [('new_fixed20_upgrade','prediction'),('same_flow_messages_off','p0')]:
            v=pred[name+'_'+col].astype(float);table[model]={subset:metrics(v[m],y[m]) for subset,m in [('overall',np.ones(len(y),dtype=bool)),('weak',abs(y)<=1),('strong',abs(y)>1)]};check=independent(v,y);err=max(abs(check[k]-table[model]['overall'][k]) for k in check);assert err<1e-12;errors[name][model]=err
        overlap=p['official_role_training_overlap'][name];result[name]=dict(rows=len(y),videos=len(set(s.split('[')[0] for s in ids)),training_overlap=overlap,metrics=table,weak_count=int((abs(y)<=1).sum()),strong_count=int((abs(y)>1).sum()))
        with (a.out/(name+'_fixed_predictions_and_labels.csv')).open('w',encoding='utf8',newline='') as f:
            w=csv.writer(f);w.writerow(['row_id','video','y','new_prediction','same_flow_off_prediction','merged_fold_role'])
            rolemap=p['merged_ID_to_role']
            for i,s in enumerate(ids):w.writerow([s,s.split('[')[0],y[i],pred[name+'_prediction'][i],pred[name+'_p0'][i],rolemap[s]])
    np.savez(a.out/'actual_VAL_TEST_labels.npz',**labels)
    write(a.out/'actual_VAL_TEST_five_metrics.json',dict(status='FIXED_NEW_UPGRADE_VAL_TEST_FIVE_METRICS_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,prediction_SHA=p['role_prediction_reference']['SHA'],adapter_state_SHA=p['adapter_state_SHA'],roles=result,metric_semantics=SEMANTICS,independent_arithmetic_max_errors=errors,no_training_or_checkpoint_selection=True,all_five_same_fixed_checkpoint=True,original_test_overlap_in_training_disclosed=True,not_independent_official_benchmark=True));print(read(a.out/'actual_VAL_TEST_five_metrics.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','prediction','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())
