"""Read saved histories and inspect fixed TRAIN/CAL heads; no INNER labels or training."""
import argparse,datetime,io,json,os,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from contract import sha,read,write
from pipeline import Mechanism
def run(a):
    assert sha(a.plan)==a.plan_sha;p=read(a.plan);torch.set_num_threads(2)
    assert sha(a.train)==p['train_archive_SHA'] and sha(a.cache)==p['cache_archive_SHA']
    with zipfile.ZipFile(a.train) as z:
        s=torch.load(io.BytesIO(z.read('out/complete_small_training_state.pt')),map_location='cpu');roles=json.loads(z.read('out/actual_role_identity.json'));fit_y=np.load(io.BytesIO(z.read('out/original_FIT_supervision.npy')),allow_pickle=False)
    ymap=dict(zip(roles['fit_ids'],fit_y));tr_y=np.array([ymap[r] for r in roles['task_train_ids']]);cal_y=np.array([ymap[r] for r in roles['calibration_ids']])
    with zipfile.ZipFile(a.cache) as z:
        f=np.load(io.BytesIO(z.read('out/public_frozen_features.npz')),allow_pickle=False);ids=f['row_ids'].astype(str);pos={r:i for i,r in enumerate(ids)};ci=[pos[r] for r in roles['calibration_ids']];features=[f[k][ci] for k in ('text','audio','vision')]
    model=Mechanism.restore(s,'cpu');cal=model.stack.predict(features);rho=tr_y-s['oof']['p0'];crho=cal_y-cal['p0'];diagnostics={}
    def corr(x,y):return None if np.std(x)<1e-12 or np.std(y)<1e-12 else float(np.corrcoef(x,y)[0,1])
    with torch.no_grad():
        for name in ('flow','regression'):
            rh=model.gates[name](torch.tensor(s['oof'][name]['phi'])).numpy();ch=model.gates[name](torch.tensor(cal[name]['phi'])).numpy();d=s['oof'][name]['delta'];cd=cal[name]['delta']
            diagnostics[name]=dict(meta_training_rhat_corr=corr(rh,rho),meta_training_rhat_MSE=float(np.mean((rh-rho)**2)),meta_training_rhat_std=float(rh.std()),task_TRAIN_OOF_residual_std=float(rho.std()),CAL_rhat_corr=corr(ch,crho),CAL_rhat_MSE=float(np.mean((ch-crho)**2)),CAL_rhat_std=float(ch.std()),CAL_residual_std=float(crho.std()),CAL_slope=s['calibration']['slopes'][name],task_TRAIN_OOF_all_on_utility=float(np.mean(2*rho*d-d*d)),CAL_all_on_utility=float(np.mean(2*crho*cd-cd*cd)),meta_training_rows_not_OOF_for_controller=True)
    history={}
    for kind in sorted({r['kind'] for r in s['records']}):
        items=[r['complete']['history'] for r in s['records'] if r['kind']==kind];history[kind]=dict(models=len(items),first_40_mean=float(np.mean([x[:40] for x in items])),last_40_mean=float(np.mean([x[-40:] for x in items])))
    raw_pred=[];raw_y=[]
    for r in s['records']:
        if r['kind']=='baseline' and 9900<=r.get('serial',-1)<9905:raw_pred.extend(r['oof_prediction']);raw_y.extend([ymap[i] for i in r['prediction_held_ids']])
    raw_pred=np.array(raw_pred);raw_y=np.array(raw_y);assert len(raw_y)==len(tr_y)
    a.out.mkdir();result=dict(status='SAVED_TRAIN_CAL_DIAGNOSTIC_COMPLETE_NO_TRAIN_OR_INNER_LABELS',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,train_archive_SHA=p['train_archive_SHA'],baseline_full_stack_alpha=s['stack']['alpha'],baseline_full_stack_beta=s['stack']['beta'],raw_baseline_TRAIN_OOF_MAE=float(np.mean(abs(raw_pred-raw_y))),raw_baseline_TRAIN_OOF_MSE=float(np.mean((raw_pred-raw_y)**2)),meta_TRAIN_OOF_calibrated_baseline_MAE=float(np.mean(abs(tr_y-s['oof']['p0']))),CAL_baseline_MAE=float(np.mean(abs(crho))),calibration_head_diagnostic=diagnostics,training_histories=history,TRAIN_CAL_scalar_labels_from_already_saved_original_FIT=True,INNER_labels_read=False,checkpoint_modified=False,training=False,small_head_CPU_forward=True,encoder_CPU_forward=False)
    write(a.out/'actual_saved_train_diagnostic.json',result);print(result)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','train','cache','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())
