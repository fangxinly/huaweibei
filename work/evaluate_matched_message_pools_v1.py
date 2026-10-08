"""Frozen EVAL-only post-prediction accounting. No parameter selection or fitting."""
from pathlib import Path
import argparse,datetime,hashlib,json,zipfile,numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def select_labels(path,requested,authorized):
    if not np.array_equal(requested,authorized):raise PermissionError('Exactly fixed EVAL rows only')
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist())=={'row.npy','y.npy'}
        with z.open('y.npy') as f:
            version=np.lib.format.read_magic(f)
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
            assert shape==(1281,) and not fortran and dtype.kind=='f';offset=f.tell();y=[]
            for i in requested:
                f.seek(offset+int(i)*dtype.itemsize);b=f.read(dtype.itemsize);assert len(b)==dtype.itemsize
                y.append(float(np.frombuffer(b,dtype=dtype,count=1)[0]))
    return np.asarray(y,dtype=np.float64)
def metrics(pred,pf,native,y,video):
    e=pred-y;eq=e*e;ref=(pf-y)**2;q=eq-ref;qn=eq-(native-y)**2;move=pred-pf;accept=move!=0
    per=[dict(video=str(v),rows=int((video==v).sum()),mse=float(eq[video==v].mean()),mae=float(np.abs(e[video==v]).mean()),risk_change_F=float(q[video==v].mean()),risk_change_native=float(qn[video==v].mean())) for v in np.unique(video)]
    return dict(mse=float(eq.mean()),mae=float(np.abs(e).mean()),video_equal_mse=float(np.mean([v['mse'] for v in per])),video_equal_mae=float(np.mean([v['mae'] for v in per])),
        mean_risk_change_F=float(q.mean()),mean_risk_change_native=float(qn.mean()),acceptance_rows=int(accept.sum()),harmful_fraction_accepted=float(np.mean(q[accept]>1e-12)) if accept.any() else None,
        beneficial_fraction_accepted=float(np.mean(q[accept]<-1e-12)) if accept.any() else None,all_row_harmful_fraction=float(np.mean(q>1e-12)),Q95=float(np.quantile(q,.95)),Q99=float(np.quantile(q,.99)),
        mean_linear_term=float(np.mean(2*(pf-y)*move)),mean_quadratic_term=float(np.mean(move*move)),per_video=per)
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    plan=json.loads((r/'evaluation_plan.json').read_text());assert sha(__file__)==plan['source_sha256']
    assert sha(r/'plan.json')==plan['gpu_plan_sha256']
    assert sha(plan['label_archive'])==plan['label_archive_sha256']
    receipt=json.loads((r/'execute/receipt.json').read_text());ex=json.loads((r/'execute_exit.json').read_text())
    assert receipt['passed'] and receipt['rows']==1281 and ex['exit_code']==0 and not receipt['real_labels_read']
    preds=r/'execute/predictions_frozen.npz';psha=sha(preds);assert psha==receipt['prediction_sha256']
    audit=json.loads((r/'execute/independent_audit.json').read_text());assert audit['arrays_sha256']==psha
    z=np.load(preds,allow_pickle=False);roles=np.load(r/'roles.npz',allow_pickle=False)
    rows=np.flatnonzero(roles['role']=='evaluation');assert len(rows)==863 and len(np.unique(roles['video'][rows]))==34
    assert np.array_equal(z['row'],np.arange(1281)) and np.array_equal(z['role'],roles['role'])
    # Every arm was already produced and hashed before any requested numeric EVAL y.
    out=r/'evaluation';out.mkdir(exist_ok=False)
    (out/'label_access_gate.json').write_text(json.dumps(dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_eight_predictions_sha256=psha,original_execute_exit_sha256=sha(r/'execute_exit.json'),evaluation_plan_sha256=sha(r/'evaluation_plan.json'),EVAL_values_not_yet_decoded=True),indent=2))
    y=select_labels(plan['label_archive'],rows,rows)
    pf=z['pf'][rows].astype(np.float64);native=z['native_prediction'][rows].astype(np.float64);video=z['video'][rows];pred=z['selected_prediction'][rows].astype(np.float64)
    names=['old_O','old_T','old_OT','old_Opm','cal_O','cal_T','cal_OT','cal_Opm'];result={'F':metrics(pf,pf,native,y,video),'native':metrics(native,pf,native,y,video)}
    for i,name in enumerate(names):result[name]=metrics(pred[:,i],pf,native,y,video)
    pools=[[0,1,2,3,4],[0,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,9,10,11,12]]
    oracle=[];oracle_idx=[];allpred=z['candidate_prediction'][rows].astype(np.float64);valid=z['candidate_valid'][rows]
    for ids in pools:
        error=(allpred[:,ids]-y[:,None])**2;error[~valid[:,ids]]=np.inf
        idx=np.asarray(ids)[error.argmin(1)];oracle_idx.append(idx);op=allpred[np.arange(len(rows)),idx];oracle.append(op)
    for i,name in enumerate(['oracle_O','oracle_T','oracle_OT','oracle_Opm']):result[name]=metrics(oracle[i],pf,native,y,video)
    np.savez_compressed(out/'eval_arrays.npz',row=rows,fold=z['fold'][rows],video=video,y=y,pf=pf,native=native,selected_prediction=pred,
        selected_index=z['selected_index'][rows],candidate_prediction=allpred,candidate_valid=valid,oracle_prediction=np.stack(oracle,1),oracle_index=np.stack(oracle_idx,1))
    assert sha(preds)==psha
    mainq=(pred[:,6]-y)**2-(pred[:,7]-y)**2;oq=(oracle[2]-y)**2-(oracle[3]-y)**2
    report=dict(status='MATCHED_MESSAGE_POOLS_EVAL_ACCOUNTING_COMPLETE_NOT_TRAINING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prediction_sha256=psha,
        source_sha256=sha(__file__),evaluation_plan_sha256=sha(r/'evaluation_plan.json'),eval_arrays_sha256=sha(out/'eval_arrays.npz'),rows=863,videos=34,metrics=result,
        main_cal_OT_minus_Opm_mean_squared_error=float(mainq.mean()),main_cal_OT_minus_Opm_video_equal=float(np.mean([mainq[video==v].mean() for v in np.unique(video)])),
        same_cap_oracle_OT_minus_Opm_mean_squared_error=float(oq.mean()),oracle_additional_capacity_rows=int((oq<-1e-12).sum()),
        per_fold={str(k):{name:metrics(pp[z['fold'][rows]==k],pf[z['fold'][rows]==k],native[z['fold'][rows]==k],y[z['fold'][rows]==k],video[z['fold'][rows]==k]) for name,pp in [('F',pf),('native',native),('cal_OT',pred[:,6]),('cal_Opm',pred[:,7])]} for k in range(3)},
        numeric_label_values_decoded=863,CAL_refit=False,DEV_read=False,TEST_read=False,limits=plan['limits'])
    (out/'receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:{m:d[m] for m in ['mse','mae','mean_risk_change_F','mean_risk_change_native']} for k,d in result.items()}))
if __name__=='__main__':main()
