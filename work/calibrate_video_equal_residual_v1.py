"""Frozen CAL-only residual signal fit. No model, EVAL decoding or threshold search."""
from pathlib import Path
import argparse,datetime,hashlib,json,time,zipfile,os,sys
import numpy as np

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def select_cal_label_values(path,requested,authorized,total_rows):
    requested=np.asarray(requested,dtype=np.int64);authorized=np.asarray(authorized,dtype=np.int64)
    if not np.array_equal(requested,authorized) or len(np.unique(requested))!=len(requested):
        raise PermissionError('Exactly frozen CAL rows only; reject EVAL/FIT/INNER and mixed requests')
    if np.any(requested<0) or np.any(requested>=total_rows):raise PermissionError('Invalid row')
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist())=={'row.npy','y.npy'}
        with z.open('y.npy') as f:
            version=np.lib.format.read_magic(f)
            if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
            elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
            else:raise ValueError('Unsupported NPY header')
            assert shape==(total_rows,) and not fortran and dtype.kind=='f' and not dtype.hasobject
            offset=f.tell();values=[]
            for row in requested:
                f.seek(offset+int(row)*dtype.itemsize)
                data=f.read(dtype.itemsize);assert len(data)==dtype.itemsize
                values.append(float(np.frombuffer(data,dtype=dtype,count=1)[0]))
    y=np.asarray(values,dtype=np.float64);assert np.isfinite(y).all()
    return y

def signal(p,mu,y,video):
    r=y-p;t=mu-p;videos=np.unique(video)
    per=[]
    for v in videos:
        s=video==v
        per.append(dict(video=str(v),rows=int(s.sum()),C=float(np.mean(r[s]*t[s])),B=float(np.mean(t[s]**2)),reference_mse=float(np.mean(r[s]**2)),teacher_mse=float(np.mean((mu[s]-y[s])**2))))
    C=float(np.mean([d['C'] for d in per]));B=float(np.mean([d['B'] for d in per]))
    lam=0. if B==0 else float(np.clip(C/B,0,1))
    loo=[]
    for v in videos:
        remain=[d for d in per if d['video']!=str(v)]
        c=float(np.mean([d['C'] for d in remain]));b=float(np.mean([d['B'] for d in remain]))
        loo.append(dict(removed_video=str(v),C=c,B=b,lambda_video_equal=0. if b==0 else float(np.clip(c/b,0,1))))
    rowC=float(np.mean(r*t));rowB=float(np.mean(t*t));rowlam=0. if rowB==0 else float(np.clip(rowC/rowB,0,1))
    return dict(C_video_equal=C,B_video_equal=B,lambda_video_equal=lam,C_row_equal=rowC,B_row_equal=rowB,lambda_row_equal_descriptive=rowlam,per_video=per,leave_one_video_out=loo,loo_lambda_range=[min(d['lambda_video_equal'] for d in loo),max(d['lambda_video_equal'] for d in loo)],loo_C_sign_flip=any((d['C']>0)!=(C>0) for d in loo),cal_fitted_surrogate_output_relative_risk=lam*lam*B-2*lam*C,rule='Global video-equal positive shrinkage only; not real-message performance or all possible nonlinear candidates.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['precheck','execute'],required=True);a=p.parse_args()
    start=time.monotonic();plan=read(a.root/'plan.json')
    assert plan['status']=='FROZEN_CAL_ONLY_VIDEO_EQUAL_RESIDUAL_SIGNAL'
    assert sha(__file__)==plan['source_sha256']
    for name,expected in plan['input_pins'].items():assert sha(a.root/name)==expected,name
    roles=np.load(a.root/'row_fold_calibration_roles.npz',allow_pickle=False)
    assert set(roles.files)=={'row_ids','fold','video','role'}
    rows=roles['row_ids'];fold=roles['fold'];video=roles['video'];role=roles['role']
    assert np.array_equal(rows,np.arange(1281)) and len(np.unique(video))==52
    geometry=np.load(a.root/'geometry_inputs.npz',allow_pickle=False)
    assert set(geometry.files)=={'row','fold','video','pf','mu','old_prediction'}
    assert np.array_equal(geometry['row'],rows) and np.array_equal(geometry['fold'],fold) and np.array_equal(geometry['video'],video)
    for k in range(3):
        cal=np.load(a.root/f'calibration_{k}.npy',allow_pickle=False)
        assert np.array_equal(cal,rows[(fold==k)&(role=='calibration')])
        assert len(np.unique(video[cal]))==6
        assert not set(video[cal])&set(video[(fold==k)&(role=='evaluation')])
    output=a.root/a.phase;output.mkdir(exist_ok=False)
    write(output/'runtime.json',dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),actual_python_argv=sys.argv,phase=a.phase))
    if a.phase=='precheck':
        toy=output/'synthetic_labels.npz';np.savez_compressed(toy,row=np.arange(8),y=np.arange(8,dtype=np.float64))
        allowed=np.asarray([1,3,6]);assert np.array_equal(select_cal_label_values(toy,allowed,allowed,8),allowed.astype(np.float64))
        denied=0
        for req in [np.asarray([0,3,6]),np.asarray([1,3,6,7]),np.asarray([1,3,3]),np.asarray([6,3,1])]:
            try:select_cal_label_values(toy,req,allowed,8)
            except PermissionError:denied+=1
        assert denied==4
        vids=np.repeat(np.arange(6).astype(str),[1,2,3,4,5,6]);t=np.ones(len(vids));zero=np.zeros(len(vids))
        positive=signal(zero,t,t*.125,vids);negative=signal(zero,t,-t,vids)
        assert positive['lambda_video_equal']==.125 and negative['lambda_video_equal']==0.
        assert positive['cal_fitted_surrogate_output_relative_risk']==-.125**2
        labels_decoded=0;pre=dict(status='CAL_SIGNAL_PRECHECK_PASSED',actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),plan_sha256=sha(a.root/'plan.json'),role_rows=418,role_videos=18,denied_wrong_role_requests=denied,real_label_values_decoded=labels_decoded,synthetic_positive_lambda=.125,synthetic_negative_lambda=0.,no_eval_statistics=True)
        write(output/'receipt.json',pre);print(json.dumps(pre));return
    gate=read(a.root/'precheck/receipt.json');assert gate['status']=='CAL_SIGNAL_PRECHECK_PASSED' and gate['real_label_values_decoded']==0
    assert gate['source_sha256']==sha(__file__) and gate['plan_sha256']==sha(a.root/'plan.json')
    with np.load(a.root/'metric_labels.npz',allow_pickle=False) as z:assert np.array_equal(z['row'],rows) # Never load z['y']; selective reader below.
    pf=geometry['pf'].astype(np.float64);mu=geometry['mu'].astype(np.float64);old=geometry['old_prediction'].astype(np.float64)
    fields={k:[] for k in ['row','fold','video','pf','mu','y','old_prediction']};result=[];access=[]
    for k in range(3):
        cal=np.load(a.root/f'calibration_{k}.npy',allow_pickle=False)
        y=select_cal_label_values(a.root/'metric_labels.npz',cal,cal,1281);access.extend(cal.tolist())
        res=signal(pf[cal],mu[cal],y,video[cal]);cap=float(np.quantile(np.abs(old[cal]-pf[cal]),.95,method='linear'))
        res.update(fold=k,cal_rows=len(cal),cal_videos=6,output_cap_old_native_abs_shift_quantile95=cap)
        result.append(res)
        for name,value in [('row',cal),('fold',fold[cal]),('video',video[cal]),('pf',pf[cal]),('mu',mu[cal]),('y',y),('old_prediction',old[cal])]:fields[name].append(value)
    assert len(access)==len(set(access))==418 and set(access)==set(rows[role=='calibration'].tolist())
    arrays=output/'calibration_inputs.npz';np.savez_compressed(arrays,**{k:np.concatenate(v) for k,v in fields.items()})
    calibration=dict(status='CAL_ONLY_VIDEO_EQUAL_PARAMETERS_FROZEN',actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),folds=result,source_sha256=sha(__file__),plan_sha256=sha(a.root/'plan.json'),inputs_sha256=sha(arrays),outcome_decoded_row_ids=sorted(access),outcome_decoded_rows=418,outcome_decoded_videos=18,eval_values_decoded=0,eval_metrics_computed=False,prior_all_train_exploration=True,whole_pipeline_crossfit=False,reference_full_train_fitted_dev_selected=True,message_control_executed=False,epsilon_fitted=False,output_cap_rule='Predeclared 95th percentile CAL original native control shift, label-free cap scale.',lambda_scope='Per own teacher fold positive global surrogate target shrinkage. C<=0 rules out this class only, not arbitrary nonlinear control.')
    write(output/'calibration_result.json',calibration)
    receipt=dict(status='CAL_SIGNAL_EXECUTE_COMPLETED',actual_utc=calibration['actual_utc'],source_sha256=sha(__file__),plan_sha256=sha(a.root/'plan.json'),result_sha256=sha(output/'calibration_result.json'),arrays_sha256=sha(arrays),seconds=time.monotonic()-start,rows=418,videos=18,real_eval_values_decoded=0,model_forward_or_training=False,archive_note='The previously explored original TRAIN label ZIP is byte-hashed and may be decompressed internally while seeking. Only authorized CAL numeric elements are decoded/returned; no EVAL numeric array or statistics is created.')
    write(output/'receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],folds=[{key:r[key] for key in ['fold','C_video_equal','B_video_equal','lambda_video_equal','loo_lambda_range','loo_C_sign_flip','output_cap_old_native_abs_shift_quantile95']} for r in result])))

if __name__=='__main__':main()
