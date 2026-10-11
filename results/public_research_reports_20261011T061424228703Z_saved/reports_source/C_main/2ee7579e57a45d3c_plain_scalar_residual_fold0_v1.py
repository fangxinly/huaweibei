"""Frozen low-capacity, local CPU exploration; no teacher or message training."""
import argparse, ctypes, datetime as dt, hashlib, json, math, shutil, time, zipfile
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj): Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def weights(v):
    unique, count = np.unique(v, return_counts=True)
    return np.array([1.0/(len(unique)*count[np.where(unique==x)[0][0]]) for x in v])
def fit(p, y, v, ridge=.01):
    w=weights(v); r=y-p; mean=float(w@p); scale=float(np.sqrt(w@((p-mean)**2))); c=float(w@r)
    if scale<=1e-6: slope=0.0
    else:
        u=(p-mean)/scale; slope=float((w@(u*(r-c)))/(w@(u*u)+ridge))
    return dict(mean=mean, scale=scale, constant=c, intercept=c, slope=slope, degenerate=scale<=1e-6)
def predict(p, rule):
    affine=np.full(len(p),rule['intercept'])
    if not rule['degenerate']: affine=affine+rule['slope']*(p-rule['mean'])/rule['scale']
    return np.stack([p,p+rule['constant'],p+affine],axis=1)
def metrics(p, pred, y, v):
    result={}; names=['zero','constant','affine']
    for j,name in enumerate(names):
        e=pred[:,j]-y; q=e*e-(p-y)**2; h=pred[:,j]-p
        rows=[dict(video=str(g),rows=int(np.sum(v==g)),q=float(np.mean(q[v==g])),mse=float(np.mean(e[v==g]**2))) for g in np.unique(v)]
        qs=np.array([x['q'] for x in rows]); g=len(qs)
        deleted=[float(np.mean(np.delete(qs,k))) for k in range(g)] if g>1 else []
        result[name]=dict(pooled_mse=float(np.mean(e*e)),pooled_mae=float(np.mean(abs(e))),video_mse=float(np.mean([x['mse'] for x in rows])),video_q=float(qs.mean()),improved_videos=int(np.sum(qs<0)),videos=g,correction_rms=float(np.sqrt(np.mean(h*h))),residual_rms=float(np.sqrt(np.mean((y-p)**2))),residual_mean=float(np.mean(y-p)),per_video=rows,delete_one_video_q=deleted,stable_heuristic=bool(qs.mean()<0 and np.sum(qs<0)>=math.ceil(2*g/3) and deleted and max(deleted)<0))
    return result
def peak_memory():
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    c=Counters();c.cb=ctypes.sizeof(c)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
    ps=ctypes.WinDLL('psapi',use_last_error=True);ps.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    assert ps.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb)
    return int(c.PeakWorkingSetSize)

class LabelGuard:
    def __init__(self,path,fit_rows,inner_rows):
        self.path=path;self.allowed={'fit':set(map(int,fit_rows)), 'inner':set(map(int,inner_rows))};self.journal=[]
    def read(self,rows,role,purpose):
        rows=np.asarray(rows,dtype=np.int64)
        assert role in self.allowed and set(map(int,rows))<=self.allowed[role]
        with zipfile.ZipFile(self.path) as z, z.open('y.npy') as f:
            version=np.lib.format.read_magic(f)
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dtype=reader(f);offset=f.tell()
            assert shape==(1281,) and dtype==np.dtype('<f8') and not fortran
            out=[]
            for row in rows:
                f.seek(offset+int(row)*8);out.append(np.frombuffer(f.read(8),dtype=dtype)[0])
        self.journal.append(dict(utc=utc(),role=role,purpose=purpose,rows=rows.tolist()))
        return np.array(out,dtype=np.float64)

def selfcheck(out):
    p=np.array([-1.,0.,1.,2.]);v=np.array(['a','a','b','b']);y=p+.2+.3*p
    rule=fit(p,y,v);w=weights(v);u=(p-rule['mean'])/rule['scale'];x=np.stack([np.ones(4),u],axis=1)
    expected=np.linalg.solve(x.T@(w[:,None]*x)+np.diag([0.,.01]),x.T@(w*(y-p)))
    assert np.max(abs(expected-[rule['intercept'],rule['slope']]))<1e-12
    assert fit(np.ones(4),y,v)['degenerate']
    g=LabelGuard('never_open_this_path',[0],[1])
    rejected=0
    for rows,role in [([1],'fit'),([0],'inner'),([2],'outer')]:
        try:g.read(rows,role,'synthetic_forbidden')
        except AssertionError:rejected+=1
    assert rejected==3
    write(out,dict(passed=True,utc=utc(),synthetic_only=True,normal_equation_error=float(max(abs(expected-[rule['intercept'],rule['slope']]))),forbidden_label_role_requests_rejected=rejected,actual_labels_read=False))

def execute(plan_path,out):
    start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=False);plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    assert plan['fold']==0 and plan['ridge']==.01 and plan['B_predeclared'] is True
    for entry in plan['pins'].values():assert sha(entry['path'])==entry['sha256']
    assert sha(__file__)==plan['source_sha256'] and shutil.disk_usage('D:/').free>=plan['minimum_D_free_bytes']
    pins=plan['pins']; f=np.load(pins['fit_predictions']['path'],allow_pickle=False);n=np.load(pins['inner_predictions']['path'],allow_pickle=False)
    fr=f['row_ids'];nr=n['row_ids'];p=f['mu'].astype(np.float64);q=n['mu'].astype(np.float64)
    assert np.array_equal(fr,np.load(pins['fit_rows']['path'],allow_pickle=False)) and np.array_equal(nr,np.load(pins['inner_rows']['path'],allow_pickle=False))
    assert np.all(f['fold']==0) and np.all(n['fold']==0) and not set(fr)&set(nr)
    mapping=json.loads(Path(pins['video_mapping']['path']).read_text(encoding='utf-8'));assert [x['row'] for x in mapping]==list(range(1281))
    vids=np.array([x['video_id'] for x in mapping]);fv=vids[fr];nv=vids[nr]
    roles=json.loads(Path(pins['teacher_plan']['path']).read_text(encoding='utf-8'))['folds'][0]
    assert set(fv)==set(roles['fit_videos']) and set(nv)==set(roles['inner_videos']) and not set(fv)&set(nv)
    assert len(fr)==695 and len(nr)==153 and len(np.unique(nv))==4
    rec=json.loads(Path(pins['collection_receipt']['path']).read_text(encoding='utf-8'))
    assert rec['passed'] and rec['phase']=='execute' and rec['checkpoint_sha256']==plan['checkpoint_sha256']
    for key,entry in [('fit_scalar_inputs.npz',pins['fit_predictions']),('inner_scalar_inputs.npz',pins['inner_predictions'])]:assert rec['output_files'][key]['sha256']==entry['sha256']
    with np.load(pins['label_container']['path'],allow_pickle=False) as z:assert np.array_equal(z['row_ids'],np.arange(1281))
    guard=LabelGuard(pins['label_container']['path'],fr,nr)
    yf=guard.read(fr,'fit','A_fit_only');rule=fit(p,yf,fv,plan['ridge']);pred_A=predict(q,rule)
    np.savez(out/'A_frozen_predictions.npz',row_ids=nr,video=nv,p_F=q,pred=pred_A)
    write(out/'A_frozen_rule.json',rule)
    A_freeze=dict(utc=utc(),predictions_sha256=sha(out/'A_frozen_predictions.npz'),rule_sha256=sha(out/'A_frozen_rule.json'),INNER_labels_read_before_freeze=False)
    write(out/'A_prediction_freeze.json',A_freeze)
    yn=guard.read(nr,'inner','A_paired_evaluation_only_after_prediction_freeze')
    pred_B=np.empty((len(nr),3));rules_B={}
    for video in np.unique(nv):
        train=nv!=video;held=~train
        train_y=guard.read(nr[train],'inner','B_fit_excluding_video_'+str(video))
        rules_B[str(video)]=fit(q[train],train_y,nv[train],plan['ridge'])
        pred_B[held]=predict(q[held],rules_B[str(video)])
    np.savez(out/'B_frozen_predictions.npz',row_ids=nr,video=nv,p_F=q,pred=pred_B)
    write(out/'B_frozen_rules.json',rules_B)
    write(out/'B_prediction_freeze.json',dict(utc=utc(),predictions_sha256=sha(out/'B_frozen_predictions.npz'),rules_sha256=sha(out/'B_frozen_rules.json'),held_video_labels_excluded_from_each_head=True,all_INNER_labels_already_used_for_A_metrics_and_teacher_selection=True))
    np.savez(out/'scoped_inputs.npz',fit_rows=fr,fit_video=fv,fit_p=p,fit_y=yf,inner_rows=nr,inner_video=nv,inner_p=q,inner_y=yn)
    result=dict(A=metrics(q,pred_A,yn,nv),B=metrics(q,pred_B,yn,nv),fit=metrics(p,predict(p,rule),yf,fv),affine_vs_constant={})
    for arm,pred in [('A',pred_A),('B',pred_B)]:
        dif=(pred[:,2]-yn)**2-(pred[:,1]-yn)**2;vs=np.array([dif[nv==g].mean() for g in np.unique(nv)])
        result['affine_vs_constant'][arm]=dict(video_q=float(vs.mean()),improved_videos=int(sum(vs<0)),delete_one_q=[float(np.delete(vs,i).mean()) for i in range(4)],stable_heuristic=bool(vs.mean()<0 and sum(vs<0)>=3 and max(np.delete(vs,i).mean() for i in range(4))<0))
    write(out/'results.json',result);write(out/'label_access_journal.json',guard.journal)
    seconds=time.perf_counter()-start;peak=peak_memory();size=sum(x.stat().st_size for x in out.iterdir() if x.is_file())
    assert seconds<=plan['maximum_seconds'] and peak<=plan['maximum_peak_working_set_bytes'] and size<=plan['maximum_output_bytes']
    write(out/'receipt.json',dict(status='ACTUAL_LOCAL_CPU_SCALAR_EXPLORATION_COMPLETE',utc=utc(),seconds=seconds,peak_process_working_set_bytes=peak,output_bytes_before_receipt=size,plan_sha256=sha(plan_path),source_sha256=sha(__file__),fold=0,FIT_rows=695,INNER_rows=153,FIT_videos=30,INNER_videos=4,teacher_model_parameters_updated=False,new_GPU_forward=False,other_node_CPU_execution=False,inner_teacher_checkpoint_selection_reuse=True,whole_pipeline_crossfit=False,new_confirmation_set=False,outer_labels_used=False,CAL_EVAL_used=False,OOF_mu_used=False))
    print(json.dumps(dict(status='COMPLETE',A=result['A'],B=result['B'],seconds=seconds,peak=peak),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selfcheck');parser.add_argument('--plan');parser.add_argument('--out');a=parser.parse_args()
    if a.selfcheck:selfcheck(a.selfcheck)
    else:execute(a.plan,a.out)
