"""One locked T0 constant development measurement; no refit or fresh confirmation."""
import argparse,datetime as dt,hashlib,json,math,time,zipfile,shutil
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

class DevelopmentLabels:
    def __init__(self,path,allowed):self.path=Path(path);self.allowed=np.asarray(allowed);self.journal=[]
    def read_after_freeze(self,rows,predictions,freeze):
        rows=np.asarray(rows)
        if rows.dtype.kind not in 'iu' or not np.array_equal(rows,self.allowed):raise ValueError('ONLY_EXACT_LOCKED_OUTER_ROWS')
        f=json.loads(Path(freeze).read_text())
        if sha(predictions)!=f['predictions_sha256'] or f['labels_read_before_freeze'] is not False:raise ValueError('PREDICTION_SHA_BEFORE_LABELS')
        with zipfile.ZipFile(self.path) as z,z.open('y.npy') as stream:
            version=np.lib.format.read_magic(stream)
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            shape,fortran,dtype=reader(stream);offset=stream.tell()
            if shape!=(1281,) or fortran or dtype!=np.dtype('<f8'):raise ValueError('OFFICIAL_TRAIN_LABEL_CONTAINER_FORMAT')
            values=[]
            for row in rows:
                stream.seek(offset+int(row)*8);values.append(np.frombuffer(stream.read(8),dtype=dtype)[0])
        self.journal.append({'utc':utc(),'role':'T0_OUTER_DEVELOPMENT_ONLY_NOT_NEW_CONFIRMATION','rows':rows.tolist(),'purpose':'ONE_LOCKED_CONSTANT_PAIRED_MEASUREMENT_AFTER_PREDICTION_SHA','no_other_label_values_decoded':True})
        return np.asarray(values,dtype=np.float64)

def execute(plan_path,out):
    started=time.perf_counter();plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan['scope']!='ONE_LOCKED_T0_CONSTANT_OUTER_DEVELOPMENT' or not plan['no_refit_no_sweeps']:raise ValueError('LOCKED_SCOPE')
    if sha(__file__)!=plan['source_sha256']:raise ValueError('SOURCE_SHA')
    for e in plan['pins'].values():
        if sha(e['path'])!=e['sha256']:raise ValueError('PIN_SHA: '+e['path'])
    if shutil.disk_usage('D:/').free<plan['minimum_D_free_bytes']:raise ValueError('D_SPACE')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);pins=plan['pins']
    rows=np.load(pins['outer_rows']['path'],allow_pickle=False)
    fit=np.load(pins['fit_rows']['path'],allow_pickle=False);inner=np.load(pins['inner_rows']['path'],allow_pickle=False)
    if len(rows)!=433 or set(map(int,rows))&(set(map(int,fit))|set(map(int,inner))):raise ValueError('OUTER_ROW_ROLES')
    mapping=json.loads(Path(pins['mapping']['path']).read_text());allv=np.array([r['video_id'] for r in mapping]);v=allv[rows]
    if len(np.unique(v))!=18 or set(v)&(set(allv[fit])|set(allv[inner])):raise ValueError('OUTER_VIDEO_ROLES')
    completion=json.loads(Path(pins['teacher_completion']['path']).read_text())
    if completion['fold']!=0 or completion['completed_epochs']!=100 or completion['outer_labels_read'] is not False or completion['outer_predictions_sha256']!=pins['outer_predictions']['sha256']:raise ValueError('ORIGINAL_TEACHER_OUTER_PROVENANCE')
    c=float(json.loads(Path(pins['old_rule']['path']).read_text())['constant'])
    if c!=plan['locked_constant'] or completion['full_checkpoint_sha256']!=plan['checkpoint_sha256']:raise ValueError('REFERENCE_AND_CONSTANT_LOCK')
    with np.load(pins['outer_predictions']['path'],allow_pickle=False) as z:
        if set(z.files)!={'row_ids','mu'} or not np.array_equal(z['row_ids'],rows):raise ValueError('ORIGINAL_OUTER_ARRAY_ROWS')
        p=z['mu'].astype(np.float64)
    if p.shape!=(433,) or not np.isfinite(p).all():raise ValueError('FINITE_ORIGINAL_OUTER_PREDICTIONS')
    predictions=np.column_stack((p,p+c))
    np.savez(out/'locked_predictions.npz',row_ids=rows,video=v,p_F=p,predictions=predictions)
    write(out/'prediction_freeze.json',{'utc':utc(),'predictions_sha256':sha(out/'locked_predictions.npz'),'plan_sha256':sha(plan_path),'constant':c,'labels_read_before_freeze':False,'old_all_TRAIN_exploration_history_exists':True,'new_confirmation':False})
    guard=DevelopmentLabels(pins['labels']['path'],rows)
    y=guard.read_after_freeze(rows,out/'locked_predictions.npz',out/'prediction_freeze.json')
    if not np.isfinite(y).all():raise ValueError('FINITE_LABELS')
    errors=predictions-y[:,None];q=errors[:,1]**2-errors[:,0]**2
    videos=[]
    for video in np.unique(v):
        use=v==video
        videos.append({'video':str(video),'rows':int(use.sum()),'F_mse':float(np.mean(errors[use,0]**2)),'constant_mse':float(np.mean(errors[use,1]**2)),'Q':float(np.mean(q[use]))})
    qs=np.asarray([g['Q'] for g in videos]);fmse=np.mean([g['F_mse'] for g in videos]);cmse=np.mean([g['constant_mse'] for g in videos])
    deleted=[float(np.delete(qs,i).mean()) for i in range(18)]
    result={'scope':'T0_OUTER_LOCKED_CONSTANT_DEVELOPMENT','rows':433,'videos':18,'constant':c,
        'video_equal_F_mse':float(fmse),'video_equal_constant_mse':float(cmse),'video_equal_Q':float(qs.mean()),'relative_video_equal_mse_reduction_percent':float(100*(fmse-cmse)/fmse),
        'pooled_F_mse':float(np.mean(errors[:,0]**2)),'pooled_constant_mse':float(np.mean(errors[:,1]**2)),
        'pooled_F_mae':float(np.mean(abs(errors[:,0]))),'pooled_constant_mae':float(np.mean(abs(errors[:,1]))),
        'improved_videos':int(np.sum(qs<0)),'delete_one_video_Q':deleted,'per_video':videos,
        'predeclared_cost_heuristic_passed':bool(qs.mean()<0 and np.sum(qs<0)>=12 and max(deleted)<0),
        'decision_rule':'Heuristic passes: retain locked constant only for this T0 development baseline; fails: stop extending this scalar branch without changing fold/seed/sign/scale. Neither outcome authorizes Q heads/message100 or fresh-confirmation claims.',
        'new_confirmation':False,'whole_pipeline_crossfit':False,'new_teacher_or_message_parameters':False}
    np.savez(out/'scoped_development_values.npz',row_ids=rows,video=v,p_F=p,y=y,predictions=predictions,Q=q)
    write(out/'results.json',result);write(out/'label_access_journal.json',guard.journal)
    elapsed=time.perf_counter()-started
    if elapsed>plan['maximum_seconds']:raise ValueError('CPU_BUDGET')
    files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.iterdir() if p.is_file()}
    if sum(e['bytes'] for e in files.values())>plan['maximum_output_bytes']:raise ValueError('OUTPUT_BUDGET')
    write(out/'receipt.json',{'status':'ACTUAL_LOCKED_T0_CONSTANT_OUTER_DEVELOPMENT_COMPLETE','actual_utc':utc(),'elapsed_seconds':elapsed,'plan_sha256':sha(plan_path),'source_sha256':sha(__file__),'files':files,'label_role':'Only exact T0 OUTER433 values decoded after prediction freeze; old CAL/EVAL rows may overlap but their protocol/arrays unused. All TRAIN had past exploration.','new_GPU':False,'new_Torch':False,'new_fit':False,'other_node_CPU':False,'new_confirmation':False,'new_remote_capture':False})
    print(json.dumps({k:result[k] for k in ('scope','rows','videos','video_equal_F_mse','video_equal_constant_mse','video_equal_Q','relative_video_equal_mse_reduction_percent','improved_videos','predeclared_cost_heuristic_passed')},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();execute(a.plan,a.out)
