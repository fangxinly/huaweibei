"""Independent covariance/confusion audit of a completed saved-array diagnostic."""
import argparse, ast, csv, datetime, hashlib, json, math, os, shutil, subprocess, sys, zipfile
from pathlib import Path
import numpy as np

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def mean(x): return math.fsum(float(v) for v in x)/len(x)
def covfit(x,y,w):
    mx=math.fsum(float(a*b) for a,b in zip(w,x));my=math.fsum(float(a*b) for a,b in zip(w,y))
    vx=math.fsum(float(a*(b-mx)**2) for a,b in zip(w,x))
    slope=math.fsum(float(a*(b-mx)*(c-my)) for a,b,c in zip(w,x,y))/vx
    intercept=my-slope*mx;q=slope*x+intercept;e=y-q
    vy=math.fsum(float(a*(b-my)**2) for a,b in zip(w,y));ve=math.fsum(float(a*b*b) for a,b in zip(w,e))
    return dict(slope=slope,intercept=intercept,R2=1-ve/vy,residual_rms=math.sqrt(ve)),q,e
def pearson(x,y):
    xc=x-mean(x);yc=y-mean(y)
    return math.fsum(float(a*b) for a,b in zip(xc,yc))/math.sqrt(math.fsum(float(a*a) for a in xc)*math.fsum(float(a*a) for a in yc))
def binary(p,y):
    c=[[sum(int(t>=0)==i and int(z>=0)==j for z,t in zip(p,y)) for j in range(2)] for i in range(2)]
    support=[sum(v) for v in c];pred=[c[0][j]+c[1][j] for j in range(2)]
    f1=math.fsum(support[i]*(2*c[i][i]/(support[i]+pred[i]) if support[i]+pred[i] else 0) for i in range(2))/len(y)
    return (c[0][0]+c[1][1])/len(y),f1,c
def metric(p,y):
    mask=y!=0;acc,f1,c=binary(p[mask],y[mask]);az,fz,cz=binary(p,y)
    return dict(Acc7=mean([round(max(-3.,min(3.,float(z))))==round(max(-3.,min(3.,float(t)))) for z,t in zip(p,y)]),
        Acc2=acc,F1=f1,MAE=mean(abs(p-y)),MSE=mean((p-y)**2),Corr=pearson(p,y),Has0_Acc2=az,Has0_F1=fz,
        samples=len(y),nonzero_samples=int(mask.sum()),zero_label_samples=int((~mask).sum()),exact_zero_predictions=int((p==0).sum()),
        correlation_defined=True,nonzero_confusion_matrix=c,haszero_confusion_matrix=cz)
def checkdict(got,expected):
    errors=[]
    for k,v in got.items():
        if isinstance(v,(int,bool,list)) or v is None: assert v==expected[k],(k,v,expected[k])
        else:
            error=abs(v-expected[k]);assert error<2e-12,(k,error);errors.append(error)
    return max(errors,default=0.)
def boot(q,p,y,video,draws,seed):
    videos=sorted(set(video));rng=np.random.default_rng(seed)
    # Counts of whole-video draws, rather than the first worker's indexed-sum method.
    selected=rng.integers(0,len(videos),size=(draws,len(videos)))
    multiplicity=np.zeros((draws,len(videos)),dtype=np.int64)
    for j in range(len(videos)):multiplicity[:,j]=(selected==j).sum(axis=1)
    result={}
    for region,mask in [('all',np.ones(len(y),bool)),('weak',abs(y)<=1),('strong',abs(y)>1)]:
        count=np.array([sum((video==v)&mask) for v in videos]);den=multiplicity@count;ok=den>0;out={}
        for name,d in [('MAE',abs(q-y)-abs(p-y)),('MSE',(q-y)**2-(p-y)**2)]:
            sums=np.array([math.fsum(float(t) for t in d[(video==v)&mask]) for v in videos])
            b=(multiplicity@sums)[ok]/den[ok]
            out[name]=dict(point=mean(d[mask]),percentile95=np.quantile(b,[.025,.975]).tolist(),valid_draws=int(ok.sum()))
        result[region]=out
    return result
def worker(a):
    assert sha(a.plan)==a.plan_sha;plan=read(a.plan);assert sha(__file__)==plan['worker_SHA']
    parent=Path(plan['parent_root']);pointer=read(plan['parent_pointer']);assert sha(parent/'actual/actual_result.json')==pointer['result_SHA']
    protocol=read(parent/'source/protocol.json');assert sha(parent/'source/protocol.json')==pointer['protocol_SHA'];original=read(parent/'actual/actual_result.json')
    errors=[];data={};supplement={};counts={};same_identity={}
    for model in ('best20','best36'):
        data[model]={};supplement[model]={};counts[model]={}
        for role in ('fit','inner'):
            name=f'{model}_{role}_y_b_p_video.csv';path=parent/'actual'/name;rr=original['models'][model]['roles'][role];assert sha(path)==rr['csv_SHA']
            with path.open(encoding='utf8',newline='') as f:records=list(csv.DictReader(f))
            cols=list(records[0]);x={k:np.array([r[k] for r in records]) if k in ('row_id','video') else np.array([float(r[k]) for r in records]) for k in cols}
            with np.load(parent/f'source/{model}_{role}_bfp.npz',allow_pickle=False) as z:
                assert x['row_id'].tolist()==z['row_ids'].astype(str).tolist()
                assert str(z['selected_state_sha256'])==protocol['selected_state_SHA'][model]
                for k in ('b','f','p'):assert np.array_equal(x[k],z[k].astype(float).reshape(-1))
            assert np.array_equal(x['y'],np.load(parent/f'source/{model}_{role}_targets.npy',allow_pickle=False).astype(float).reshape(-1))
            assert x['video'].tolist()==[r.split('[')[0] for r in x['row_id']];assert len(set(x['row_id']))==len(x['y'])
            if model=='best36':
                with np.load(parent/f'source/best36_{role}_donor_masks.npz',allow_pickle=False) as z:
                    assert np.array_equal(x['p0'],z['off'].astype(float)) and np.array_equal(x['p1'],z['on'].astype(float))
                assert np.array_equal(x['p1'],x['p'])
            data[model][role]=x;w=np.ones(len(records))/len(records);d=x['p']-x['b'];rho=x['y']-x['b'];st=rr['structure']
            for target,key in [(x['p'],'target_on_base'),(d,'delta_on_base'),(rho,'residual_on_base'),(x['f'],'flow_readout_on_base')]:
                errors.append(checkdict(covfit(x['b'],target,w)[0],st[key]))
            de=covfit(x['b'],d,w)[2];re=covfit(x['b'],rho,w)[2]
            errors.append(abs(pearson(de,re)-st['partial_corr_delta_rho_controlling_linear_base']));assert errors[-1]<2e-12
            preds={k:x[k] for k in ('b','p')}
            for scheme in ('rows','videos'):
                fit=data[model]['fit'];v=fit['video'];fv=sorted(set(v));fw=np.ones(len(v))/len(v) if scheme=='rows' else np.array([1/(len(fv)*sum(v==t)) for t in v])
                for k in ('b','p'):
                    t,q,e=covfit(fit[k],fit['y'],fw);expected=original['models'][model]['calibration'][scheme][k+'_to_y_FIT'];errors.append(checkdict(t,expected))
                    preds[k+'_calibrated_'+scheme]=t['slope']*x[k]+t['intercept']
            for region,mask in [('all',np.ones(len(records),bool)),('weak',abs(x['y'])<=1),('strong',abs(x['y'])>1)]:
                for key,p in preds.items():errors.append(checkdict(metric(p[mask],x['y'][mask]),rr['scores'][region][key]))
            if role=='inner':
                for scheme in ('rows','videos'):
                    qb=preds['b_calibrated_'+scheme];qp=preds['p_calibrated_'+scheme]
                    b=boot(qb,x['p'],x['y'],x['video'],protocol['bootstrap_draws'],protocol['bootstrap_seed'])
                    expected=rr['paired_video_bootstrap']['b_calibrated_'+scheme]
                    for region in b:
                        for k in b[region]:
                            assert abs(b[region][k]['point']-expected[region][k]['point'])<2e-12
                            assert np.max(abs(np.array(b[region][k]['percentile95'])-expected[region][k]['percentile95']))<2e-12
                    supplement[model][scheme+'_calibrated_p_minus_calibrated_b']=boot(qp,qb,x['y'],x['video'],protocol['bootstrap_draws'],protocol['bootstrap_seed'])
            counts[model][role]=dict(rows=len(records),videos=len(set(x['video'])),weak=int((abs(x['y'])<=1).sum()),strong=int((abs(x['y'])>1).sum()))
        assert not set(data[model]['fit']['video'])&set(data[model]['inner']['video'])
    for role in ('fit','inner'):
        same_identity[role]=all(np.array_equal(data['best20'][role][k],data['best36'][role][k]) for k in ('row_id','video','y'));assert same_identity[role]
    author=parent/'source/pinned_author_model_reflow_new.py';tree=ast.parse(author.read_text(encoding='utf8'));decoders={}
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='Sequential':
            for t in n.targets:
                if isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr in ('fusion','predictor'):
                    layers=[z.func.attr for z in n.value.args if isinstance(z,ast.Call) and isinstance(z.func,ast.Attribute)];assert layers==['Linear','ReLU','Linear'];decoders[t.attr]=dict(line=n.lineno,layers=layers)
    assert set(decoders)=={'fusion','predictor'}
    result=dict(status='INDEPENDENT_SAVED_ARRAY_COVARIANCE_CONFUSION_CSV_AUDIT_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,
        parent_result_SHA=pointer['result_SHA'],plan_SHA=a.plan_sha,all_source_member_sha256_verified=True,all_csv_values_and_IDs_exact=True,
        independent_covariance_OLS_metrics_bootstrap_passed=True,max_covariance_and_metric_error=max(errors),counts=counts,old_new_same_identity=same_identity,
        fixed_calibrated_p_minus_calibrated_b=supplement,decoder_AST=dict(source_SHA=sha(author),decoder_assignments=decoders,explicit_output_bounding_activation=False,
            limitation='Source excludes explicit final clipping. No internal activations inspected; effective activation saturation remains untested.'),
        no_new_model_forward=True,no_new_training=True,no_TEST_OUTER_files_opened=True,exploratory_selected_INNER=True)
    write(a.out,result);print(json.dumps(dict(status=result['status'],max_error=max(errors),supplement=supplement),ensure_ascii=False))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--clock');ap.add_argument('--worker',action='store_true');ap.add_argument('--plan',type=Path);ap.add_argument('--plan-sha');ap.add_argument('--out',type=Path);a=ap.parse_args()
    if a.worker:return worker(a)
    assert a.clock;pointer_path=Path('work/scale_shortcut_current.json').resolve();pointer=read(pointer_path);parent=Path(pointer['D'])
    stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z';root=parent.parent/('scale_shortcut_independent_audit_'+stamp);root.mkdir();source=root/'source';source.mkdir();script=source/Path(__file__).name;shutil.copy2(__file__,script)
    hashes=read(parent/'member_SHA.json')
    for name,h in hashes.items():assert sha(parent/name)==h,name
    with zipfile.ZipFile(parent/'complete_local_originals.zip') as z:
        assert sha(parent/'complete_local_originals.zip')==pointer['zip_SHA'];assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    plan=dict(actualclock_freeze_UTC=a.clock,parent_root=str(parent),parent_pointer=str(pointer_path),parent_result_SHA=pointer['result_SHA'],worker_SHA=sha(script),
        scope='Independent covariance OLS; Python round/confusion/f-sum metric audit; exact all CSV/ID/state checks; same fixed video bootstrap via draw multiplicities.',
        supplementary='Both already FIT-calibrated p minus FIT-calibrated b; 10000 whole-video paired draws seed20261008. No fit/selection change; all fixed comparisons reported.',
        no_new_model_forward=True,no_new_training=True,no_TEST_OUTER_files_opened=True)
    write(source/'protocol.json',plan);psha=sha(source/'protocol.json');cmd=[sys.executable,'-X','utf8',str(script.resolve()),'--worker','--plan',str((source/'protocol.json').resolve()),'--plan-sha',psha,'--out',str((root/'actual_independent_result.json').resolve())]
    write(root/'dispatch.json',dict(actual_UTC=utc(),fullargv=cmd))
    with (root/'stdout.log').open('wb') as out,(root/'stderr.log').open('wb') as err:
        child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));code=child.wait()
    write(root/'natural_exit.json',dict(pid=child.pid,fullargv=cmd,natural_exit=code,actual_UTC=utc(),plan_SHA=psha))
    if code:print((root/'stderr.log').read_text(encoding='utf8'));raise SystemExit(code)
    members={str(p.relative_to(root)).replace('\\','/'):sha(p) for p in root.rglob('*') if p.is_file()};write(root/'member_SHA.json',members);zp=root/'complete_local_originals.zip'
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
        for name in list(members)+['member_SHA.json']:z.write(root/name,name)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for name,h in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    audit=dict(actual_UTC=utc(),ZIP_SHA=sha(zp),CRC_pass=True,unique=True,all_member_SHA=True)
    write(root/'actual_local_archive_audit.json',audit);p=dict(status='INDEPENDENT_AUDIT_COMPLETE',D=str(root),result_SHA=sha(root/'actual_independent_result.json'),protocol_SHA=psha,ZIP_SHA=audit['ZIP_SHA'])
    write('work/scale_shortcut_independent_current.json',p);print(json.dumps(p));print((root/'stdout.log').read_text(encoding='utf8'))
if __name__=='__main__':main()
