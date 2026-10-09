"""New, explicit local scoring protocol. Frozen remote stages are never modified."""
import argparse, csv, datetime, hashlib, importlib.util, json, math, os, pickle, platform, statistics, sys
from pathlib import Path
import numpy as np

KEYS = ('Acc7','Acc2','F1','MAE','Corr')
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def module(p):
    spec=importlib.util.spec_from_file_location('frozen_author_metrics',p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def independent(pred, labels):
    p=list(map(float,pred));y=list(map(float,labels));n=len(y)
    if n==0 or len(p)!=n or not all(math.isfinite(v) for v in p+y): raise ValueError('Invalid arrays')
    nz=[i for i,v in enumerate(y) if v!=0]; conf=[[0,0],[0,0]]
    for i in nz: conf[int(y[i]>=0)][int(p[i]>=0)]+=1
    f1=0.0
    for c in (0,1):
        support=sum(conf[c]);denom=support+sum(conf[j][c] for j in (0,1));f1+=support*(2*conf[c][c]/denom if denom else 0)
    pm=statistics.mean(p);ym=statistics.mean(y)
    pp=sum((v-pm)**2 for v in p);yy=sum((v-ym)**2 for v in y)
    return dict(Acc7=sum(round(min(3,max(-3,a)))==round(min(3,max(-3,b))) for a,b in zip(p,y))/n,
        Acc2=(conf[0][0]+conf[1][1])/len(nz) if nz else None,F1=f1/len(nz) if nz else None,
        MAE=sum(abs(a-b) for a,b in zip(p,y))/n,
        Corr=sum((a-pm)*(b-ym) for a,b in zip(p,y))/math.sqrt(pp*yy) if n>1 and pp>0 and yy>0 else None)
def synthetic(m):
    cases=[([-4,-2.5,-1.5,-.5,0,.5,1.5,2.5,4],[-3,-2,-1,0,0,1,2,3,3]),
        ([0,0,-.2,.8],[-1,0,1,2]),([1,1,1],[-1,0,1]),([1],[1]),([0,1],[0,0])]
    rng=np.random.RandomState(128)
    for n in (13,229,685):cases.append((rng.normal(size=n),rng.randint(-3,4,size=n)))
    maximum=0.0
    for p,y in cases:
        a=m.metrics(p,y);b=independent(p,y)
        for k in KEYS:
            if a[k] is None or b[k] is None:assert a[k] is b[k],k
            else: maximum=max(maximum,abs(a[k]-b[k]));assert abs(a[k]-b[k])<1e-12,(k,a[k],b[k])
    for p,y in [([],[]),([float('nan')],[1]),([1],[1,2])]:
        try:m.metrics(p,y)
        except ValueError:pass
        else:raise AssertionError('Bad synthetic arrays accepted')
    return dict(cases=len(cases),max_error=maximum,threshold=1e-12,real_candidate_predictions_not_scored=True)
def qualify(a):
    assert np.__version__=='1.26.4'
    old=read(a.old_plan);root=a.old_plan.parent;ev=a.evidence
    assert sha(a.metric_source)==old['source_sha256']['sentiment_metrics_careflow_v1.py']
    assert sha(a.old_plan)==a.old_plan_sha
    assert sha(a.dataset)==old['asset_sha256']['assets/mosi.pkl']
    assert sha(a.prediction)==old['prediction_SHA']
    for name in ('audit','inference','prediction_byte_audit'):
        assert read(ev/('A100_'+name+'_exit.json'))['natural_exit']==0,name
    audit=read(ev/'A100_audit_result.json');assert audit['status']=='CANDIDATE_FULL4000_ADAM_SELECTION_AND_CPU_TAIL_PASSED'
    peer=read(ev/'A100_prediction_byte_audit.json');assert peer['prediction_SHA']==old['prediction_SHA']
    assert peer['selected_state_SHA']==old['selected_state_SHA'];assert peer['true_labels_not_consumed']
    assert read(ev/'A100_prediction_publication_client_exit.json')['natural_exit']==0
    pub=read(ev/'A100_prediction_publication.json');assert pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED'
    assert any(v['sha256']==old['prediction_SHA'] for v in pub['assets'])
    obs=read(ev/'A100_lease_end_readonly_observation.json');assert obs['score_root_exists'] is False and obs['score_once_token_exists'] is False
    for role in ('VAL','TEST'):
        assert len(old['official_role_IDs'][role])=={'VAL':229,'TEST':685}[role]
    with a.dataset.open('rb') as f:data=pickle.load(f)
    with np.load(a.prediction,allow_pickle=False) as pred:
        for role,key in [('VAL','dev'),('TEST','test')]:
            ids=[v[2].decode() if isinstance(v[2],bytes) else v[2] for v in data[key]]
            assert ids==old['official_role_IDs'][role]==pred[role+'_ids'].tolist()
            assert pred[role+'_prediction'].shape==(len(ids),) and np.isfinite(pred[role+'_prediction']).all()
    m=module(a.metric_source);check=synthetic(m)
    provenance={}
    for k in ('DEV_source','TEST_source'):
        q=Path(old['CaReFlow_baseline_original_reference'][k]);h=sha(q)
        assert h==old['CaReFlow_baseline_original_reference'][k+'_SHA'];provenance[k]=dict(path=str(q),sha256=h)
    assert a.plan.exists() is False
    p=dict(status='NEW_LOCAL_OFFICIAL_SCORING_PROTOCOL_QUALIFIED',actual_UTC=utc(),execution_host=platform.node(),execution_OS=platform.platform(),
        new_protocol_not_remote_stage_relabel=True,remote_original_plan=str(a.old_plan),remote_original_plan_SHA=a.old_plan_sha,
        official_dataset=str(a.dataset),official_dataset_SHA=sha(a.dataset),prediction=str(a.prediction),prediction_SHA=sha(a.prediction),
        metric_source=str(a.metric_source),metric_source_SHA=sha(a.metric_source),runner_SHA=sha(__file__),python_executable=sys.executable,
        python_version=sys.version,numpy_version=np.__version__,runtime_scope='CPU array scoring only; no torch/model/inference',
        selected_state_SHA=old['selected_state_SHA'],selected_epoch=73,official_role_IDs=old['official_role_IDs'],fixed_CaReFlow_five=old['fixed_CaReFlow_five'],
        baseline_provenance=provenance,recovery_compute_disclosure=old['recovery_compute_disclosure'],synthetic_qualification=check,
        score_once_token=str(a.plan.parent/'candidate_A_selected73_official_score_once.token'),
        evidence={n:dict(path=str(ev/n),sha256=sha(ev/n)) for n in ['A100_audit_result.json','A100_audit_exit.json','A100_inference_result.json','A100_inference_exit.json','A100_prediction_byte_audit.json','A100_prediction_byte_audit_exit.json','A100_prediction_publication.json','A100_prediction_publication_client_exit.json','A100_lease_end_readonly_observation.json']},
        original_dataset_labels_not_changed=True,TEST_previously_seen=True,new_remote_execution=False,other_node_metric_recalculation_pending=True)
    write(a.plan,p);print(json.dumps(dict(status=p['status'],plan_SHA=sha(a.plan),synthetic=check)))
def verified(a):
    assert sha(a.plan)==a.plan_sha;p=read(a.plan)
    assert p['status']=='NEW_LOCAL_OFFICIAL_SCORING_PROTOCOL_QUALIFIED'
    assert p['runner_SHA']==sha(__file__);assert p['numpy_version']==np.__version__=='1.26.4';assert sys.executable==p['python_executable']
    for key in ('official_dataset','prediction','metric_source'):assert sha(p[key])==p[key+'_SHA'],key
    for v in p['evidence'].values():assert sha(v['path'])==v['sha256']
    return p
def score(a):
    p=verified(a);m=module(p['metric_source']);a.out.mkdir()
    fd=os.open(p['score_once_token'],os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,a.plan_sha.encode());os.close(fd)
    pred=dict(np.load(p['prediction'],allow_pickle=False))
    with Path(p['official_dataset']).open('rb') as f:data=pickle.load(f)
    roles={};label_arrays={}
    for role,key in [('VAL','dev'),('TEST','test')]:
        ids=[r[2].decode() if isinstance(r[2],bytes) else r[2] for r in data[key]]
        assert ids==p['official_role_IDs'][role]==pred[role+'_ids'].tolist()
        y=np.asarray([np.asarray(r[1]).reshape(-1)[0] for r in data[key]],np.float32).astype(float)
        new=m.metrics(pred[role+'_prediction'],y);off=m.metrics(pred[role+'_p0'],y);base=p['fixed_CaReFlow_five'][role]
        margins={k:new[k]-base[k] for k in KEYS}
        roles[role]=dict(rows=len(y),new=new,messages_off=off,careflow=base,new_minus_careflow=margins,
            all_five_strict_point_improvement=all(margins[k]>0 for k in ('Acc7','Acc2','F1','Corr')) and margins['MAE']<0)
        label_arrays[role+'_ids']=np.array(ids);label_arrays[role+'_y']=y
        with (a.out/(role+'_actual_predictions.csv')).open('w',encoding='utf8',newline='') as f:
            w=csv.writer(f);w.writerow(['row_id','video','y','new','messages_off'])
            for i,s in enumerate(ids):w.writerow([s,s.split('[')[0],y[i],pred[role+'_prediction'][i],pred[role+'_p0'][i]])
    np.savez(a.out/'actual_official_labels.npz',**label_arrays)
    write(a.out/'official_aligned_five_result.json',dict(status='OFFICIAL_CANDIDATE_A_LOCAL_AUTHOR_FIVE_ONCE_COMPLETE_OTHER_NODE_METRICS_PENDING',
        actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,roles=roles,candidate_mode='factorized_aux',selected_epoch=73,
        prediction_SHA=p['prediction_SHA'],selected_state_SHA=p['selected_state_SHA'],official_dataset_SHA=p['official_dataset_SHA'],
        plan_SHA=a.plan_sha,recovery_compute_disclosure=p['recovery_compute_disclosure'],baseline_source_reference=p['baseline_provenance'],
        author_metric_source_unchanged=True,numpy_version=np.__version__,metric_semantics=m.SEMANTICS,other_node_metric_recalculation_pending=True,
        historical_TEST_access_disclosed=True,not_a_new_blind_TEST=True,model_not_loaded=True,new_inference_or_training=False))
    print(json.dumps(dict(status='LOCAL_OFFICIAL_AUTHOR_FIVE_ONCE_SAVED',result_SHA=sha(a.out/'official_aligned_five_result.json'))))
def peer(a):
    p=verified(a);res=read(a.result);pred=dict(np.load(p['prediction'],allow_pickle=False))
    with Path(p['official_dataset']).open('rb') as f:data=pickle.load(f)
    maximum=0.0;roles={}
    for role,key in [('VAL','dev'),('TEST','test')]:
        ids=[r[2].decode() if isinstance(r[2],bytes) else r[2] for r in data[key]];assert ids==pred[role+'_ids'].tolist()==p['official_role_IDs'][role]
        # float32 cast exactly matches the prequalified author's dataset label conversion.
        y=np.asarray([np.asarray(r[1]).reshape(-1)[0] for r in data[key]],np.float32).astype(float)
        roles[role]=independent(pred[role+'_prediction'],y)
        for k in KEYS:delta=abs(roles[role][k]-res['roles'][role]['new'][k]);maximum=max(maximum,delta);assert delta<1e-12
    a.out.mkdir();write(a.out/'independent_same_host_metric_check.json',dict(status='INDEPENDENT_STDLIB_SAME_HOST_METRIC_CALCULATION_PASSED',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,
        max_error=maximum,roles=roles,result_SHA=sha(a.result),prediction_SHA=p['prediction_SHA'],official_dataset_SHA=p['official_dataset_SHA'],
        same_host_not_other_node=True,other_node_metric_recalculation_pending=True))
    print(json.dumps(dict(status='INDEPENDENT_SAME_HOST_CHECK_PASSED',max_error=maximum)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['qualify','score','peer'])
    for k in ('plan','old-plan','dataset','prediction','metric-source','evidence','out','result'):ap.add_argument('--'+k,type=Path)
    ap.add_argument('--old-plan-sha');ap.add_argument('--plan-sha');a=ap.parse_args();{'qualify':qualify,'score':score,'peer':peer}[a.stage](a)
