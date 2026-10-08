"""Actual five-metric DEV audit. Reads only existing authorized DEV arrays."""
import argparse,ast,datetime,hashlib,io,json,shutil,zipfile
from pathlib import Path
import numpy as np
from sentiment_metrics_careflow_v1 import metrics, SEMANTICS

ROOT=Path(__file__).parent.parent
D=Path('D:/CodexBackups/selective_flow_20261003_1105')
OLD=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen')
AUTHOR=OLD/'outputs/own_flow_v6_followup/careflow_upstream_audit_20261004T2120Z/upstream/train_reflow_new.py'
SNAP=D/'finite_c2_completed_snapshots_20261005T1627Z'
REF=OLD/'outputs/repeat5_experiments/completed/careflow_seed128'
AUDIT=OLD/'outputs/repeat5_experiments/monitoring/careflow_five_seed_audit_20261002T204346Z.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
write=lambda p,a:Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def verify_semantics():
    # Neutral exclusion changes counts; zero prediction is nonnegative.
    a=metrics(np.array([-.2,.2,0,0]),np.array([.2,-.2,0,0]))
    assert a['Acc2']==0 and a['F1']==0 and a['Has0_Acc2']==.5 and a['nonzero_samples']==2
    assert metrics([0,0],[-1,1])['Acc2']==.5
    # Unequal class supports must produce weighted, not macro or positive-class F1.
    b=metrics([-1,-1,1,1],[-1,-1,-1,1])
    assert abs(b['F1']-(.75*.8+.25*(2/3)))<1e-15
    # Clip only classification, NumPy half-to-even; preserve regression overshoot.
    c=metrics([-4,4,-.5,.5,1.5,2.5],[-3,3,0,0,2,2])
    assert c['Acc7']==1 and c['MAE']==2/3
    assert metrics([1,1],[1,2])['Corr'] is None
    assert metrics([0,0],[0,0])['Acc2'] is None
    bad_cases=0
    for p,y in (([1],[1,2]),([np.nan],[1]),([1],[np.inf]),([],[])):
        try:metrics(p,y)
        except ValueError:bad_cases+=1
    assert bad_cases==4
    author=ast.parse(AUTHOR.read_text(encoding='utf-8'))
    nodes=[x for x in author.body if isinstance(x,ast.FunctionDef) and x.name=='multiclass_acc']
    assert len(nodes)==1
    env={'np':np};exec(compile(ast.Module(nodes,type_ignores=[]),str(AUTHOR),'exec'),env)
    return env['multiclass_acc'], {'neutral_zero_weighted_F1_round_clip_undefined_and_bad_inputs_passed':True,'negative_cases':4,'Torch_imported':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stamp',required=True);args=parser.parse_args()
    out=ROOT/'work'/('five_metric_DEV_actual_'+args.stamp);out.mkdir(exist_ok=False)
    author_acc7,fixture=verify_semantics()
    original_audit=read(AUDIT);ref_entry=next(x for x in original_audit['verified_runs'] if x['seed']==128)
    assert all(ref_entry['checks'].values())
    # Freeze source, inputs, roles and all compared arms before decoding any task arrays.
    plan={'actual_frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source_sha256':sha(__file__),'metric_module_sha256':sha(Path(__file__).with_name('sentiment_metrics_careflow_v1.py')),
      'author_evaluator_source_sha256':sha(AUTHOR),'metric_semantics':SEMANTICS,
      'official_paper':'https://arxiv.org/html/2602.19140v1#S4.T1',
      'published_MOSI_reference_only_not_DEV_baseline':{'Acc7_percent':50.6,'Acc2_percent':89.8,'F1_percent':89.7,'MAE':.616,'Corr':.858},
      'published_MOSEI_reference_only_not_this_run':{'Acc7_percent':55.7,'Acc2_percent':87.9,'F1_percent':88.0,'MAE':.504,'Corr':.799},
      'reference_predeclared_seed':128,'reference_files_pins':ref_entry['files_sha256'],
      'own_arm_sources':{a:{'zip':str(SNAP/a/'snapshot.zip'),'zip_sha256':sha(SNAP/a/'snapshot.zip')} for a in ('a','b','c')},
      'roles':'Existing official DEV229 only, selected and previously explored. Descriptive reaggregation, no new model inference, fit, seed choice, checkpoint/threshold/readout selection, TEST arrays, new generalization or stability claim.',
      'comparisons':'Three prespecified own final default-control predictions and one predeclared cached CaReFlow seed128 DEV; no matched seed/training-capacity claim.',
      'future_new_flow':'After original100 preservation and earliest INNER-selected full-state replay: same five metrics on fixed predictions; no separate best checkpoint per metric, no Test-guided optimization.',
      'all_five_rule':'Acc7,Acc2,F1,Corr strictly higher and MAE strictly lower on the same eligible evaluation rows/protocol, using unrounded values; partial wins recorded separately. Paper row cannot decide DEV superiority.',
      'new_TEST_decode_allowed':False}
    write(out/'frozen_metric_plan.json',plan)
    plan_sha=sha(out/'frozen_metric_plan.json')
    (out/'author_metric_source.py').write_bytes(AUTHOR.read_bytes())
    for name,digest in ref_entry['files_sha256'].items():assert sha(REF/name)==digest
    protocol,selection=read(REF/'protocol.json'),read(REF/'selection.json')
    assert sha(REF/'protocol.json')==selection['protocol_sha256'] and protocol['valid_samples']==229
    # NPZ zip member CRCs verified, but only valid_* members are decompressed.
    with zipfile.ZipFile(REF/'predictions.npz') as z:
        for n in ('valid_pred.npy','valid_y.npy'):
            raw=z.read(n)
            a=np.load(io.BytesIO(raw),allow_pickle=False)
            if n=='valid_pred.npy':ref_pred=a.copy()
            else:y=a.copy()
    assert y.shape==ref_pred.shape==(229,)
    rows={'CaReFlow_cached_seed128_DEV':metrics(ref_pred,y)}
    arrays={'DEV_y':y,'CaReFlow_seed128_prediction':ref_pred}
    provenance={'reference':{'prediction_archive_sha256':sha(REF/'predictions.npz'),'protocol_sha256':sha(REF/'protocol.json'),
      'original_remote_audit_sha256':sha(AUDIT),'original_selection':selection,'original_data_sha256':protocol['data_sha256'],
      'source_commit':protocol['commit'],'deviations':protocol['deviations'],'only_valid_members_decoded':True}}
    native_mae_differences={}
    for arm in ('a','b','c'):
        prefix='finite_c2/diagnostics_v4/' if arm=='c' else 'finite_diagnostics_v3/'
        with zipfile.ZipFile(SNAP/arm/'snapshot.zip') as z:
            data=z.read(prefix+'diagnostics.npz');receipt_raw=z.read(prefix+'diagnostics_receipt.json')
            receipt=json.loads(receipt_raw)
            assert hashlib.sha256(data).hexdigest()==receipt['diagnostics_sha256']
            assert receipt['tensor_sha_before']==receipt['tensor_sha_after'] and not receipt['test_requested'] and receipt['rows']==229
            with np.load(io.BytesIO(data),allow_pickle=False) as d:
                yy=d['valid_y'].copy();pred=d['prediction_default'].copy()
            assert np.array_equal(yy,y), 'DEV labels/order differ; cannot silently compare'
        (out/('original_'+arm+'_diagnostics_receipt.json')).write_bytes(receipt_raw)
        name='Own_'+('C2' if arm=='c' else arm.upper())+'_saved_default_DEV'
        rows[name]=metrics(pred,yy);arrays[name+'_prediction']=pred
        native_mae_differences[name]=abs(rows[name]['MAE']-float(np.mean(np.abs(pred-yy))))
        provenance[arm]={'original_array_sha256':hashlib.sha256(data).hexdigest(),'original_receipt_sha256':hashlib.sha256(receipt_raw).hexdigest(),
         'label_values_order_equal_reference':True,'pairing_scope':'Same declared DEV split and identical ordered labels, not proof of newly independent or same-seed training.'}
    independent_checks={}
    for name,row in rows.items():
        pred=ref_pred if name.startswith('CaReFlow') else arrays[name+'_prediction']
        assert row['Acc7']==author_acc7(np.clip(pred,-3,3),np.clip(y,-3,3))
        assert abs(row['Corr']-np.corrcoef(pred.astype(np.float64),y.astype(np.float64))[0,1])<1e-13
        mask=y!=0;truth=y[mask]>=0;binary=pred[mask]>=0
        total=0.
        for cls in (False,True):
            tp=np.count_nonzero((truth==cls)&(binary==cls));fp=np.count_nonzero((truth!=cls)&(binary==cls));fn=np.count_nonzero((truth==cls)&(binary!=cls))
            denom=2*tp+fp+fn
            total+=(2*tp/denom if denom else 0)*np.count_nonzero(truth==cls)/len(truth)
        assert abs(total-row['F1'])<1e-14
        independent_checks[name]={'author_Acc7_exact':True,'independent_F1_confusion_vs_one_vs_rest_error':abs(total-row['F1']),'Pearson_np_corrcoef_error':abs(row['Corr']-np.corrcoef(pred,y)[0,1])}
    np.savez_compressed(out/'DEV_five_metric_original_arrays.npz',**arrays)
    reference=rows['CaReFlow_cached_seed128_DEV'];diffs={}
    for name,row in rows.items():
        if name.startswith('Own'):
            signs={k:(reference[k]-row[k] if k=='MAE' else row[k]-reference[k]) for k in ('Acc7','Acc2','F1','MAE','Corr')}
            diffs[name]={'improvement_signed':signs,'strictly_better_each':{k:v>0 for k,v in signs.items()},'all_five_strictly_better':all(v>0 for v in signs.values()),'comparison_scope':'Existing selected DEV only; no paper Test/future-risk superiority claim'}
    result={'status':'ACTUAL_ARCHIVED_DEV_FIVE_METRICS_CPU_REAGGREGATION_COMPLETE','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'frozen_plan_sha256':plan_sha,'rows':rows,'versus_cached_CaReFlow_DEV':diffs,'provenance':provenance,'independent_checks':independent_checks,
      'metric_contract_fixtures':fixture,'float32_native_to_float64_MAE_abs_difference':native_mae_differences,
      'derived_original_arrays_sha256':sha(out/'DEV_five_metric_original_arrays.npz'),'new_model_inference':False,'TEST_arrays_decoded':False,
      'new_other_node_CPU_audit':False,'new_GPU_experiment':False,'current_new_flow100_measured':False,'scope':plan['roles']}
    write(out/'actual_five_metric_result.json',result)
    permanent=D/('five_metric_DEV_actual_'+args.stamp);permanent.mkdir(exist_ok=False)
    for p in [*out.iterdir(),Path(__file__),Path(__file__).with_name('sentiment_metrics_careflow_v1.py')]:shutil.copyfile(p,permanent/p.name)
    members={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in permanent.iterdir() if p.is_file()}
    write(permanent/'member_manifest.json',{'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'members':members,'fresh_C_D_space':{'C':shutil.disk_usage('C:/').free,'D':shutil.disk_usage('D:/').free}})
    with zipfile.ZipFile(permanent/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in members:z.write(permanent/n,n)
        z.write(permanent/'member_manifest.json','member_manifest.json')
    with zipfile.ZipFile(permanent/'records.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)+1
        for n,r in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==r['sha256']
    saved={'status':'ACTUAL_DEV_FIVE_METRICS_COMPLETE_ORIGINAL_FILES_D_SHA_CRC_MEMBER_PASSED','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'directory':str(permanent),'members':len(members)+1,'ZIP_sha256':sha(permanent/'records.zip'),'new_remote_capture_or_other_node_CPU':False}
    write(permanent/'preservation_receipt.json',saved)
    print(json.dumps({'rows':rows,'differences':diffs,'saved':saved},ensure_ascii=True))

if __name__=='__main__':main()
