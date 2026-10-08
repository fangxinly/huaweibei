"""Other-node public original byte audit and independent array metric recomputation."""
import argparse, csv, hashlib, os, subprocess, sys, urllib.request, zipfile
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score
from common import read, sha, utc, write

def independent(p,y):
    nonzero=y!=0
    return dict(Acc7=float(accuracy_score(np.round(np.clip(y,-3,3)),np.round(np.clip(p,-3,3)))),Acc2=float(accuracy_score(y[nonzero]>=0,p[nonzero]>=0)),F1=float(f1_score(y[nonzero]>=0,p[nonzero]>=0,average='weighted')),MAE=float(np.mean(np.abs(p-y))),Corr=float(pearsonr(p,y).statistic),MSE=float(np.mean((p-y)**2)))

def run(a):
    publication=read(a.publication);cap=read(a.capture)
    assert publication['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED' and cap['natural_exit']==0 and cap['ZIP_CRC_unique_all_members']
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid=='GPU-609b23d6-282d-8a2b-23f5-9433b824d212'
    rows=sorted([x for x in publication['assets'] if x['source_sha256']==cap['archive_SHA']],key=lambda x:x['source_offset'])
    assert rows and sum(x['bytes'] for x in rows)<=40_000_000
    a.out.mkdir(exist_ok=False);archive=a.out/'complete_small_original.zip'
    if a.received_archive:
        assert a.received_archive.stat().st_size==cap['archive_bytes'] and sha(a.received_archive)==cap['archive_SHA']
        os.link(a.received_archive,archive)
    else:
        with archive.open('xb') as target:
            for row in rows:
                assert target.tell()==row['source_offset'];h=hashlib.sha256();n=0
                with urllib.request.urlopen(row['url'],timeout=120) as f:
                    for block in iter(lambda:f.read(1024**2),b''):
                        target.write(block);h.update(block);n+=len(block)
                assert n==row['bytes'] and h.hexdigest()==row['sha256']
    assert archive.stat().st_size==cap['archive_bytes'] and sha(archive)==cap['archive_SHA']
    root=a.out/'original';root.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for n in z.namelist():assert (root/n).resolve().is_relative_to(root.resolve())
        for n,h in read_manifest(z).items():assert hashlib.sha256(z.read(n)).hexdigest()==h
        z.extractall(root)
    assert read(root/'natural_exit.json')['natural_exit']==0
    result=dict(status='B_PUBLIC_SMALL_ORIGINAL_SHA_CRC_UNIQUE_MEMBERS_PASSED',stage=a.stage,actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,UUID=uuid,source_SHA=sha(__file__),archive_SHA=cap['archive_SHA'],SHA_CRC_unique_member_SHA_passed=True,original_retained=str(root),CPU_encoder_forward=False)
    result.update(original_transport='verified_C_original_SFTP' if a.received_archive else 'public_Release_download',public_download=a.received_archive is None,Release_publication_receipt_verified=True)
    if a.received_archive:
        result['status']='B_RECEIVED_SMALL_ORIGINAL_SHA_CRC_UNIQUE_MEMBERS_PASSED'
    if a.stage=='prediction':
        r=read(root/'out/inference_result.json');pred=root/'out/fixed_official_VAL_TEST_prediction.npz';assert sha(pred)==r['prediction_SHA']
        assert r['scalar_VAL_TEST_true_labels_not_indexed'] and r['all_parameters_buffers_RNG_unchanged']
        with np.load(pred,allow_pickle=False) as z:
            for role,n in [('VAL',229),('TEST',685)]:
                ids=z[role+'_ids'];on=z[role+'_prediction'];off=z[role+'_p0'];assert len(ids)==n and len(set(ids.tolist()))==n and on.shape==off.shape==(n,) and np.isfinite(on).all() and np.isfinite(off).all()
                assert r['checks'][role]['dummy0vs7_and_replay_exact']
        result.update(prediction_SHA=r['prediction_SHA'],selected_state_SHA=r['selected_state_SHA'],prediction_ID_arrays_finite_and_dummy_receipts_passed=True,true_labels_not_consumed=True)
    else:
        r=read(root/'out/official_aligned_five_result.json');error=0.;tables={}
        assert a.prediction_original and sha(a.prediction_original)==r['prediction_SHA']
        exact=dict(np.load(a.prediction_original,allow_pickle=False));csv_rounding_maxerror=0.
        for role in ('VAL','TEST'):
            with (root/'out'/f'{role}_actual_predictions.csv').open(encoding='utf8',newline='') as f:data=list(csv.DictReader(f))
            ids=[row['row_id'] for row in data];assert len(ids)==len(set(ids))==r['roles'][role]['rows']
            y=np.array([float(row['y']) for row in data]);csv_on=np.array([float(row['new']) for row in data]);csv_off=np.array([float(row['messages_off']) for row in data])
            assert ids==exact[role+'_ids'].tolist()
            on=exact[role+'_prediction'].astype(float);off=exact[role+'_p0'].astype(float)
            assert np.array_equal(csv_on.astype(np.float32),exact[role+'_prediction']) and np.array_equal(csv_off.astype(np.float32),exact[role+'_p0'])
            csv_rounding_maxerror=max(csv_rounding_maxerror,float(np.max(abs(csv_on-on))),float(np.max(abs(csv_off-off))))
            table={'new':independent(on,y),'messages_off':independent(off,y),'groups':{g:independent(on[m],y[m]) for g,m in [('weak',abs(y)<=1),('strong',abs(y)>1)]}}
            for mode in ('new','messages_off'):
                error=max(error,max(abs(table[mode][k]-r['roles'][role][mode][k]) for k in table[mode]))
            for group in table['groups']:
                error=max(error,max(abs(table['groups'][group][k]-r['roles'][role]['groups'][group][k]) for k in table['groups'][group]))
            tables[role]=table
        assert error<1e-12
        result.update(status='B_RECEIVED_ORIGINAL_INDEPENDENT_VAL_TEST_ALL_FIVE_PASSED' if a.received_archive else 'B_PUBLIC_ORIGINAL_INDEPENDENT_VAL_TEST_ALL_FIVE_PASSED',prediction_SHA=r['prediction_SHA'],selected_state_SHA=r['selected_state_SHA'],independent_sklearn_scipy_maxerror=error,independent_results=tables,CSV_labels_only_no_training=True,predictions_from_frozen_original_NPZ=True,CSV_numeric_rounding_maxerror=csv_rounding_maxerror,CSV_roundtrip_float32_exact=True)
    write(a.out/'audit_result.json',result);print(result,flush=True)

def read_manifest(z):
    import json
    return json.loads(z.read('member_SHA.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prediction','score'],required=True)
    p.add_argument('--received-archive',type=Path)
    p.add_argument('--prediction-original',type=Path)
    for n in ('publication','capture','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
