"""Synthetic all25 fixture and denial gates; never reads original input values."""
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
design_root=Path(json.loads((HERE/'test_selected_group5_pointer.json').read_bytes())['root'])
source=design_root/'source/group_reference'
sys.path.insert(0,str(source))
from group5_release_transport_v1 import write,digest
from group5_pooled_score_v1 import METHODS,preflight,frozen_predictions,aggregate
from fold_contract import make_folds


def test():
    ids=[f'synthetic_video_{v:02d}[{i}]' for v in range(93) for i in range(24 if v<56 else 23)]
    split=dict(rows=2195,videos=93,canonical_row_ids=ids,folds=make_folds(ids))
    checks=[]
    with tempfile.TemporaryDirectory(prefix='g5_score_synthetic_',dir=HERE) as tmp:
        root=Path(tmp);entries=[]
        def evidence(name,obj):
            p=root/name;write(p,obj);return dict(path=str(p),SHA=digest(p))
        for m in METHODS:
            for f in range(5):
                spec=split['folds'][f];prefix=f'{m}_{f}';state='synthetic_state_'+prefix
                path=root/(prefix+'.npz');p=np.asarray([((ids.index(s)%7)-3)/2 for s in spec['row_ids']['outer']],dtype=np.float32)
                np.savez(path,row_ids=np.asarray(spec['row_ids']['outer']),prediction=p,model_state_sha256=np.asarray(state))
                stage=evidence(prefix+'_plan.json',dict(split_SHA='synthetic_split',source_SHA={'synthetic.py':'synthetic_source'}))
                updates=(20 if m==METHODS[0] else 100)*int(np.ceil(spec['rows']['fit']/32))
                r=dict(method=m,fold=f,historical_task_weights_used=False,public_pretraining_fresh_start=True,outer_labels_decoded=False,
                       status='COMPOSITE_COMPLETE_OUTER_UNSCORED_CPU_TRANSPORT_PENDING' if m in METHODS[:2] else 'DIRECT100_COMPLETE_OUTER_UNSCORED_CPU_TRANSPORT_PENDING',
                       checkpoint={'SHA':'synthetic_checkpoint_'+prefix},exact_plan_SHA=stage['SHA'],fit_ids=spec['row_ids']['fit'],inner_ids=spec['row_ids']['inner'],
                       updates=updates,outer_prediction_SHA=digest(path),selected_state_SHA=state,guard_journal=[dict(role='outer',labels_read=False)],parent_checkpoint_SHA='synthetic_parent_'+prefix)
                cpu=dict(method=m,fold=f,status='INDEPENDENT_CPU_ORIGINAL_COMPOSITE_STATE_AUDIT_PASS' if m in METHODS[:2] else 'INDEPENDENT_CPU_ORIGINAL_DIRECT_STATE_AUDIT_PASS',
                         original_native_exit=0,checkpoint_SHA=r['checkpoint']['SHA'],exact_plan_SHA=stage['SHA'],source_SHA={'synthetic.py':'synthetic_source'},updates=updates)
                e=dict(method=m,fold=f,receipt=evidence(prefix+'_receipt.json',r),CPU_audit=evidence(prefix+'_cpu.json',cpu),stage_plan=stage,
                       restoration=evidence(prefix+'_restore.json',dict(all_member_SHA_CRC_unique_exact_set_passed=True,whole_SHA='synthetic_archive_'+prefix,downloaded_bytes=123)),
                       original_archive_SHA='synthetic_archive_'+prefix,original_archive_bytes=123,
                       original_member_manifest=evidence(prefix+'_members.json',[dict(name='out/OUTER_prediction_only.npz',bytes=path.stat().st_size,sha256=digest(path))]),
                       prediction_member='out/OUTER_prediction_only.npz',prediction_path=str(path))
                if m in METHODS[:2]:
                    parent=dict(method='anchored_parent' if m==METHODS[0] else 'old_A_teacher',fold=f,historical_task_weights_used=False,public_pretraining_fresh_start=True,
                                fit_ids=spec['row_ids']['fit'],inner_ids=spec['row_ids']['inner'],outer_labels_decoded=False,GitHub_original_restoration_verified=True,
                                CPU_original_state_qualified=True,checkpoint={'SHA':r['parent_checkpoint_SHA']})
                    e['parent_qualification']=evidence(prefix+'_parent.json',parent)
                entries.append(e)
        plan=dict(split_SHA='synthetic_split');qualified=preflight(plan,entries,split);frozen=frozen_predictions(qualified,split)
        targets=np.asarray([((i%7)-3)/2 for i in range(2195)],dtype=np.float64)
        report=aggregate(frozen,targets,split)
        assert len(report['methods'])==5
        for r in report['methods'].values():
            assert r['concatenated_OOF']['MAE']==0 and r['concatenated_OOF']['Acc7']==1 and r['concatenated_OOF']['Corr']>1-1e-14
            assert all(v['sample_SD']<1e-14 for v in r['unweighted_fold_summary'].values())
        checks.append('all25 synthetic exact OOF coverage and fold mean/sample SD')
        def reject(name,fn):
            try:fn()
            except (ValueError,PermissionError,KeyError):checks.append(name)
            else:raise AssertionError('Gate accepted: '+name)
        reject('24 outputs forbidden',lambda:preflight(plan,entries[:-1],split))
        reject('duplicate method/fold forbidden',lambda:preflight(plan,entries[:-1]+[entries[0]],split))
        wrong=copy.deepcopy(entries);wrong[0]['original_archive_SHA']='other_archive'
        reject('wrong original restoration forbidden',lambda:preflight(plan,wrong,split))
        wrong=copy.deepcopy(entries);wrong[0]['receipt']['SHA']='0'*64
        reject('changed receipt bytes forbidden',lambda:preflight(plan,wrong,split))
        wrong=copy.deepcopy(entries);r=json.loads(Path(wrong[0]['receipt']['path']).read_bytes());r['outer_labels_decoded']=True
        wrong[0]['receipt']=evidence('bad_scope.json',r)
        reject('training outer target access forbidden',lambda:preflight(plan,wrong,split))
        wrong=copy.deepcopy(entries);p=json.loads(Path(wrong[0]['parent_qualification']['path']).read_bytes());p['fold']=4
        wrong[0]['parent_qualification']=evidence('bad_parent.json',p)
        reject('cross-fold trained parent forbidden',lambda:preflight(plan,wrong,split))
        wrong=copy.deepcopy(entries);p['fold']=0;p['CPU_original_state_qualified']=False
        wrong[0]['parent_qualification']=evidence('unaudited_parent.json',p)
        reject('unaudited parent forbidden',lambda:preflight(plan,wrong,split))
        shuffled=copy.deepcopy(frozen);ids0,p=shuffled[(METHODS[0],0)];shuffled[(METHODS[0],0)]=(ids0[:-1],p[:-1])
        reject('missing OOF row forbidden',lambda:aggregate(shuffled,targets,split))
        from sentiment_metrics_careflow_v1 import metrics
        r=metrics(np.array([0.,.5,-.5,3.5]),np.array([0.,1.,-1.,3.]))
        assert r['nonzero_samples']==3 and r['Acc2']==1 and r['MAE']==.375
        assert r['Acc7']==.5  # half-to-even rounding, clip only Acc7
        assert metrics(np.ones(4),np.arange(4))['Corr'] is None
        checks.append('neutral exclusion, weighted binary, half-even clipping and undefined correlation semantics')
    return dict(status='SYNTHETIC_SCORE_SOURCE_GATES_PASS',checks=checks,original_arrays_labels_or_models_loaded=False,formal_score_once_consumed=False)


if __name__=='__main__':print(json.dumps(test(),ensure_ascii=False))
