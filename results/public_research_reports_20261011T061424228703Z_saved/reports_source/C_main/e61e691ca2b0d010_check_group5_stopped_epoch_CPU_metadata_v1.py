"""Adversarial synthetic metadata for the prepared stopped-epoch CPU auditor."""
import ast
import copy
import datetime as dt
import json
from pathlib import Path
import sys

from group5_composite_stopped_epoch_CPU_audit_v1 import metadata
from group5_release_transport_v1 import digest,write

WORK=Path(__file__).resolve().parent


def fixture(method):
    row_ids={'fit':['v0:0','v0:1','v0:2'],'inner':['v1:0'],'outer':['v2:0']}
    fold=dict(rows={'fit':3,'inner':1,'outer':1},row_ids=row_ids)
    plan=dict(status='GROUP5_COMPOSITE_TRAIN_FROZEN',execution_enabled=True,resume=None,method=method,fold=0,
              source_SHA={'auditor':'synthetic-only'},split_SHA='synthetic-split',old_task_weight_reuse=False,
              parent_checkpoint={'SHA':'synthetic-parent'},cache_reference={'files_SHA':{'cache':'synthetic-cache'}},
              updates=20 if method=='anchored_message20' else 100)
    construction=dict(method=method,fold=0,source_SHA=plan['source_SHA'],split_SHA=plan['split_SHA'],
              clean_initial_state_SHA='synthetic-clean',initial_rng_SHA='synthetic-rng',parent_checkpoint_SHA='synthetic-parent',
              parent_selected_state_SHA='synthetic-parent-tensors',cache_SHA={'cache':'synthetic-cache'},
              frozen_tail_or_teacher_SHA='synthetic-terminal',fit_ids=row_ids['fit'],inner_ids=row_ids['inner'],
              exact_plan_SHA='synthetic-plan-sha',exact_dispatch_plan_SHA='synthetic-plan-sha',historical_task_weights_used=False,
              outer_labels_decoded=False,public_pretraining_fresh_start=True)
    m=copy.deepcopy(construction);m['updates']=2;m['history']=[dict(epoch=i,updates=i,state_SHA='synthetic-state') for i in (1,2)]
    if method=='old_fixed_A':
        # The tie must retain epoch1; selecting epoch2 is a deliberate counterexample.
        for row in m['history']:row.update(inner_MSE=1.,best_epoch=1,best_MSE=1.)
    saved=dict(metadata=m,completed_epoch_recovery_only=True,selected_model=None if method=='anchored_message20' else {'synthetic':'not a tensor'},
               selected_inner_prediction=None if method=='anchored_message20' else ['synthetic-not-an-array'],
               guard_journal=[dict(role='fit',labels_read=True),dict(role='outer',labels_read=False)])
    return plan,construction,saved,fold


def main():
    results=[]
    for method in ('anchored_message20','old_fixed_A'):
        p,c,s,f=fixture(method);got=metadata(p,c,s,f,'synthetic-plan-sha')
        assert got['completed_epochs']==2 and got['updates']==2
        if method=='old_fixed_A':assert got['best_epoch']==1
        results.append(dict(case='synthetic_nonfinal_metadata_'+method,passed=True,real_model_qualification=False))
    cases=[
      ('not_frozen',lambda p,c,s,f:p.__setitem__('status','PREPARED')),
      ('not_enabled',lambda p,c,s,f:p.__setitem__('execution_enabled',False)),
      ('string_enabled',lambda p,c,s,f:p.__setitem__('execution_enabled','true')),
      ('resume_chain',lambda p,c,s,f:p.__setitem__('resume',{'restored_updates':1})),
      ('wrong_method',lambda p,c,s,f:p.__setitem__('method','careflow')),
      ('wrong_fold',lambda p,c,s,f:p.__setitem__('fold',1)),
      ('boolean_fold',lambda p,c,s,f:s['metadata'].__setitem__('fold',False)),
      ('missing_fresh_scope',lambda p,c,s,f:s['metadata'].__setitem__('public_pretraining_fresh_start',False)),
      ('wrong_dispatch_plan',lambda p,c,s,f:s['metadata'].__setitem__('exact_dispatch_plan_SHA','different')),
      ('historical_weight',lambda p,c,s,f:s['metadata'].__setitem__('historical_task_weights_used',True)),
      ('outer_exposed_flag',lambda p,c,s,f:s['metadata'].__setitem__('outer_labels_decoded',True)),
      ('changed_cache',lambda p,c,s,f:p['cache_reference'].__setitem__('files_SHA',{'cache':'different'})),
      ('different_parent',lambda p,c,s,f:p['parent_checkpoint'].__setitem__('SHA','historical-global-parent')),
      ('wrong_fit_ids',lambda p,c,s,f:f['row_ids'].__setitem__('fit',['different'])),
      ('wrong_full_budget',lambda p,c,s,f:p.__setitem__('updates',99)),
      ('no_durable_epoch',lambda p,c,s,f:s['metadata'].__setitem__('history',[])),
      ('mid_epoch_updates',lambda p,c,s,f:s['metadata'].__setitem__('updates',3)),
      ('boolean_epoch',lambda p,c,s,f:s['metadata']['history'][0].__setitem__('epoch',True)),
      ('NaN_inner',lambda p,c,s,f:s['metadata']['history'][0].__setitem__('inner_MSE',float('nan'))),
      ('negative_inner',lambda p,c,s,f:s['metadata']['history'][0].__setitem__('inner_MSE',-1.)),
      ('boolean_inner',lambda p,c,s,f:s['metadata']['history'][0].__setitem__('inner_MSE',True)),
      ('tie_selects_late',lambda p,c,s,f:s['metadata']['history'][1].__setitem__('best_epoch',2)),
      ('boolean_best_epoch',lambda p,c,s,f:s['metadata']['history'][0].__setitem__('best_epoch',True)),
      ('missing_atomic_marker',lambda p,c,s,f:s.__setitem__('completed_epoch_recovery_only',False)),
      ('missing_selected_model',lambda p,c,s,f:s.__setitem__('selected_model',None)),
      ('missing_selected_inner',lambda p,c,s,f:s.__setitem__('selected_inner_prediction',None)),
      ('outer_access_journal',lambda p,c,s,f:s['guard_journal'].append(dict(role='outer',labels_read=True))),
      ('string_access_journal',lambda p,c,s,f:s['guard_journal'].append(dict(role='fit',labels_read='false'))),
    ]
    for name,edit in cases:
        p,c,s,f=fixture('old_fixed_A');edit(p,c,s,f)
        try:metadata(p,c,s,f,'synthetic-plan-sha')
        except (PermissionError,ValueError):pass
        else:raise AssertionError('Unsafe synthetic metadata accepted: '+name)
        results.append(dict(case=name,passed=True))
    for name,edit in (
      ('message_inner_exposure',lambda s:s['guard_journal'].append(dict(role='inner',labels_read=True))),
      ('message_INNER_selection',lambda s:s['metadata']['history'][0].__setitem__('inner_MSE',1.)),
      ('message_selected_state',lambda s:s.__setitem__('selected_model',{'synthetic':'forbidden'})),
    ):
        p,c,s,f=fixture('anchored_message20');edit(s)
        try:metadata(p,c,s,f,'synthetic-plan-sha')
        except (PermissionError,ValueError):pass
        else:raise AssertionError('Unsafe fixed20 metadata accepted: '+name)
        results.append(dict(case=name,passed=True))
    p,c,s,f=fixture('old_fixed_A')
    s['metadata']['history']=[dict(epoch=i,updates=i,state_SHA='synthetic',inner_MSE=1.,best_epoch=1,best_MSE=1.) for i in range(1,101)]
    s['metadata']['updates']=100
    try:metadata(p,c,s,f,'synthetic-plan-sha')
    except ValueError:pass
    else:raise AssertionError('Completed original accepted as nonfinal recovery')
    results.append(dict(case='completed_stage_not_recovery',passed=True))
    source=WORK/'group5_composite_stopped_epoch_CPU_audit_v1.py';tree=ast.parse(source.read_text(encoding='utf8'))
    audit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit_stopped_epoch')
    imports=[n.lineno for n in ast.walk(audit) if isinstance(n,ast.Import) and any(v.name in ('torch','numpy') for v in n.names)]
    ram_guard=next(n.lineno for n in ast.walk(audit) if isinstance(n,ast.If) and 'available < 6 * 1024 ** 3'==ast.unparse(n.test))
    assert min(imports)>ram_guard
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not any(name in calls for name in ('backward','step','forward','predict','fit'))
    assert not any(name in sys.modules for name in ('torch','numpy'))
    output=WORK/('group5_stopped_epoch_CPU_metadata_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir()
    report=dict(status='STOPPED_EPOCH_CPU_AUDITOR_SOURCE_AND_SYNTHETIC_METADATA_PASS',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                passed_cases=len(results),cases=results,source_SHA=digest(source),original_partial_CPU_audit_source_implemented=True,
                dedicated_frozen_audit_driver_pending=True,real_original_checkpoint_audit_executed=False,
                real_model_Adam_RNG_CUDA_recovery_qualified=False,scientific_imports=False,task_array_or_target_decode=False,
                new_fit_inference_score=False,new_once_consumed=False,RAM_gate_precedes_scientific_import=True)
    write(output/'actual_synthetic_report.json',report)
    write(WORK/'group5_stopped_epoch_CPU_audit_preparation_pointer.json',dict(root=str(output),report=str(output/'actual_synthetic_report.json'),SHA=digest(output/'actual_synthetic_report.json')))
    print(json.dumps(dict(status=report['status'],passed_cases=len(results),scientific_imports=False,report=str(output/'actual_synthetic_report.json'))))


if __name__=='__main__':main()
