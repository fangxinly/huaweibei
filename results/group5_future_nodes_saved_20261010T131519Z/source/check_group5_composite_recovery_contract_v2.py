"""Synthetic metadata counterexamples, with no scientific imports or arrays."""
import ast
import copy
import datetime as dt
import json
from pathlib import Path
import sys

from group5_release_transport_v1 import digest, write
from group5_composite_resume_contract_v2 import qualify, copy_completed_epoch_evidence
from group5_composite_epoch_resume_v2 import history_state

WORK = Path(__file__).resolve().parent
ROOT = WORK / ('group5_recovery_synthetic_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))


def fixture(name, method='old_fixed_A'):
    root = ROOT / name
    (root / 'out').mkdir(parents=True)
    epochs = 20 if method == 'anchored_message20' else 100
    source = {'prepared_composite_v2.py': 'synthetic-only-source-identity'}
    origin = dict(status='GROUP5_COMPOSITE_TRAIN_FROZEN', resume=None, source_SHA=source,
                  asset_SHA={'public': 'synthetic-only-asset'}, split_SHA='synthetic-split',
                  orders_SHA='synthetic-orders', updates=epochs*2, task_seed=128,
                  method=method, fold=0, old_task_weight_reuse=False,
                  new_once_token=str(root/'already_consumed_original_once'),
                  parent_checkpoint={'SHA':'synthetic-parent'}, cache_reference={'files_SHA':{'cache':'synthetic-cache'}})
    (root/'out/checkpoint.bin').write_bytes(b'SYNTHETIC METADATA FIXTURE ONLY; NOT A MODEL OR PT')
    checkpoint=dict(path=str(root/'out/checkpoint.bin'),SHA=digest(root/'out/checkpoint.bin'),bytes=(root/'out/checkpoint.bin').stat().st_size)
    write(root/'plan.json',origin)
    def spec(path):return dict(path=str(path),SHA=digest(path))
    with (root/'out/FIT_steps.jsonl').open('w',encoding='utf8') as stream:
        for update in range(1,7):stream.write(json.dumps(dict(updates=update))+'\n')
    history=[]
    for epoch in (1,2):
        name=f'INNER_epoch{epoch:03d}'
        path=root/'out'/(name+'.npz')
        path.write_bytes(b'SYNTHETIC NON-ARRAY BYTES '+str(epoch).encode())
        row=dict(epoch=epoch,updates=2*epoch,state_SHA='synthetic-state-'+str(epoch),prediction_SHA=digest(path))
        if method=='old_fixed_A':row.update(inner_MSE=1.,best_epoch=1,best_MSE=1.)
        history.append(row)
        write(root/'out'/(name+'_freeze.json'),dict(SHA=digest(path),state_SHA=row['state_SHA']))
    members=[]
    for path in sorted(root.rglob('*')):
        if path.is_file():members.append(dict(name=path.relative_to(root).as_posix(),sha256=digest(path),bytes=path.stat().st_size))
    write(root/'member_manifest.json',members)
    cpu=dict(status='INDEPENDENT_CPU_ORIGINAL_COMPOSITE_EPOCH_STATE_AUDIT_PASS',original_child=123,
             method=method,fold=0,origin_plan_SHA=digest(root/'plan.json'),checkpoint_SHA=checkpoint['SHA'],
             source_SHA=source,split_SHA='synthetic-split',outer_labels_decoded=False,inner_labels_decoded=False,
             completed_epochs=2,updates=4,parent_checkpoint_SHA='synthetic-parent',cache_SHA={'cache':'synthetic-cache'})
    write(root/'synthetic_cpu.json',cpu)
    write(root/'synthetic_restoration.json',dict(status='RELEASE_FULL_ORIGINAL_RESTORED_SHA_ZIP_VERIFIED',
          all_member_SHA_CRC_unique_exact_set_passed=True,whole_SHA='synthetic-original-archive',bytes=999,
          member_manifest_SHA=digest(root/'member_manifest.json')))
    write(root/'synthetic_exit.json',dict(child=123,natural_exit=1))
    write(root/'synthetic_cost.json',dict(origin_FIT_log_SHA=digest(root/'out/FIT_steps.jsonl'),logical_updates_restored=4,
          observed_extra_completed_updates=2,unlogged_inflight_update_possible=True))
    write(root/'synthetic_state_qualification.json',dict(status='ORIGINAL_COMPOSITE_COMPLETE_STATE_CPU_CUDA_RECOVERY_QUALIFICATION_PASS',
          method=method,fold=0,source_SHA=source,parent_checkpoint_SHA='synthetic-parent',cache_SHA={'cache':'synthetic-cache'},
          checks={key:True for key in ('current_model','selected_model','all_Adam','Python_NumPy_Torch_CUDA_RNG','FIT_orders_statistics','next_update_matches_continuous')}))
    plan=copy.deepcopy(origin)
    plan['new_once_token']=str(root/'new_recovery_once')
    plan['resume']=dict(complete_epoch_recovery_qualified=True,full_Adam_RNG_recovery_qualified=True,CUDA_recovery_qualified=True,
          origin_plan=spec(root/'plan.json'),CPU_audit=spec(root/'synthetic_cpu.json'),restoration=spec(root/'synthetic_restoration.json'),
          member_manifest=spec(root/'member_manifest.json'),process_exit=spec(root/'synthetic_exit.json'),
          cost_accounting=spec(root/'synthetic_cost.json'),state_recovery_qualification=spec(root/'synthetic_state_qualification.json'),
          checkpoint=checkpoint,original_child=123,original_archive_SHA='synthetic-original-archive',original_archive_bytes=999,
          checkpoint_member='out/checkpoint.bin',FIT_steps=spec(root/'out/FIT_steps.jsonl'),restored_updates=4,origin_root=str(root))
    return root,plan,history


def change_evidence(plan,key,edit):
    spec=plan['resume'][key];path=Path(spec['path']);obj=json.loads(path.read_bytes());edit(obj);write(path,obj);spec['SHA']=digest(path)


def main():
    ROOT.mkdir();results=[]
    for method in ('old_fixed_A','anchored_message20'):
        root,plan,history=fixture('positive_'+method,method)
        got=qualify(plan,method,0,'train')
        assert got['restored_updates']==4 and got['observed_extra_completed_updates']==2
        assert history_state(method,history,2)['completed_epochs']==2
        destination=root/'copied';destination.mkdir();copy_completed_epoch_evidence(plan['resume'],history,method,destination)
        assert (destination/'INNER_epoch001.npz').exists()==(method=='old_fixed_A')
        assert digest(destination/'recovery_origin/original_FIT_steps.jsonl')==plan['resume']['FIT_steps']['SHA']
        assert not Path(plan['new_once_token']).exists()
        results.append(dict(case='synthetic_metadata_accept_'+method,passed=True,model_qualification=False))
    cases=[
        ('absent_CUDA',lambda p:p['resume'].__setitem__('CUDA_recovery_qualified',False)),
        ('string_CUDA_flag',lambda p:p['resume'].__setitem__('CUDA_recovery_qualified','true')),
        ('wrong_fold',lambda p:p.__setitem__('fold',1)),
        ('wrong_method',lambda p:p.__setitem__('method','careflow')),
        ('same_once',lambda p:p.__setitem__('new_once_token',json.loads(Path(p['resume']['origin_plan']['path']).read_bytes())['new_once_token'])),
        ('changed_source',lambda p:p['source_SHA'].__setitem__('prepared_composite_v2.py','changed')),
        ('held_outer_exposed',lambda p:change_evidence(p,'CPU_audit',lambda d:d.__setitem__('outer_labels_decoded',True))),
        ('fake_CPU_status',lambda p:change_evidence(p,'CPU_audit',lambda d:d.__setitem__('status','SYNTHETIC_PASS'))),
        ('unknown_child_exit',lambda p:change_evidence(p,'process_exit',lambda d:d.__setitem__('natural_exit',None))),
        ('boolean_exit',lambda p:change_evidence(p,'process_exit',lambda d:d.__setitem__('natural_exit',False))),
        ('wrong_child',lambda p:change_evidence(p,'process_exit',lambda d:d.__setitem__('child',456))),
        ('missing_whole_restore',lambda p:change_evidence(p,'restoration',lambda d:d.__setitem__('all_member_SHA_CRC_unique_exact_set_passed',False))),
        ('unbound_member_manifest',lambda p:change_evidence(p,'restoration',lambda d:d.__setitem__('member_manifest_SHA','unbound'))),
        ('final_epoch_restart',lambda p:change_evidence(p,'CPU_audit',lambda d:d.update(completed_epochs=100,updates=200))),
        ('zero_epoch',lambda p:change_evidence(p,'CPU_audit',lambda d:d.update(completed_epochs=0,updates=0))),
        ('partial_epoch_updates',lambda p:change_evidence(p,'CPU_audit',lambda d:d.__setitem__('updates',5))),
        ('hidden_replay_cost',lambda p:change_evidence(p,'cost_accounting',lambda d:d.__setitem__('observed_extra_completed_updates',0))),
        ('undisclosed_inflight',lambda p:change_evidence(p,'cost_accounting',lambda d:d.__setitem__('unlogged_inflight_update_possible',False))),
        ('missing_Adam_qualification',lambda p:change_evidence(p,'state_recovery_qualification',lambda d:d['checks'].__setitem__('all_Adam',False))),
        ('wrong_parent',lambda p:change_evidence(p,'CPU_audit',lambda d:d.__setitem__('parent_checkpoint_SHA','old-global-parent'))),
        ('changed_checkpoint',lambda p:Path(p['resume']['checkpoint']['path']).write_bytes(b'changed')),
        ('stale_evidence_SHA',lambda p:Path(p['resume']['cost_accounting']['path']).write_bytes(b'changed')),
        ('chain_recovery_not_qualified',lambda p:change_evidence(p,'origin_plan',lambda d:d.__setitem__('resume',{'restored_updates':2}))),
    ]
    for name,mutate in cases:
        root,plan,history=fixture(name);mutate(plan)
        try:qualify(plan,'old_fixed_A',0,'train')
        except (PermissionError,ValueError,KeyError):pass
        else:raise AssertionError('Unsafe synthetic recovery accepted: '+name)
        assert not Path(plan['new_once_token']).exists()
        results.append(dict(case=name,passed=True))
    root,plan,history=fixture('precheck_recovery')
    try:qualify(plan,'old_fixed_A',0,'precheck')
    except PermissionError:pass
    else:raise AssertionError('Precheck recovery accepted')
    results.append(dict(case='precheck_recovery_rejected',passed=True))
    assert qualify({'resume':None},'old_fixed_A',0,'train') is None
    assert qualify({'resume':False},'anchored_message20',0,'precheck') is None
    results.append(dict(case='fresh_path_has_no_recovery',passed=True))
    root,plan,history=fixture('copied_freeze_tamper')
    write(root/'out/INNER_epoch001_freeze.json',dict(SHA='wrong',state_SHA='wrong'))
    destination=root/'copy';destination.mkdir()
    try:copy_completed_epoch_evidence(plan['resume'],history,'old_fixed_A',destination)
    except PermissionError:pass
    else:raise AssertionError('Tampered freeze copied')
    results.append(dict(case='tampered_original_freeze_rejected',passed=True))
    # Compare unchanged science-bearing local functions, without importing them.
    before=ast.parse((WORK/'group5_composite_runtime_v1.py').read_text(encoding='utf-8-sig'))
    after=ast.parse((WORK/'group5_composite_runtime_v2.py').read_text(encoding='utf-8-sig'))
    def functions(tree):return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)}
    old,new=functions(before),functions(after)
    compared=('cpu','rng','rng_SHA','combined','fixed_state','forward','predict','step','save')
    assert all(old[name]==new[name] for name in compared)
    validate=next(node for node in after.body if isinstance(node,ast.FunctionDef) and node.name=='validate')
    qualify_line=min(node.lineno for node in ast.walk(validate) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='qualify')
    token_line=min(node.lineno for node in ast.walk(validate) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='open')
    assert qualify_line<token_line
    for name in ('group5_composite_runtime_v2.py','group5_composite_epoch_resume_v2.py','group5_composite_CPU_audit_v2.py','group5_composite_resume_contract_v2.py'):
        ast.parse((WORK/name).read_text(encoding='utf-8-sig'))
    assert not any(name in sys.modules for name in ('torch','numpy'))
    report=dict(status='COMPOSITE_V2_METADATA_SYNTHETIC_AND_AST_CHECKS_PASS',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                cases=results,passed_cases=len(results),unchanged_science_functions=list(compared),
                qualification_precedes_token_creation=True,scientific_imports=False,dataset_or_model_decode=False,
                model_Adam_RNG_or_CUDA_execution=False,real_partial_CPU_audit_implemented=False,
                real_recovery_qualified=False,fixture_root=str(ROOT),new_training_or_scoring_once_consumed=False)
    write(ROOT/'actual_synthetic_report.json',report)
    write(WORK/'group5_composite_recovery_v2_check_pointer.json',dict(root=str(ROOT),report=str(ROOT/'actual_synthetic_report.json'),SHA=digest(ROOT/'actual_synthetic_report.json')))
    print(json.dumps(dict(status=report['status'],passed_cases=len(results),AST_functions=len(compared),scientific_imports=False,report=str(ROOT/'actual_synthetic_report.json'))))


if __name__=='__main__':main()
