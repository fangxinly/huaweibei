"""Create separate prepared v2 sources; never modify frozen v1 originals."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Frozen source anchor is not unique: ' + before[:70])
    return text.replace(before, after, 1)


def output(name, text):
    path = ROOT / name
    with path.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(text)


def main():
    runtime = (ROOT / 'group5_composite_runtime_v1.py').read_text(encoding='utf-8-sig')
    runtime = replace_once(runtime, 'Prepared native runner for fresh same-fold message20 and old fixed-A100.',
                           'Prepared runner with separately gated complete-epoch composite recovery.')
    runtime = replace_once(runtime, "    if p.get('status')!=wanted or not p.get('execution_enabled') or p.get('resume'):\n        raise PermissionError('Fresh composite stage not separately qualified; resume requires a separate qualified runner')",
                           "    if p.get('status')!=wanted or not p.get('execution_enabled'):\n        raise PermissionError('Composite stage is not separately frozen for execution')")
    runtime = replace_once(runtime, "    token=Path(p['new_once_token']);token.parent.mkdir(parents=True,exist_ok=True)",
                           "    from group5_composite_resume_contract_v2 import qualify\n    recovery=qualify(p,a.method,a.fold,a.stage)  # must pass before token creation and scientific imports\n    token=Path(p['new_once_token']);token.parent.mkdir(parents=True,exist_ok=True)")
    runtime = replace_once(runtime, '    return p\n', '    return p,recovery\n')
    runtime = replace_once(runtime, '    p=validate(a)',
                           "    p,recovery=validate(a)\n    from group5_composite_resume_contract_v2 import copy_completed_epoch_evidence")
    runtime = replace_once(runtime, 'exact_plan_SHA=a.plan_sha,exact_dispatch_plan_SHA=a.plan_sha',
                           "exact_plan_SHA=recovery['origin_plan_SHA'] if recovery else a.plan_sha,exact_dispatch_plan_SHA=a.plan_sha")
    runtime = replace_once(runtime, "('method','fold','clean_initial_state_SHA','initial_rng_SHA','parent_checkpoint_SHA','cache_SHA')",
                           "('method','fold','source_SHA','split_SHA','clean_initial_state_SHA','initial_rng_SHA','parent_checkpoint_SHA','cache_SHA')")
    runtime = replace_once(runtime,
                           "    best=float('inf');best_epoch=0;best_state=None;best_p=None\n    for epoch in range(epochs):",
                           """    best=float('inf');best_epoch=0;best_state=None;best_p=None;start_epoch=0
    if recovery:
        from group5_composite_epoch_resume_v2 import restore
        original=torch.load(p['resume']['checkpoint']['path'],map_location='cpu')
        restored=restore(model,optimizers,tail if a.method=='anchored_message20' else None,session,
                         original,p['source_SHA'],recovery['origin_plan_SHA'],orders,ownership,parent_SHA,
                         receipt['cache_SHA'],receipt)
        if restored['updates']!=recovery['restored_updates'] or restored['completed_epochs']!=recovery['completed_epochs']:
            raise PermissionError('Original CPU evidence and loaded complete epoch disagree')
        history=restored['history'];updates=restored['updates'];start_epoch=restored['completed_epochs']
        best=restored['best_MSE'];best_epoch=restored['best_epoch']
        best_state=restored['selected_model'];best_p=restored['selected_inner_prediction']
        copy_completed_epoch_evidence(p['resume'],history,a.method,a.out)
        receipt['recovery']=dict(recovery,reconstruction_journal=restored['reconstruction_journal'],
                                 reconstruction_is_extra_cost=True,prior_token_not_reused=True)
        write(a.out/'construction.json',receipt);write(a.out/'history.json',history)
        del original,restored
    for epoch in range(start_epoch,epochs):""")
    output('group5_composite_runtime_v2.py', runtime)

    helper = (ROOT / 'group5_composite_epoch_resume_v1.py').read_text(encoding='utf-8-sig')
    helper = replace_once(helper, 'orders,owners,parent_SHA,cache_SHA):', 'orders,owners,parent_SHA,cache_SHA,expected):')
    helper = replace_once(helper, "    m=saved['metadata'];fold=session.fold;method=m['method']",
                          """    m=saved['metadata'];fold=session.fold;method=m['method']
    for key in ('method','fold','split_SHA','clean_initial_state_SHA','initial_rng_SHA',
                'parent_checkpoint_SHA','frozen_tail_or_teacher_SHA'):
        if m[key]!=expected[key]:raise PermissionError('Original composite construction differs: '+key)
    for original in (saved['parent_model'],saved['model'],saved.get('selected_model') or {}):
        if any(t.is_floating_point() and not torch.isfinite(t).all() for t in original.values()):
            raise ValueError('Nonfinite original composite tensor')""")
    helper = replace_once(helper, "    model.load_state_dict(saved['model'],strict=True)",
                          """    for opt,mapping in zip(saved['optimizers'],owners):
        ids=[i for group in opt['param_groups'] for i in group['params']]
        if len(ids)!=len(set(ids)) or set(map(str,ids))!=set(mapping) or set(opt['state'])!=set(ids):
            raise ValueError('Complete original Adam ownership/state differs')
        for index,entry in opt['state'].items():
            tensor=saved['model'][mapping[str(index)]]
            if set(entry)!={'step','exp_avg','exp_avg_sq'} or float(entry['step'])!=state['updates']:
                raise ValueError('Original complete Adam update count differs')
            for key in ('exp_avg','exp_avg_sq'):
                if entry[key].shape!=tensor.shape or not torch.isfinite(entry[key]).all():
                    raise ValueError('Original complete Adam moments differ')
    if set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=torch.cuda.device_count():
        raise PermissionError('Original complete CUDA RNG/device inventory differs')
    model.load_state_dict(saved['model'],strict=True)""")
    output('group5_composite_epoch_resume_v2.py', helper)

    auditor = (ROOT / 'group5_composite_CPU_audit_v1.py').read_text(encoding='utf-8-sig')
    auditor = replace_once(auditor, "    if json.loads((root/'wrapper_exit.json').read_bytes())['natural_exit']!=0:raise PermissionError('Original composite child did not exit0')",
                           """    exit_code=json.loads((root/'wrapper_exit.json').read_bytes())['natural_exit']
    if type(exit_code) is not int or exit_code!=0:raise PermissionError('Original composite child did not exit0')
    from group5_composite_resume_contract_v2 import qualify
    recovery=qualify(plan,r['method'],r['fold'],'train' if plan['status']=='GROUP5_COMPOSITE_TRAIN_FROZEN' else 'precheck')
    origin_plan_sha=recovery['origin_plan_SHA'] if recovery else plan_sha
    if r['exact_plan_SHA']!=origin_plan_sha or r['exact_dispatch_plan_SHA']!=plan_sha:
        raise PermissionError('Original composite origin/dispatch plan differs')""")
    auditor = replace_once(auditor, "    if m['exact_plan_SHA']!=plan_sha or m['source_SHA']!=plan['source_SHA'] or m['split_SHA']!=plan['split_SHA'] or m['updates']!=updates:",
                           "    if m['exact_plan_SHA']!=origin_plan_sha or m['exact_dispatch_plan_SHA']!=plan_sha or m['source_SHA']!=plan['source_SHA'] or m['split_SHA']!=plan['split_SHA'] or m['updates']!=updates:")
    auditor = replace_once(auditor, "int(entry['step'])!=updates", "float(entry['step'])!=updates")
    auditor = replace_once(auditor, "    split=root/'source/split.json'",
                           """    if recovery:
        if m.get('recovery',{}).get('restored_updates')!=recovery['restored_updates']:
            raise PermissionError('Loaded original recovery ancestry differs')
        for key,name in (('FIT_steps','original_FIT_steps.jsonl'),('cost_accounting','original_cost_accounting.json')):
            if digest(root/'out/recovery_origin'/name)!=plan['resume'][key]['SHA']:
                raise PermissionError('Original recovery replay-cost evidence differs')
    split=root/'source/split.json'""")
    output('group5_composite_CPU_audit_v2.py', auditor)
    print('COMPOSITE_V2_SEPARATE_SOURCE_PREPARATION_COMPLETE_NO_MODEL_EXECUTION')


if __name__ == '__main__':
    main()
