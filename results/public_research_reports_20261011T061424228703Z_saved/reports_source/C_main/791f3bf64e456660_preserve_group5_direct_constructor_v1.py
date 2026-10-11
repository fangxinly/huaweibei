"""Assemble isolated method source bundles; AST and fail-closed checks only."""
import ast,datetime as dt,importlib.util,json,pathlib,shutil,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from raw_TRAIN_capture_v1 import raw,sha,seal
from raw_TRAIN_save_and_publish_v1 import publish

def main():
    parent=P(json.loads((BASE/'work/test_selected_group5_pointer.json').read_bytes())['root'])
    root=parent/('direct_source_actual_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir()
    ref=parent/'source/group_reference';latest=parent/'source/latest_AB_reference'
    entries={};count=0
    # Each process gets its own exact anchored_flow version. Incompatible parents
    # are not merged into one Python module namespace or shared task checkpoint.
    for method in ('careflow','anchored_parent','factorized_aux','regression_aux'):
        dst=root/method;dst.mkdir()
        for p in ref.glob('*.py'):
            if p.name in ['fold_runtime.py','fold_session.py','fold_score.py','fold_cpu_audit.py','fold_evidence.py','prepare.py','native_fixture_runner.py'] or p.name.startswith('test_'):continue
            shutil.copyfile(p,dst/p.name)
        if method in ('factorized_aux','regression_aux'):
            for p in latest.glob('*.py'):shutil.copyfile(p,dst/p.name)
        for name in ('group5_fresh_direct_session_v1.py','group5_test_selected_contract_v1.py'):
            shutil.copyfile(BASE/'work'/name,dst/name)
        shutil.copyfile(ref/'split.json',dst/'split.json')
        files={p.name:dict(SHA=sha(p),bytes=p.stat().st_size) for p in dst.iterdir() if p.is_file()}
        for p in dst.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'));count+=1
        expected=latest/'anchored_flow.py' if method in ('factorized_aux','regression_aux') else ref/'anchored_flow.py'
        assert sha(dst/'anchored_flow.py')==sha(expected)
        entries[method]=dict(source=files,exact_method_anchored_source_verified=True,execution_enabled=False,native_model_qualification=False)
    # A preparation plan must reject before pickle/scientific imports. These
    # fixtures are metadata only and do not claim CPU/model or GPU qualification.
    spec=importlib.util.spec_from_file_location('fresh_direct',BASE/'work/group5_fresh_direct_session_v1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    plan=root/'not_execution_plan.json';plan.write_bytes(raw(dict(status='SOURCE_PREPARATION_ONLY')))
    failures=0
    for method in entries:
        try:module.construct(method,root/method,root,plan,sha(plan),0)
        except PermissionError:failures+=1
        else:raise AssertionError('Preparation constructed a model')
    assert failures==4 and 'torch' not in sys.modules and 'numpy' not in sys.modules
    (root/'source_preparation_receipt.json').write_bytes(raw(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='FOUR_FRESH_DIRECT_CONSTRUCTORS_ASSEMBLED_AST_PASS_NOT_NATIVE_QUALIFIED',methods=entries,
        source_AST_files=count,unqualified_constructor_rejections=4,
        actual_new_fit_labels_pickle_native_model_or_SSH=False,
        unsupported_stages=['Fresh oldA teacher constructor/objective/cache','Same-fold selected-parent composite message20','OldA fixed addon composite training','Full-state fold training/resume/audit/scoring runtime'],
        prior_SSH19826_observation=dict(tool='write_stdin',error='Unknown process id 19826',remote_execution_confirmed=False,
                                      automatic_approval_rejection=False,permanently_do_not_reuse=True))))
    (root/'构造源码进度.md').write_text('四个直接/父模型构造器已组装，CaReFlow、Anchored父、新A、新B采用各自独立源码命名空间。源码AST与准备计划拒绝门通过；没有导入Torch或NumPy、没有加载pickle/标签/模型，也没有新SSH。旧A教师、两种复合增量训练以及完整训练/恢复/审核/评分仍需实现和原生资格。SSH19826实际Unknown process，永久不复用；这次不是新审批拒绝。\n',encoding='utf8')
    shutil.copyfile(P(__file__),root/P(__file__).name)
    proof=seal(root,'complete_actual_group5_direct_source.zip')
    (root/'D_preservation_receipt.json').write_bytes(raw(proof))
    publish(P(proof['archive']),proof['archive_SHA'],'group5-direct-source-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
    DC=P('D:/CodexBackups/selective_flow_20261003_1105/candidate_posttrain_lowC_20261009T005229Z')
    state=json.loads((DC/'D_current_research_state.json').read_bytes());state['latest_human_TEST_selected_group5']['latest_direct_source']=dict(root=str(root),D_receipt=proof,Release=json.loads((root/'Release_receipt.json').read_bytes()),native_qualified=False)
    state['latest_human_TEST_selected_group5']['retired_SSH_ids']=[19826]
    state['updated_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    for p in [DC/'D_current_research_state.json',BASE/'outputs/自主优化实际接续.json']:
        tmp=p.with_name(p.name+'.direct.tmp');tmp.write_bytes(raw(state));tmp.replace(p)
    assert (DC/'D_current_research_state.json').read_bytes()==(BASE/'outputs/自主优化实际接续.json').read_bytes()
    with (BASE/'outputs/研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+state['updated_at_utc']+' 四直接/父构造源码隔离组装、AST与不允许准备计划构造模型门实际过，完整新源Release digest过；非原生模型资格。旧SSH19826实际Unknown永久禁复用，未确认远端执行，非新审批拒绝。旧A教师/两复合阶段/训练恢复审核评分与原生资源门仍待；没有新训练，35阶段预算保持。\n')
    print(json.dumps(dict(root=str(root),proof=proof,source_AST_files=count,not_native_qualified=True)))

if __name__=='__main__':main()
