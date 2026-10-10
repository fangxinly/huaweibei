"""Freeze new isolated direct/teacher runtime sources, preserve and publish."""
import ast,datetime as dt,json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from group5_release_transport_v1 import digest,seal,write,plan,upload
from raw_TRAIN_save_and_publish_v1 import publish

BASE=Path(__file__).resolve().parent.parent
DC=Path('D:/CodexBackups/selective_flow_20261003_1105/candidate_posttrain_lowC_20261009T005229Z')


def main():
    parent=Path(json.loads((BASE/'work/test_selected_group5_pointer.json').read_bytes())['root'])
    previous=Path(json.loads((DC/'D_current_research_state.json').read_bytes())['latest_human_TEST_selected_group5']['latest_direct_source']['root'])
    root=Path('D:/CodexBackups/selective_flow_20261003_1105')/('g5pipe_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir();sources=root/'source';sources.mkdir();count=0
    for method in ('careflow','anchored_parent','factorized_aux','regression_aux','old_A_teacher'):
        dst=sources/method;dst.mkdir()
        origin=previous/('anchored_parent' if method=='old_A_teacher' else method)
        for p in origin.iterdir():
            if p.is_file():shutil.copyfile(p,dst/p.name)
        for name in ('group5_fresh_direct_session_v1.py','group5_direct_runtime_v1.py','group5_teacher_optimizer_v1.py',
                     'group5_release_transport_v1.py','group5_test_selected_contract_v1.py'):
            shutil.copyfile(BASE/'work'/name,dst/name)
        if method=='old_A_teacher':
            for name in ('counterfactual_flow_model.py','encoder_adapter.py','legacy_flow_model.py'):
                shutil.copyfile(parent/'source/old_fixed_reference'/name,dst/name)
        for fold in range(5):shutil.copyfile(parent/'source/group_reference'/f'orders_fold{fold}.npy',dst/f'orders_fold{fold}.npy')
        for p in dst.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'));count+=1
    for name in ('group5_release_transport_v1.py','check_group5_transport_v1.py','group5_teacher_optimizer_origin.json',
                 'group5_teacher_optimizer_v1.py','assemble_group5_pipeline_v1.py'):
        shutil.copyfile(BASE/'work'/name,sources/name)
    q=Path('D:/CodexBackups/selective_flow_20261003_1105/g5transport_20261010T031621Z')
    transport=sources/'transport';transport.mkdir()
    for name in ('qualification.json','manifest.json','remote_upload_receipt.json','actual_remote_restoration.json','local_qualification.json'):
        shutil.copyfile(q/name,transport/name)
    (sources/'进度.md').write_text('本包为直接模型与旧A教师原生预检/100轮运行源码，尚无原生训练资格。GitHub合成5片实际下载全SHA/ZIP/成员校验通过；并非真实大检查点CPU资格。旧A优化器按原acc37dce源码AST提取，教师不加原本不存在的裁剪。两复合阶段缓存/训练及25结果统一评分仍待。全部旧原件/once不变。\n',encoding='utf8')
    write(sources/'source_receipt.json',dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),source_AST_files=count,
        original_design_SHA=digest(parent/'group5_design.json'),status='FIVE_DIRECT_TEACHER_RUNTIME_SOURCE_NOT_NATIVE_QUALIFIED',
        transport_synthetic_real_download_verified=True,actual_training_started=False,composite_pipeline_complete=False))
    proof=seal(sources,sources/'source.zip');write(root/'D_preservation_receipt.json',proof)
    os.environ['PATH']='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+os.environ['PATH']
    ranges=plan(Path(proof['archive']),'group5-direct-teacher-runtime-'+proof['archive_SHA'][:12]+'.zip')
    write(root/'range_manifest.json',ranges)
    uploaded=upload(ranges,root/'range_upload_receipt.json')
    assert len(uploaded['parts'])==1
    write(root/'Release_receipt.json',dict(remote_digest_verified=True,url=uploaded['parts'][0]['url'],whole_SHA=proof['archive_SHA'],range_receipt=uploaded))
    write(BASE/'work/group5_pipeline_pointer.json',dict(root=str(root),source_root=str(sources),proof=proof,Release=json.loads((root/'Release_receipt.json').read_bytes())))
    state=json.loads((DC/'D_current_research_state.json').read_bytes())
    state['latest_human_TEST_selected_group5']['latest_runtime_source']=dict(root=str(root),D_receipt=proof,Release=json.loads((root/'Release_receipt.json').read_bytes()),native_qualified=False,transport_synthetic_actual_download_passed=True)
    state['updated_at_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    for p in (DC/'D_current_research_state.json',BASE/'outputs/自主优化实际接续.json'):write(p,state)
    with (BASE/'outputs/研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+state['updated_at_utc']+' GitHub合成5片真实下载全SHA/ZIP/member还原过，五直接/教师新runtime源码Release永久保存，教师优化器原SHA/AST过；未原生资格、未训练，两复合阶段仍待。\n')
    print(json.dumps(dict(root=str(root),proof=proof,Release=json.loads((root/'Release_receipt.json').read_bytes()))))


if __name__=='__main__':main()
