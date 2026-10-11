"""Record actual input-only original A/D evidence; never claims a model/other-CPU audit."""
import argparse,hashlib,json,shutil,sys,zipfile
from pathlib import Path
BASE=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
sys.path.insert(0,str(BASE/'work/paired_fulltrain_preservation_sources_20261007T005109Z'))
from paired_fulltrain_evidence_candidate_v1 import capture_association,stage_association,sha,read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--clock-utc',required=True);a=p.parse_args()
    D=Path('D:/CodexBackups/selective_flow_20261003_1105/official_TRAIN_DEV_input_identity_actual_20261007T005631Z');base=D/'a'
    if min(shutil.disk_usage(BASE).free,shutil.disk_usage(D).free)<6*1024**3:raise RuntimeError('Fresh actual local C/D space floor')
    c,m=capture_association(base);planfile=base/'source/paired_TRAIN_DEV_input_identity_plan.json';plan_sha=sha(planfile);plan=read(planfile)
    r,e=stage_association(base/'run',plan_sha)
    for n,h in plan['source_sha256'].items():
        if sha(base/'source'/n)!=h:raise RuntimeError('Actual identity original frozen source mismatch')
    identity=hashlib.sha256(json.dumps(r['ids'],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if identity!=r['official_train_dev_ID_identity_sha256'] or r['TRAIN_labels_read'] or r['DEV_labels_read'] or r['TEST_entry_indexed']:
        raise RuntimeError('Actual role/ID identity mismatch')
    for name,n in [('train',1281),('dev',229)]:
        if len(r['ids'][name])!=n or len(set(r['ids'][name]))!=n:raise RuntimeError('Actual official unique ID count mismatch')
    if set(r['ids']['train'])&set(r['ids']['dev']):raise RuntimeError('Actual official ID overlap')
    result={'status':'ACTUAL_OFFICIAL_TRAIN_DEV_INPUT_IDENTITY_D_CAPTURE_ASSOCIATION_PASSED_NO_LABELS_MODEL_OR_OTHER_CPU',
            'actual_local_audit_clock_utc':a.clock_utc,'D':str(D),'root':r['root'],'plan_sha256':plan_sha,
            'actual_child_launch':read(base/'run/actual_child_launch.json'),'actual_natural_exit':e,
            'actual_collection_utc':r['actual_utc'],'actual_capture_utc':c['actual_utc'],
            'original_receipt_sha256':sha(base/'run/out/actual_stage_receipt.json'),'original_exit_sha256':sha(base/'run/natural_exit.json'),
            'capture_receipt_sha256':sha(base/'capture_receipt.json'),'capture_exit_sha256':sha(base/'capture_actual_exit.json'),
            'ZIP_sha256':c['snapshot_sha256'],'ZIP_members':c['members'],'full_ZIP_SHA_CRC_unique_all_members_source_argv_association_passed':True,
            'official_train_dev_ID_identity_sha256':identity,'TRAIN_rows':1281,'DEV_rows':229,
            'input_layout_signatures':{k:len(r[k+'_input_layouts']) for k in ('train','dev')},
            'public_asset_sha256':r['original_public_asset_sha256'],'fresh_actual_A_preflight':read(D/'actual_A_preflight.json'),
            'original_preflight_sha256':sha(D/'actual_A_preflight.json'),
            'Torch_import_or_model_forward_or_training':False,'TRAIN_or_DEV_true_labels_accessed':False,'TEST_entry_indexed':False,
            'trusted_whole_pickle_other_role_bytes_materialized':True,'label_scale_not_proven_by_input_ID_shape':True,
            'original_other_CPU_audit_executed':False,'source_only_CPU_capture_joint_fresh_replay_candidates':str(BASE/'work/paired_fulltrain_preservation_sources_20261007T005109Z'),
            'new_five_metric_results_or_formal_benchmark_complete':False,'overall_research_and_lease_preservation_complete':False,
            'sessions_closed_actual0':{'A_SSH':35744,'A_SFTP':90136},'all_old_session_IDs_must_not_be_reused':True,
            'actual_C_free_bytes':shutil.disk_usage(BASE).free,'actual_D_free_bytes':shutil.disk_usage(D).free,
            'next':['Bind this physical official identity and asset provenance into new complete paired source protocol',
                    'Validate original target-scale source lineage without new unapproved role labels',
                    'Complete fresh-selected replay after100 D capture associator and full protocol/fresh execution+2h gates before precheck/training']}
    joint=D/'complete_input_identity_A_D_capture_association.json'
    if joint.exists():raise RuntimeError('Preserve original actual result immutable')
    write(joint,result)
    doc='''# 官方TRAIN/DEV零标签身份：实际A采集与D捕获关联

A新SSH/SFTP在真实password提示后认证；原UUID匹配、空compute、完整进程argv、已完成原任务natural0、空间/公共资产全SHA和保守租期执行加2h门通过，child2263实际UTC00:56:06.887101完成，wrapper自然exit0。新输入范围仅官方TRAIN1281/DEV229的ID和原输入shape/dtype；没有Torch/模型/训练/真标签或TEST条目索引。whole可信pickle确实物化其它角色字节，不冒从未进内存。

新身份SHA为9462739ff690facc2c10eec2e72e79a02c1b7c0a6ee346e893716894ed690747；TRAIN/DEV各自唯一且不重叠，输入布局签名264/129。这是实际行绑定与输入布局，没有标签尺度核验或新效果分数。

新根/data/coding/paired_fulltrain_identity_20261007T005246Z的原receipt、natural_exit、完整argv和确切源被实际捕获：UTC00:56:41.332076 COMPLETE，随后自然0。receipt/exit先下载后完整ZIP实际下载D，15成员SHA/CRC/唯一/源/argv/原件关联通过。ZIP SHA59bc76183ce219651cfa0f355a1c10f1d2e1f7368534d17fa6f43db855d0289b。目录后缀是命名，实际child/capture时刻由原receipt证明，未回填。

本轮没有异节点CPU模型或数组审核、没有新weight下载、没有新100或232/201任务。source-only TorchCPU/双capture/D joint及fresh实例重放候选见005109准备；它们还没有实际执行。A SSH35744/SFTP90136明确exit/bye实际0，禁止复用任何旧session ID。B/C没有本轮fresh核验，不能称三机实时。

下一将这份物理ID与公共资产来源绑定进完整双方协议，补标签尺度源链与fresh selected重放后D关联，核完整资源/保存门后再新预检/正式训练。完整fullTRAIN协议未冻，既有完成实验不得重跑，旧加权仿射扩容与201救分仍暂停。没有新五项成绩、最终TEST或超过CaReFlow结论，整体研究和租期保存仍未完成。
'''
    (D/'actual_result.md').write_text(doc,encoding='utf-8');shutil.copy2(__file__,D/Path(__file__).name)
    outputs=BASE/'outputs'
    shutil.copy2(joint,outputs/'正式官方TRAIN_DEV零标签身份实际结果.json');shutil.copy2(D/'actual_result.md',outputs/'正式官方TRAIN_DEV零标签身份实际结果.md')
    provenance=outputs/'正式CaReFlow比较来源与预算本地准备接续.json';v=read(provenance)
    v['latest_actual_OFFICIAL_TRAIN_DEV_input_identity']={'status':result['status'],'D':str(D),'joint':str(joint),'joint_sha256':sha(joint),
          'official_ID_identity_sha256':identity,'actual_collection_utc':r['actual_utc'],'actual_capture_utc':c['actual_utc']};write(provenance,v)
    short=outputs/'研究接续状态.md';previous=short.read_text(encoding='utf-8')
    short.write_text('actualclock '+a.clock_utc+'：官方TRAIN1281/DEV229的零标签原ID/input布局实际A child2263 UTC00:56:06.887101自然0；ID SHA9462739ff690facc2c10eec2e72e79a02c1b7c0a6ee346e893716894ed690747，原capture00:56:41 COMPLETE/natural0→15成员完整ZIP实际D SHA/CRC/唯一/源/argv关联通过。先补读正式官方TRAIN_DEV零标签身份实际结果.md/json及来源预算接续最新指针。无Torch/真标签/新模型或五项分数、无异节点CPU；A新SSH35744/SFTP90136明确exit/bye0，所有旧ID禁复用；B/C无本轮fresh。标签尺度/完整fullTRAIN源协议及fresh重放后D门未过，不重跑旧完成实验，整体目标/保存未完成。\n\n'+previous,encoding='utf-8')
    ledger=outputs/'研究建议交流接续.json';v=read(ledger)
    v['latest_input_identity_preparation_actual_not_advisor_batch']={'actual_local_audit_clock_utc':a.clock_utc,'D':str(D),'joint_sha256':sha(joint),
          'kind':'Actual input-only role identity/source binding, no labels/model/five-metric experiment','advisor_message_sent':False,
          'independent_reason':'Input-only provenance preparation is not a major new experiment; prior16 failed report unchanged, no duplicate same-batch retry.'};write(ledger,v)
    control=D/'control';control.mkdir()
    for n in ['研究接续状态.md','正式CaReFlow比较来源与预算本地准备接续.json','正式官方TRAIN_DEV零标签身份实际结果.md','正式官方TRAIN_DEV零标签身份实际结果.json','研究建议交流接续.json']:
        shutil.copy2(outputs/n,control/n)
    members={f.relative_to(D).as_posix():sha(f) for f in D.rglob('*') if f.is_file()};write(D/'member_SHA.json',members)
    archive=D/'local_full_identity_evidence.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n in [*members,'member_SHA.json']:z.write(D/n,n)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())) or len(z.namelist())!=len(members)+1:raise RuntimeError('Local full D evidence CRC/unique')
        for n,h in members.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('Local full D identity evidence member SHA')
    write(D/'local_preservation_receipt.json',{'actual_local_audit_clock_utc':a.clock_utc,'ZIP_SHA':sha(archive),'members':len(members)+1,
          'full_SHA_CRC_unique':True,'original_remote_capture_SHA':c['snapshot_sha256'],'original_joint_SHA':sha(joint),'scope':'Actual input-ID metadata/source/capture and local control; no original other CPU or new model score'})
    print(json.dumps({'joint':str(joint),'joint_SHA':sha(joint),'D_members':len(members)+1,'status':result['status']},ensure_ascii=False))

if __name__=='__main__':main()
