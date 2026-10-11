"""Freeze only the input-only identity operation; never authorizes a model/training/TEST."""
import argparse,ast,hashlib,json,shutil,zipfile
from pathlib import Path

BASE=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
BACKUP=Path('D:/CodexBackups/selective_flow_20261003_1105')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--clock-utc',required=True);a=p.parse_args()
    stamp=a.clock_utc[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
    root=BASE/'work'/('paired_input_identity_protocol_'+stamp);D=BACKUP/root.name
    if root.exists() or D.exists():raise RuntimeError('Fresh frozen identity bundle required')
    if min(shutil.disk_usage(BASE).free,shutil.disk_usage(BACKUP).free)<6*1024**3:raise RuntimeError('Fresh local C/D space floors')
    root.mkdir();D.mkdir()
    parent=BASE/'work/paired_fulltrain_preservation_sources_20261007T005109Z'
    names=['paired_fulltrain_evidence_candidate_v1.py','official_fulltrain_dev_guard_candidate.py',
           'paired_fulltrain_identity_input_only_candidate_v1.py','paired_fulltrain_auxiliary_natural_wrapper_candidate_v1.py',
           'paired_fulltrain_capture_candidate_v1.py','paired_fulltrain_capture_natural_wrapper_candidate_v1.py']
    for n in names:
        origin=BASE/'work'/n if n=='paired_fulltrain_identity_input_only_candidate_v1.py' else parent/n
        shutil.copy2(origin,root/n)
    sources={}
    for n in names:
        tree=ast.parse((root/n).read_text(encoding='utf-8'));compile(tree,n,'exec');sources[n]=sha(root/n)
    identity_tree=ast.parse((root/'paired_fulltrain_identity_input_only_candidate_v1.py').read_text())
    if any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value in (1,'test') for n in ast.walk(identity_tree)):
        raise RuntimeError('True label/TEST entry indexing in input-only collector')
    assets=json.loads((BASE/'work/minimal_fixed_staged_reference_v1_20261006T151117Z/runtime_candidate_plan.json').read_text(encoding='utf-8'))['asset_sha256']
    if assets.get('assets/mosi.pkl')!='5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b':raise RuntimeError('Original trusted public official data pin changed')
    plan={'status':'PAIRED_TRAIN_DEV_IDENTITY_INPUT_ONLY_PROTOCOL_FROZEN','clock_source_freeze_utc':a.clock_utc,
          'source_sha256':sources,'asset_sha256':assets,'assigned_gpu_UUID':'GPU-53696803-875e-eec8-2231-29db63579891',
          'labels_enabled':False,'model_or_training_enabled':False,'TEST_entry_enabled':False,
          'TRAIN_rows_expected':1281,'DEV_rows_expected':229,'CPU_only_metadata_operation_no_model_forward':True,
          'identity_execution_budget_seconds':600,'conservative_lease_end_UTC':'2026-10-07T13:30:00+00:00',
          'human_lease_provenance_reference':'Human second24h lease reported in this chat, approx Oct7UTC13:35; 13:30 is conservative and not platform confirmed',
          'trusted_whole_pickle_other_role_bytes_materialized_but_only_TRAIN_DEV_entries_indexed':True,
          'label_scale_not_verified_from_shape_or_ID':True,'fullTRAIN_model_execution_protocol_still_not_frozen':True,
          'no_old_completed_experiment_rerun_or_source_change':True}
    file=root/'paired_TRAIN_DEV_input_identity_plan.json';write(file,plan);plan_sha=sha(file)
    remote_bundle='/data/coding/paired_input_identity_protocol_'+stamp
    remote_root='/data/coding/paired_fulltrain_identity_'+stamp
    arguments=['--root',remote_root,'--bundle',remote_bundle,'--assets','/data/coding/multimodal_flow_public_20261006T1341Z','--plan-sha',plan_sha]
    write(root/'reserved_identity_child_arguments.json',arguments)
    with zipfile.ZipFile(root/'upload_bundle.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in [*names,file.name,'reserved_identity_child_arguments.json']:z.write(root/n,n)
    with zipfile.ZipFile(root/'upload_bundle.zip') as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise RuntimeError('Frozen identity upload bundle CRC/unique')
        for n in [*names,file.name,'reserved_identity_child_arguments.json']:
            if hashlib.sha256(z.read(n)).hexdigest()!=sha(root/n):raise RuntimeError('Frozen identity upload member SHA')
    receipt={'status':'INPUT_ONLY_IDENTITY_PROTOCOL_FROZEN_NOT_EXECUTED_NOT_FULLTRAIN_FREEZE','clock_source_freeze_utc':a.clock_utc,
             'local':str(root),'D':str(D),'plan_sha256':plan_sha,'bundle_zip_sha256':sha(root/'upload_bundle.zip'),
             'reserved_remote_bundle':remote_bundle,'reserved_remote_root_not_actual_launch':remote_root,
             'source_sha256':sources,'actual_remote_UUID_compute_argv_assets_space_lease_gates_pending':True,
             'official_input_identity_collected':False,'labels_model_training_TEST_executed':False,
             'parent_source_only_candidate':str(parent)}
    write(root/'local_input_identity_freeze_receipt.json',receipt)
    doc='''# 官方TRAIN/DEV零标签身份采集：小协议已冻结，未执行

本协议只允许实际公共资产全SHA核后采集TRAIN1281/DEV229的原ID与原输入shape/dtype。没有标签、模型、训练或TEST条目访问入口；whole可信pickle确实物化其它角色字节，不能称从未进内存。身份/shape不能证明标签尺度或正式五项结果。不得据此启动fullTRAIN。

六个确切源、公共资产SHA及独立natural/capture源已完整D保存；目录后缀是源冻结命名，远端root/argv仅预留，未上传或实际执行。执行前仍需fresh真实认证、A精确UUID/空compute/fullargv/source/assets/空间及执行预算加2h保存余量。无需新GPU计算或已完成实验重跑；完成后真实自然0→capture COMPLETE/natural0→原receipt与完整ZIP SHA/CRC/唯一成员D保存，才能给正式模型协议绑定行ID。

整体目标、正式benchmark与租期保存未完成；固定参考100、232候选和201一次评分不得重复执行，原加权仿射扩容仍暂停。
'''
    (root/'preparation.md').write_text(doc,encoding='utf-8')
    shutil.copytree(root,D/'source_protocol');shutil.copy2(__file__,D/Path(__file__).name)
    outputs=BASE/'outputs'
    for ext,n in [('md','preparation.md'),('json','local_input_identity_freeze_receipt.json')]:shutil.copy2(root/n,outputs/('正式官方TRAIN_DEV零标签身份协议接续.'+ext))
    pro=outputs/'正式CaReFlow比较来源与预算本地准备接续.json';v=json.loads(pro.read_text(encoding='utf-8'))
    v['latest_input_only_identity_protocol']={'local':str(root),'D':str(D),'status':receipt['status'],'plan_sha256':plan_sha,'clock_source_freeze_utc':a.clock_utc};write(pro,v)
    state=outputs/'研究接续状态.md';old=state.read_text(encoding='utf-8')
    state.write_text('actualclock '+a.clock_utc+'：只针对官方TRAIN/DEV零标签ID/input shape的独立小协议已冻结并D完整保存，未remote执行/采集/标签/模型。先读正式官方TRAIN_DEV零标签身份协议接续.md/json及来源预算接续最新指针；fresh真实认证/UUID/compute/fullargv/source/assets/空间和执行+2h门待核后按既有授权采集并实际capture→D。此冻结不授权fullTRAIN，完整benchmark门与整体目标/租期保存未完成。\n\n'+old,encoding='utf-8')
    control=D/'control';control.mkdir()
    for n in ['研究接续状态.md','正式CaReFlow比较来源与预算本地准备接续.json','正式官方TRAIN_DEV零标签身份协议接续.md','正式官方TRAIN_DEV零标签身份协议接续.json']:shutil.copy2(outputs/n,control/n)
    members={f.relative_to(D).as_posix():sha(f) for f in D.rglob('*') if f.is_file()};write(D/'member_SHA.json',members)
    with zipfile.ZipFile(D/'snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in [*members,'member_SHA.json']:z.write(D/n,n)
    with zipfile.ZipFile(D/'snapshot.zip') as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())) or len(z.namelist())!=len(members)+1:raise RuntimeError('D full identity source CRC/unique')
        for n,h in members.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('D full identity source member SHA')
    write(D/'preservation_receipt.json',{'clock_source_freeze_utc':a.clock_utc,'ZIP_SHA':sha(D/'snapshot.zip'),'members':len(members)+1,
         'CRC_unique_full_member_SHA_passed':True,'scope':'Only exact input-only source protocol and reserved argv; no remote capture or official data collected'})
    print(json.dumps({'local':str(root),'D':str(D),'plan_sha256':plan_sha,'status':receipt['status'],'members':len(members)+1},ensure_ascii=False))

if __name__=='__main__':main()
