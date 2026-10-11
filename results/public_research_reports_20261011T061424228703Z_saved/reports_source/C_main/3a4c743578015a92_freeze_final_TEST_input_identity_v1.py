"""Separate input-only execution freeze. No TEST asset decoding or model use."""
import datetime,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
OLD=BASE/'work/paired_official_fulltrain_v2_20261007T011543Z'
D=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_final_TEST_input_identity_frozen_20261007T040758Z')
LOCAL=BASE/'work/paired_final_TEST_input_identity_frozen_20261007T040758Z'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    free={x:shutil.disk_usage(x+':/').free for x in ('C','D')}
    assert free['D']>=6*1024**3
    D.mkdir(exist_ok=False)
    names=['paired_final_TEST_identity_candidate_v1.py','paired_final_TEST_identity_natural_wrapper_v1.py',
           'paired_final_TEST_identity_capture_v1.py','test_paired_final_TEST_identity_candidate_v1.py']
    for name in names:shutil.copyfile(BASE/'work'/name,D/name)
    identity_src=OLD/'original_official_INPUT_ONLY_identity_receipt.json'
    shutil.copyfile(identity_src,D/'original_qualified_TRAIN_DEV_input_identity.json')
    identity=json.loads(identity_src.read_text(encoding='utf-8'))
    assert identity['official_train_dev_ID_identity_sha256']=='9462739ff690facc2c10eec2e72e79a02c1b7c0a6ee346e893716894ed690747'
    cmd=[sys.executable,str(D/'test_paired_final_TEST_identity_candidate_v1.py')]
    before=utc()
    with (D/'synthetic_identity.stdout.log').open('wb') as out,(D/'synthetic_identity.stderr.log').open('wb') as err:
        child=subprocess.Popen(cmd,cwd=D,stdout=out,stderr=err);pid=child.pid;code=child.wait()
    write(D/'actual_synthetic_identity_tests_receipt.json',{
        'actual_start_UTC':before,'actual_natural_exit_UTC':utc(),'child_pid':pid,'fullargv':cmd,
        'natural_exit_code':code,'fixture_TEST_only_original_TRAIN_DEV_IDs_from_prior_receipt':True,
        'no_actual_TEST_role_input_ID_or_label_access':True,'stdout_sha256':sha(D/'synthetic_identity.stdout.log'),
        'stderr_sha256':sha(D/'synthetic_identity.stderr.log')})
    if code!=0:raise RuntimeError('Synthetic identity gate tests failed; no execution freeze')
    state=json.loads((BASE/'outputs/正式双方fullTRAIN100整保存与CPU待接续.json').read_text(encoding='utf-8'))
    assert state['fullTRAIN100_preservation_current_two_models_complete'] is True
    original_D=Path(state['D']);parents={}
    for m in ('minimal_fixed_F','careflow'):
        parents[m]={}
        for key,filename,expected in (
            ('training_joint_file','actual_training_joint_after_history_serialization_repair.json','training_joint_SHA'),
            ('fresh_joint_file','actual_fresh_public_selected_saved_joint.json','fresh_complete_joint_SHA')):
            src=original_D/m/filename;name=m+'_'+filename
            assert sha(src)==state['methods'][m][expected];shutil.copyfile(src,D/name);parents[m][key]=name
    oldplan=json.loads((OLD/'paired_fulltrain_execution_plan.json').read_text(encoding='utf-8'))
    srcfiles=[p for p in D.iterdir() if p.is_file()]
    hashes={p.name:sha(p) for p in srcfiles}
    plan={
        'status':'OFFICIAL_TEST_IDENTITY_INPUT_ONLY_PROTOCOL_FROZEN','actual_local_freeze_UTC':utc(),
        'actualclock_naming_reference':'2026-10-07 04:07:58 UTC','directory_suffix_naming_only':True,
        'source_sha256':hashes,'asset_sha256':oldplan['asset_sha256'],
        'qualified_fixed_parent_joints':parents,'original_training_plan_sha256':state['training_plan_SHA'],
        'role_entry_keys':['train','dev','test'],'labels_enabled':False,'model_or_training_enabled':False,
        'TEST_prediction_or_score_enabled':False,'TEST_count_or_ID_before_actual_stage':None,
        'complete_original_test_order_and_all_rows_required':True,
        'allowed_node_UUID':{'A':'GPU-53696803-875e-eec8-2231-29db63579891',
                             'B':'GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f'},
        'allowed_hosts':{'A':'REDACTED_SERVER_HOST.invalid:53314','B':'REDACTED_SERVER_HOST.invalid:53332'},
        'conservative_lease_end_UTC':'2026-10-07T13:30:00+00:00',
        'human_lease_provenance_reference':oldplan['human_lease_provenance_reference'],
        'execution_budget_seconds':900,'saving_reserve_seconds':7200,
        'fresh_native_required':True,'remote_min_free_bytes':4*1024**3,'D_min_free_bytes':6*1024**3,
        'trusted_whole_pickle_other_role_bytes_materialized':True,'old_TEST_access_disclosed':True,
        'A_original_then_B_independent_CPU_input_identity_only':True,
        'immutable_selected_F89_C93_and_readout_batch128_not_changed':True,
        'final_prediction_or_true_label_scoring_require_separate_later_complete_freeze':True}
    planpath=D/'input_identity_plan.json';write(planpath,plan);ph=sha(planpath)
    # Validate all frozen source/parent gates locally without any data/model access.
    sys.path.insert(0,str(D));import paired_final_TEST_identity_candidate_v1 as g
    assert g.frozen_gate(D,ph)==plan
    files=[p for p in D.iterdir() if p.is_file()]
    manifest={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    write(D/'local_freeze_manifest.json',{'kind':'LOCAL_INPUT_ONLY_SOURCE_FREEZE_NOT_REMOTE_CAPTURE','members':manifest})
    archive=D/'upload.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(files)
        for n,v in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
    receipt={'status':'LOCAL_TEST_INPUT_ONLY_SOURCE_FREEZE_SHA_CRC_UNIQUE_PASSED_NO_ACTUAL_TEST_ACCESS',
             'actual_local_receipt_UTC':utc(),'root':str(D),'plan_sha256':ph,'upload_sha256':sha(archive),
             'source_and_parent_member_count':len(files),'synthetic_tests_natural0':True,
             'no_actual_TEST_entry_input_ID_or_label_access':True,'no_model_forward_or_remote_capture':True,
             'remote_execution_started':False,'fresh_local_space_before':free}
    write(D/'local_freeze_receipt.json',receipt)
    LOCAL.mkdir(exist_ok=False)
    for p in files:shutil.copyfile(p,LOCAL/p.name)
    shutil.copyfile(archive,LOCAL/'upload.zip')
    pointer=dict(receipt,local_bundle=str(LOCAL),remote_bundle='/data/coding/'+D.name,
                 node_A='NOT_STARTED',node_B='NOT_STARTED',capture_D_joint='NOT_STARTED',
                 overall_research_or_lease_complete=False)
    write(BASE/'outputs/正式双方TEST零标签身份源冻结接续.json',pointer)
    doc='''这份协议只冻结官方 TEST 输入角色/身份实际获取和异节点原 CPU 独立核验，不授权模型前向、预测或真实标签评分。原 TRAIN/DEV 身份 SHA 先匹配已经真实保存的旧零标签输入门，再取 TEST 原完整 ID/行序/输入布局与摘要。未猜填 TEST 行数、未读取实际 TEST entry/输入/ID/标签。

5个合成测试通过：毒标签对象从不被索引、原行序保持、feature变化改变摘要、重复/跨角色ID及已知TRAIN顺序变化拒绝、非有限/错位输入拒绝。这里只用原合格TRAIN/DEV ID和合成TEST fixture，不能冒实际TEST资格。

真正执行时只允许当前第二租期 A/B，新根/new source/fullargv/原自然exit和完整capsule。source/公共16原assets SHA、native UUID/空compute/fullargv/空间、900秒执行加至少2小时保存余量真实通过后才打开可信整pickle。整pickle可能物化其它角色字节，record[1]永不索引。历史旧runner一次TEST访问明示。

A原身份自然0后receipt-first完整ZIP SHA/CRC/唯一/D审核，才把原件送B做输入only CPU独立ID/order/rawinput摘要核验；不是CPU模型前向。B完成同样原capture与D审核。后续仍需按实际共同官方TEST ID另冻双方同固定best89/93直接输出、batch128保尾批、完整源runtime/保存/原CPU/一次共同五项评分协议。当前最终预测或标签门尚未启用，研究全面超过与租期强化保存未完成。
'''
    (BASE/'outputs/正式双方TEST零标签身份源冻结接续.md').write_text(doc+'\nD source freeze '+str(D)+'\nplan SHA '+ph+'\n',encoding='utf-8')
    status=BASE/'outputs/研究接续状态.md'
    head='最新本地源冻结 '+receipt['actual_local_receipt_UTC']+'：官方TEST零标签身份输入协议与完整source/原capture自然wrapper已D SHA/ZIPCRC/唯一，5个合成fixture测试自然0。未访问实际TEST entry/ID/输入/标签、无模型前向或远程执行；先读《正式双方TEST零标签身份源冻结接续.md/json》。A原身份及B原CPU独立审核、真实双capture/D联合待执行；不是最终预测/评分协议冻结。固定双方完整保存与DEV五项负结果不变，租期强化保存及整体目标未完成。\n\n'
    status.write_text(head+status.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(pointer,ensure_ascii=False))

if __name__=='__main__':main()
