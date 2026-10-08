"""Save local source preparation and synthetic tests; no remote/data/model access."""
import datetime,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path

BASE=Path(__file__).resolve().parent.parent
D_PARENT=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_fulltrain100_complete_actual_20261007T030311Z')
ROOT=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_final_TEST_guard_preparation_20261007T040204Z')
CLOCK='2026-10-07 04:02:04 UTC'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def pinned(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}

def main():
    spaces={drive:shutil.disk_usage(drive+':/').free for drive in ('C','D')}
    if spaces['D']<6*1024**3:raise RuntimeError('D preparation preservation floor')
    ROOT.mkdir(exist_ok=False)
    copied={}
    for name in ('paired_final_TEST_guard_candidate_v1.py','test_paired_final_TEST_guard_candidate_v1.py',
                 'paired_final_TEST_protocol_preparation_after_train100.md',Path(__file__).name):
        src=BASE/'work'/name;dest=ROOT/name;shutil.copyfile(src,dest)
        assert sha(src)==sha(dest);copied[name]=sha(dest)
    parents={};cost={}
    current=BASE/'outputs/正式双方fullTRAIN100整保存与CPU待接续.json'
    state=json.loads(current.read_text(encoding='utf-8'))
    assert state['fullTRAIN100_preservation_current_two_models_complete'] is True
    assert state['all_five_strict_on_already_explored_DEV'] is False
    shutil.copyfile(current,ROOT/'original_current_complete_continuation.json')
    for method in ('minimal_fixed_F','careflow'):
        parents[method]={}
        for name,rel,key in (
            ('training_joint','actual_training_joint_after_history_serialization_repair.json','training_joint_SHA'),
            ('fresh_joint','actual_fresh_public_selected_saved_joint.json','fresh_complete_joint_SHA'),
            ('GPU_stage_receipt','a/run/out/actual_stage_receipt.json',None)):
            original=D_PARENT/method/rel
            if key: assert sha(original)==state['methods'][method][key]
            dest=ROOT/(method+'_'+name+'.json');shutil.copyfile(original,dest)
            assert sha(original)==sha(dest)
            parents[method][name]=dict(pinned(dest),original_path=str(original))
        receipt=json.loads((ROOT/(method+'_GPU_stage_receipt.json')).read_text(encoding='utf-8'))
        cost[method]=receipt['elapsed_seconds']
    fixed_summary=D_PARENT/'actual_both_fulltrain100_complete_CPU_fresh_fixed_DEV_summary.json'
    # Original seal's generic summary_SHA is the Markdown report, not JSON.
    prior_manifest=D_PARENT/'actual_both_fulltrain100_complete_local_seal_manifest.json'
    prior_receipt=D_PARENT/'actual_both_fulltrain100_complete_local_seal_receipt.json'
    pm=json.loads(prior_manifest.read_text(encoding='utf-8'))
    pr=json.loads(prior_receipt.read_text(encoding='utf-8'))
    json_sha='60f835f75562189a3f119fc68b001f20d9b30b9e6376f1a50ce0fe2a9a2ae741'
    md_sha='22bb7b437962c2c896a94f2eb768162b92a43b1f0787b35b601b24995030472d'
    assert sha(fixed_summary)==pm['members'][fixed_summary.name]['sha256']==json_sha
    md=fixed_summary.with_suffix('.md')
    assert sha(md)==pm['members'][md.name]['sha256']==pr['summary_SHA']==md_sha
    with zipfile.ZipFile(pr['ZIP']) as z:
        assert hashlib.sha256(z.read(fixed_summary.name)).hexdigest()==json_sha
        assert hashlib.sha256(z.read(md.name)).hexdigest()==md_sha
    for original in (prior_manifest,prior_receipt,md):shutil.copyfile(original,ROOT/('original_'+original.name))
    shutil.copyfile(fixed_summary,ROOT/'original_complete_fixed_DEV_summary.json')
    cmd=[sys.executable,str(ROOT/'test_paired_final_TEST_guard_candidate_v1.py')]
    before=utc()
    with (ROOT/'synthetic_tests.stdout.log').open('wb') as out,(ROOT/'synthetic_tests.stderr.log').open('wb') as err:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err)
        pid=child.pid;code=child.wait()
    after=utc()
    for name,h in copied.items():assert sha(ROOT/name)==h
    write(ROOT/'actual_local_synthetic_test_receipt.json',{
        'status':'ACTUAL_LOCAL_SYNTHETIC_GUARD_TESTS_NATURAL_EXIT',
        'actual_launch_UTC':before,'actual_natural_exit_UTC':after,'child_pid':pid,
        'fullargv':cmd,'natural_exit_code':code,'synthetic_only':True,
        'research_assets_or_TEST_entry_ID_label_accessed':False,'model_forward':False,
        'remote_capture_or_GPU_execution':False,'source_sha256':copied,
        'stdout':pinned(ROOT/'synthetic_tests.stdout.log'),'stderr':pinned(ROOT/'synthetic_tests.stderr.log')})
    if code!=0:raise RuntimeError('Synthetic guard tests failed; preserve failure, do not qualify')
    plan={
        'status':'LOCAL_FINAL_TEST_GUARD_PREPARATION_NOT_EXECUTION_FROZEN',
        'original_JSON_summary_sha256':json_sha,'original_Markdown_summary_sha256':md_sha,
        'original_seal_summary_SHA_is_Markdown_not_JSON':True,
        'prior_local_preparation_wrong_MD_hash_for_JSON_rejection_preserved':
            'D:/CodexBackups/selective_flow_20261003_1105/paired_final_TEST_guard_preparation_20261007T040023Z',
        'actual_local_record_UTC':utc(),'actualclock_naming_reference':CLOCK,
        'directory_suffix_is_naming_only':True,'source_sha256':copied,'qualified_parent_evidence':parents,
        'fixed_selected_methods':state['methods'],
        'training_elapsed_seconds':cost,'F_over_C_elapsed_ratio':cost['minimal_fixed_F']/cost['careflow'],
        'fixed_DEV_all_five_negative':True,'old_TEST_access_disclosed':True,
        'prospective_batch':128,'keep_tail':True,'readout':'same_already_selected_direct_output',
        'candidate_scope':'labels access and once-only gate; no dataset loader or model runner',
        'TEST_role_entry_ID_labels_accessed_in_preparation':False,
        'final_TEST_execution_enabled':False,'TEST_identity':None,'runtime_execution_protocol':None,
        'pending_before_any_TEST_input_acquisition':[
            'separate frozen input-only identity source/protocol and exact official pickle/asset sources',
            'fresh <=5min native allowed-host UUID/fullargv/compute/source/assets/space/budget with2h save',
            'genuine identity child natural0 and receipt-first full D capture audit'],
        'pending_before_prediction_execution':[
            'bind complete actual official TEST IDs/order/count; never invent count or read target for layout',
            'pin unchanged public construction and exact whole fixed selected restore/forward source',
            'freeze final inference protocol including dummy label/state/allRNG invariance and cumulative peaks',
            'new source wrapper/CPU audit/capture/complete preservation joint and once-only score sources'],
        'pending_before_true_TEST_labels':[
            'both physical fixed complete prediction SHA frozen and full D+other-node originalCPU/captures joint passed',
            'unique immutable joint score protocol/nonce path, allfive same arrays, FP64 exact author rules',
            'exclusive durable intent before labels; failure after labels burns attempt, no reset or retry'],
        'synthetic_tests_are_not_source_independent_review_or_actual_TEST_qualification':True,
        'trusted_whole_pickle_can_materialize_other_role_bytes':True,
        'candidate_guard_is_audited_access_gate_not_security_sandbox':True,
        'no_new_variant_seed_readout_baseline_swap_or_DEV_rescue':True,
        'advisor16_failed_no_recovery_no_poll_or_send':True,
        'lease_strong_actual_saves_pending_UTC':['09:30','11:30','13:00'],
        'overall_research_final_TEST_superiority_and_lease_saving_complete':False,
        'fresh_local_space_before_bytes':spaces,
        'large_already_saved_model_files_are_references_not_rehashed_or_downloaded':True}
    write(ROOT/'preparation.json',plan)
    doc='''本轮只完成最终 TEST 标签守卫的本地候选源和合成测试，未执行最终协议冻结、远程连接、模型前向或 TEST entry/ID/标签读取。当前双方 fullTRAIN100 整保存、异节点原CPU、fresh重放已通过；F89 对指定 CaReFlow93 的已选模 DEV 五项均差，不能换旧弱缓存或再挑读出救分。

第一次本地准备脚本误把Markdown摘要SHA当JSON SHA，门拒绝，原partial/source与拒绝记录另根保留，不覆盖。原seal summary_SHA对应Markdown 22bb7b...；JSON实际60f835...，与原manifest/ZIP完整原件一致。现分别pin，不修改旧source/回执/manifest/ZIP；不是旧模型、GPU或CPU完整证据失败。

冻结后的 final TEST 输入获取仍需独立输入角色源、原完整资产 SHA、身份捕获和来源审核。TEST 行数与 ID 目前未读取，不猜填。推理维持同一整 selected 模型的直接输出、batch128、原全部行和尾批；此批次与原双方 DEV 选模一致。CaReFlow 使用原作者每批 minmax，因此批次和顺序是方法的一部分，不能事后换批救分。

候选守卫逐一核双固定模型 state/whole-checkpoint SHA、同一原官方身份、prediction-only NPZ、原自然0和已经完成的 D+异CPU捕获联合报告；两侧物理预测字节相同后，先独占创建并 fsync 一次评分意图再访问标签。失败后保留令牌，禁删除令牌重新取标签。标签和五项共同评分源仍需另冻完整协议。此守卫依赖可信冻结源与独立保存审核，并非安全沙箱；trusted整pickle可能物化其它角色字节，历史旧runner TEST访问不抹去。

九个合成测试只在临时 fixture 上验证无标签输入、缺少模型、失败capture、保存字节/模型变化、ID错序/交叉/构造后变化、准备状态拒绝、重启重复访问和取标签后失败烧毁。它们不是实际 TEST 审核，也不是建议聊天第16完整审视。保持第16 failed，不重发同批或轮询。

成本仅来自原完成 GPU receipt：F 4347.465540796518 秒，CaReFlow 3910.2255212375894 秒，F约慢11.18196%。共同100轮/4000更新和选模规则已匹配，容量与耗时不同；这不证明总体方法无效或统计显著。停止当前版本 DEV 扩张是此固定比较的决策。

已保存整模型引用继续有效，本包不重复下载或哈希大权重。09:30/11:30/13:00 UTC强化真实动态保存待执行；本地准备ZIP不是新的remote COMPLETE capture。研究全面超过、最终 TEST 和全部租期保存未完成。
'''
    (ROOT/'preparation.md').write_text(doc,encoding='utf-8')
    files=[p for p in ROOT.iterdir() if p.is_file()]
    manifest={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
    write(ROOT/'manifest.json',{'status':'LOCAL_PREPARATION_FILES_MANIFEST','members':manifest})
    with zipfile.ZipFile(ROOT/'preparation.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'manifest.json']:z.write(p,p.name)
    with zipfile.ZipFile(ROOT/'preparation.zip') as z:
        assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
        for name,spec in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==spec['sha256']
        assert hashlib.sha256(z.read('manifest.json')).hexdigest()==sha(ROOT/'manifest.json')
        count=len(z.namelist())
    receipt={'status':'ACTUAL_LOCAL_PREPARATION_SHA_CRC_UNIQUE_FULL_MEMBERS_PASSED_NOT_REMOTE_CAPTURE',
             'actual_local_receipt_UTC':utc(),'root':str(ROOT),'zip':pinned(ROOT/'preparation.zip'),
             'unique_members':count,'CRC':True,'full_member_SHA':True,
             'manifest':pinned(ROOT/'manifest.json'),'synthetic_test_child_natural_exit_code':code,
             'no_research_TEST_access_or_forward':True,'D_free_after_bytes':shutil.disk_usage('D:/').free}
    write(ROOT/'receipt.json',receipt)
    link={'status':plan['status'],'D_root':str(ROOT),'D_receipt':pinned(ROOT/'receipt.json'),
          'D_preparation':pinned(ROOT/'preparation.json'),'D_zip':receipt['zip'],
          'no_remote_or_TEST_actual_execution':True,'overall_goal_complete':False}
    write(BASE/'outputs/正式双方最终TEST守卫本地准备接续.json',link)
    (BASE/'outputs/正式双方最终TEST守卫本地准备接续.md').write_text(doc+'\nD完整本地准备：'+str(ROOT)+'\n',encoding='utf-8')
    short='最新本地准备实际 '+receipt['actual_local_receipt_UTC']+'：最终TEST角色/双预测保存后一次标签门候选源及9个合成测试自然0已D SHA/ZIPCRC/唯一/全memberSHA保存；非remote capture/正式源冻结/TEST entry或ID标签访问/模型前向。先读《正式双方最终TEST守卫本地准备接续.md/json》。固定双方整模型和DEV五项负结果不变；官方TEST身份、执行源/CPU保存源和最终共同一次评分协议待另冻。租期09:30/11:30/13:00实际保存仍待执行，整体目标未完成。\n\n'
    status=BASE/'outputs/研究接续状态.md'
    status.write_text(short+status.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps({'root':str(ROOT),'receipt':pinned(ROOT/'receipt.json'),'test_pid':pid,'natural_exit':code,
                      'zip_members':count,'zip_sha256':receipt['zip']['sha256']},ensure_ascii=False))

if __name__=='__main__':main()
