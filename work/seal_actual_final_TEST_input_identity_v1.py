"""Bind actual A/B identity capsules, source lineage and natural closures."""
import datetime,hashlib,json,shutil,sys,zipfile
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
D=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_final_TEST_input_identity_actual_20261007T041107Z')
OLD=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    closed=read(D/'actual_closed_interactive_sessions.json')
    for v in closed.values():
        assert v['result']['status']=='fulfilled' and v['result']['value']['exit_code']==0
    nodes={n:read(D/n/'actual_D_saved_input_identity_audit.json') for n in ('A','B')}
    assert all(v['status']=='ACTUAL_TEST_INPUT_ONLY_SOURCE_ROOT_ARGV_CAPTURE_D_SHA_CRC_PASSED' for v in nodes.values())
    assert nodes['B']['other_CPU_original_A_sha256']==nodes['A']['saved_identity_sha256']
    assert nodes['A']['test_rows']==nodes['B']['test_rows']==685
    assert nodes['A']['test_raw_input_sha256']==nodes['B']['test_raw_input_sha256']
    provenance=D/'source_lineage';provenance.mkdir(exist_ok=False)
    sources={
        'official_repository_manifest.json':OLD/'outputs/careflow_reproduction/official_manifest.json',
        'official_README.md':OLD/'outputs/careflow_reproduction/official/README.md',
        'official_download_datasets.sh':OLD/'outputs/careflow_reproduction/official/datasets/download_datasets.sh',
        'original_public_download_manifest.json':OLD/'work/deberta-v3-base/download_manifest.json',
        'historical_reproduction_README_context_only.md':OLD/'outputs/careflow_reproduction/README.md'}
    for n,p in sources.items():shutil.copyfile(p,provenance/n)
    official=read(provenance/'official_repository_manifest.json')
    assert official['commit']=='5c9f9c7a0bb3f1202ebb3258052da2b99710c565'
    assert sha(provenance/'official_README.md')==official['source_sha256']['README.md']
    assert sha(provenance/'official_download_datasets.sh')==official['source_sha256']['datasets\\download_datasets.sh']
    manifest=read(provenance/'original_public_download_manifest.json')
    data=[x for x in manifest['files'] if x['file'].endswith('mosi.pkl')];assert len(data)==1
    data=data[0]
    assert data['sha256']==official['data_sha256']=='5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b'
    assert data['url']=='https://www.dropbox.com/s/sv94igp7zi3rsj1/mosi.pkl?dl=1'
    assert data['url'] in (provenance/'official_download_datasets.sh').read_text(encoding='utf-8')
    assert all(read(D/n/'run/out/actual_native_preflight.json')['original_assets_checked_sha256']['assets/mosi.pkl']==data['sha256'] for n in nodes)
    result={
        'status':'ACTUAL_FIXED_AUTHOR_MOSI_TEST685_INPUT_IDENTITY_A_D_OTHER_CPU_B_CAPTURE_JOINT_PASSED_NO_TEST_LABELS_OR_FORWARD',
        'actual_local_record_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'root':str(D),'nodes':nodes,
        'actual_A_B_input_identity_full_ID_order_layout_digest_equal':True,
        'author_provided_public_data_source':data,
        'official_repository':official['repository'],'pinned_author_commit':official['commit'],
        'pinned_author_download_script_SHA':sha(provenance/'official_download_datasets.sh'),
        'upstream_source_lineage_evidence':{n:{'path':str(provenance/n),'sha256':sha(provenance/n)} for n in sources},
        'fixed_public_cache_complete_roles':{'train':1281,'dev':229,'test':685},
        'author_download_lineage_and_actual_whole_asset_SHA_match':True,
        'not_raw_1284_229_686_feature_variant':True,
        'canonical_raw_SDK_or_paper_prediction_ID_coverage_proven':False,
        'Test_input_IDs_read_labels_never_indexed':True,
        'trusted_whole_pickle_may_materialize_all_role_bytes':True,
        'old_baseline_historical_TEST_once_access_not_erased':True,
        'TEST_prediction_GPU_forward_or_label_scoring_executed':False,
        'same_fixed_F89_C93_no_readout_seed_feature_or_baseline_swap':True,
        'CPU_audit_is_original_input_identity_only_not_model_forward':True,
        'current_four_interactive_sessions_actual_exit_bye_zero_closed_no_reuse':True,
        'closed_ids':[92518,92679,11700,2696],
        'no_current_C_native_observation_claim':True,
        'overall_five_metric_goal_final_TEST_and_all_lease_saving_complete':False,
        'lease_strong_actual_saves_pending_UTC':['09:30','11:30','13:00'],
        'next':'Freeze complete final prediction/selected restore/author preprocessing/batch128 keep-tail/state-allRNG/dummy invariance/original CPU-array audit/capture/preservation/once-only five-score sources. Bind both original685-row input identities and author provided file. No paper unrounded result equivalence or raw686 coverage claim; no TEST labels until both physical predictions frozen and complete D+other-node CPU original audits passed.'}
    output=D/'actual_A_D_B_input_identity_joint.json';write(output,result)
    doc='''固定作者公开 MOSI 缓存的 TEST 零标签身份实际完成：A child3521 UTC04:10:56.415690自然0，B原CPU child32123 UTC04:14:45.917733自然0。双方1281 TRAIN/229 DEV/685 TEST原ID、完整行序、词级输入布局及原输入摘要严格一致；未索引任何角色标签，未生成TEST预测或做模型前向。trusted整pickle可能物化所有角色字节，不冒安全沙箱或历史TEST从未访问。

A原capture child3638实际04:11:43.993233 COMPLETE、04:11:44.022002自然0，共20成员；B原capture32191实际04:15:39.699857 COMPLETE、04:15:39.728667自然0，共22成员。两侧receipt-first D完整ZIP SHA/CRC/唯一/所有原成员/source/fullargv/PID/自然退出/原native UUID、空compute、全公共资产、空间和保存余量联合通过。B关联的是A真实D保存原件 SHA，不是人工JSON假资格；不是CPU模型前向。大模型没有重下载或改写。

来源短证据已原样保存：CaReFlow作者commit5c9f9c7...的README和datasets/download_datasets.sh字节与原official_manifest一致；download脚本内MOSI Dropbox URL与原download_manifest相同，记录数据SHA5c3cc6.../14158784字节，并与本轮A/B原公共整资产实际SHA核验一致。因此固定比较使用作者提供的1281/229/685预处理缓存，不能换旧1284/229/686预提取特征数据或把两版本横比。这里未证明原始SDK全部ID覆盖或论文未舍入预测等价。

本批A SSH92518/SFTP92679、B SSH11700/SFTP2696全部明确exit/bye实际0，关闭ID禁复用；C未fresh，不冒三机实时。两个固定best F89/C93与已选模DEV五项负结果不变。下一步按真实共同685 ID、原作者batch128保尾批与固定直接输出，另冻完整模型还原/推理/原CPU数组审核/自然capture/D保存/一次共同五项评分来源。最终预测和真实TEST标签评分未执行，整体五项全面超过以及租期09:30/11:30/13:00实际强化保存未完成。
'''
    (D/'actual_A_D_B_input_identity_joint.md').write_text(doc,encoding='utf-8')
    shutil.copyfile(BASE/'work/audit_saved_final_TEST_input_identity_v1.py',D/'actual_local_capsule_audit_source.py')
    shutil.copyfile(Path(__file__),D/Path(__file__).name)
    files=[p for p in D.rglob('*') if p.is_file() and '/run/' not in p.as_posix()]
    members={p.relative_to(D).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    mf=D/'local_complete_identity_seal_manifest.json';write(mf,{'kind':'LOCAL_COMPLETE_IDENTITY_SEAL_NOT_NEW_REMOTE_CAPTURE','members':members})
    zp=D/'local_complete_identity_seal.zip'
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files+[mf]:z.write(p,p.relative_to(D).as_posix())
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(files)+1
        for n,v in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
    receipt={'actual_local_receipt_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'kind':'LOCAL_COMPLETE_IDENTITY_SEAL_NOT_NEW_REMOTE_CAPTURE','zip_path':str(zp),
             'zip_sha256':sha(zp),'zip_bytes':zp.stat().st_size,'members':len(files)+1,
             'full_SHA_CRC_unique_passed':True,'joint_JSON_path':str(output),'joint_JSON_sha256':sha(output),
             'joint_Markdown_path':str(D/'actual_A_D_B_input_identity_joint.md'),
             'joint_Markdown_sha256':sha(D/'actual_A_D_B_input_identity_joint.md'),
             'D_free_bytes':shutil.disk_usage('D:/').free}
    assert receipt['D_free_bytes']>=6*1024**3
    rp=D/'local_complete_identity_seal_receipt.json';write(rp,receipt)
    outputs=BASE/'outputs'
    (outputs/'正式双方TEST685零标签身份与完整保存实际结果.md').write_text(doc+'\nD原件：'+str(D)+'\n',encoding='utf-8')
    record=dict(result,local_seal_receipt_path=str(rp),local_seal_receipt_sha256=sha(rp))
    (outputs/'正式双方TEST685零标签身份与完整保存实际结果.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    continuation=outputs/'正式双方TEST零标签身份源冻结接续.json';prev=read(continuation)
    prev['status']=result['status'];prev['latest_actual_joint_path']=str(output);prev['latest_actual_joint_sha256']=sha(output)
    prev['node_A']=nodes['A'];prev['node_B']=nodes['B'];prev['capture_D_joint']='ACTUAL_PASSED'
    prev['author_provided_cache_source_lineage_qualified']=True;prev['remote_execution_started']=True
    prev['canonical_raw_or_paper_ID_coverage_proven']=False
    prev['no_actual_TEST_entry_input_ID_or_label_access']='HISTORICAL_FREEZE_ONLY: NOW ACTUAL INPUT_IDS_READ, NO_LABELS'
    prev['all_interactive_sessions_closed_no_reuse']=[92518,92679,11700,2696]
    continuation.write_text(json.dumps(prev,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    previous=outputs/'正式双方TEST零标签身份源冻结接续.md'
    previous.write_text('最新实际：A/B固定作者TEST685输入身份与完整D保存联合过，无标签或模型前向；先读《正式双方TEST685零标签身份与完整保存实际结果.md/json》。以下为冻结前历史。\n\n'+previous.read_text(encoding='utf-8'),encoding='utf-8')
    status=outputs/'研究接续状态.md'
    head='最新实际 '+receipt['actual_local_receipt_UTC']+'：固定作者公共MOSI TEST685零标签输入身份A3521/B原CPU32123自然0、完整原ID/order/layout/input摘要严格一致，A20/B22原capture自然0→D全SHA/ZIPCRC/唯一/源/fullargv/原PID联结过；原作者下载脚本URL与原download_manifest/实际整asset SHA5c3cc6...一致。先读《正式双方TEST685零标签身份与完整保存实际结果.md/json》。不是原始SDK686覆盖/论文原预测等价/CPU模型前向/TEST真实标签或推理评分。四会话92518/92679/11700/2696均明确exit/bye0禁复用；C未fresh。双方完整selected模型/DEV五项负结果不变，最终完整预测与一次五项来源协议待另冻；租期09:30/11:30/13:00动态保存及整体目标未完成。\n\n'
    status.write_text(head+status.read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
