"""Audit actual new-lease receipts; preserve failure, no remote execution/deletion."""
import datetime, hashlib, json, shutil, zipfile
from pathlib import Path

BASE = Path('D:/CodexBackups/selective_flow_20261003_1105')
LOCAL = Path(__file__).resolve().parent/'third_lease_local_evidence_20261007T142430Z'
OUT = Path(__file__).resolve().parents[1]/'outputs'
utc = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
sha = lambda b: hashlib.sha256(b).hexdigest()
dump = lambda p, x: p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')

def audit_zip(path, receipt):
    raw=path.read_bytes(); assert sha(raw)==receipt['sha256']
    with zipfile.ZipFile(path) as z:
        names=z.namelist(); assert len(names)==len(set(names)) and z.testzip() is None
        manifest=json.loads(z.read('capture_manifest.json'))
        expected=manifest['member_sha256']
        assert set(names)==set(expected)|{'capture_manifest.json'}
        for n,h in expected.items(): assert sha(z.read(n))==h
        return {'sha256':sha(raw),'bytes':len(raw),'unique_members':len(names),'CRC_all_passed':True,
                'all_original_member_SHAs_passed':True}, {n:z.read(n) for n in names}

plan_path=BASE/'third_lease_public_restore_20261007T141430Z/third_lease_public_restore_plan.json'
plan=json.loads(plan_path.read_text()); audits={}
for node in ('A','B','C'):
    rp=LOCAL/f'{node}_public_receipt.json'; receipt=json.loads(rp.read_text())
    audit, files=audit_zip(LOCAL/f'{node}_public_restore.zip',receipt)
    pf=json.loads(files['physical_preflight.json']); rt=json.loads(files['runtime_verified.json'])
    assets=json.loads(files['public_assets_verified.json']); child=json.loads(files['dependency_child.json'])
    exitrec=json.loads(files['dependency_natural_exit.json'])
    assert receipt['node']==node and receipt['uuid']==pf['uuid']==plan['nodes'][node]['uuid']
    assert pf['compute'].strip()=='' and exitrec['natural_exit']==0
    assert child['pid']==exitrec['pid'] and child['fullargv']==exitrec['fullargv']
    assert rt['versions']==plan['runtime_exact_versions'] and rt['imports_natural_exit']==0
    assert assets['asset_sha256']==plan['asset_sha256'] and assets['labels_decoded'] is False
    assert pf['source_SHA']==plan['restore_source_SHA']
    assert sha(files['third_lease_public_restore_plan_20261007T141430Z.json'])==sha(plan_path.read_bytes())
    audits[node]={'receipt':receipt,'archive_audit':audit,'physical_preflight':pf,
                 'assets':assets,'runtime':rt,'dependency_child':child,'dependency_exit':exitrec}

np=BASE/'third_lease_weak_retention_native_20261007T141430Z_capture.zip'
nrp=BASE/'third_lease_weak_retention_native_20261007T141430Z_receipt.json'
nr=json.loads(nrp.read_text()); na,nfiles=audit_zip(np,nr)
sourceplan=json.loads(nfiles['source/candidate_protocol.json'])
for n,h in sourceplan['source_sha256'].items(): assert sha(nfiles['source/'+n])==h
nc=json.loads(nfiles['native_check/actual_child.json']); ne=json.loads(nfiles['native_check/natural_exit.json'])
result=json.loads(nfiles['native_check/stdout.log']); pf=json.loads(nfiles['native_check/preflight.json'])
assert nc['pid']==ne['pid'] and nc['fullargv']==ne['fullargv'] and ne['natural_exit']==nr['natural_exit']==0
assert pf['uuid']==plan['nodes']['B']['uuid'] and pf['compute'].strip()==''
for k in ('weak_harm_gradient_direction_passed','base_comparator_detached','strong_only_zero_penalty',
          'boundary_abs1_included','initial_source_identity','dummy_label_invariance',
          'masked_padding_invariance','singleton_gradients_finite','no_real_data_training','no_GPU_used'):
    assert result[k] is True
assert not nfiles['native_check/stderr.log']
shutil.copy2(np,LOCAL/'B_native_original.zip'); shutil.copy2(nrp,LOCAL/'B_native_receipt.json')

first=json.loads((BASE/'retained_checkpoint_lossless_storage_20261007T141430Z/file_00.json').read_text())
assert first['before_sha256']==first['after_sha256']
failure={'status':'COMPRESSION_HELPER_NATURAL_EXIT1_SECOND_POST_HASH_READ_ENOSPC',
         'observed_record_utc':utc(),'helper':'work/compact_retained_checkpoint_files.py',
         'first_file_fully_SHA_verified':first,
         'second_file':str(BASE/'staged_reference100_complete_actual_20261006T183917Z/a/run/out/complete_resume_full.pt'),
         'second_post_SHA_verified':False,'second_read_only_retry_exit':1,
         'error':'OSError [Errno 28] No space left on device during file.read',
         'D_free_bytes':shutil.disk_usage(BASE).free,'no_file_deleted':True,'compression_not_retried':True,
         'compression_freed_capacity_not_claimed':True}
dump(LOCAL/'storage_failure.json',failure)

relative_candidates=[
 'paired_official_prechecks_v2_actual_20261007T012116Z/minimal_fixed_F/a/run/out/precheck_after2_full.pt',
 'paired_official_prechecks_v2_actual_20261007T012116Z/careflow/a/run/out/precheck_after2_full.pt',
 'group5_single_precheck_20261007T084136Z/A_original/out/precheck_full.pt',
 'lease_dynamic_actual_20261007T093008Z/A_frozen_prefix_full.pt']
retention={'status':'CONCRETE_RETENTION_PROPOSAL_NOT_DELETION_AUTHORIZATION','actual_utc':utc(),
           'keep':['all selected checkpoints','final complete resume states','original predictions and metrics',
                   'source/plans/orders/RNG audit/records','failures and captures'],
           'intermediate_state_candidates':[
               {'path':str(BASE/n),'logical_bytes':(BASE/n).stat().st_size} for n in relative_candidates],
           'prior_no_delete_frozen_records_constraint_requires_human_resolution':True,
           'no_candidate_deleted':True,
           'reference_first10_not_candidate_because_reference_final_post_hash_unverified':True}
retention['candidate_logical_bytes']=sum(x['logical_bytes'] for x in retention['intermediate_state_candidates'])
dump(LOCAL/'retention_proposal.json',retention)

record={'status':'NEW_THIRD_LEASE_PUBLIC_RUNTIME_AND_B_NATIVE_CHECK_AUDITED_STORAGE_UNRESOLVED',
        'actual_record_utc':utc(),'human_lease_duration_hours':24,'platform_expiry_verified':False,
        'conservative_lease_end_UTC':'2026-10-08T14:00:00+00:00',
        'local_evidence_root':str(LOCAL),'public_restoration':audits,
        'native_check':{'D_original_archive':str(np),'audit':na,'receipt':nr,'actual_child':nc,
                        'natural_exit':ne,'result':result,'source_plan_SHA':sha(nfiles['source/candidate_protocol.json'])},
        'storage_failure':failure,'retention_proposal':retention,
        'new_real_training_started':False,'new_VAL_TEST_scores_available':False,
        'limitation':'detach removes comparison-term gradient; shared b can still drift. Synthetic test is not a performance result.',
        'old_second_lease_nodes_and_sessions_not_reused':True,
        'full_research_objective_complete':False,
        'public_restoration_new_D_capture_saved':False}
dump(LOCAL/'activation_audit.json',record)
dump(OUT/'第三租期恢复与弱情绪损失原生检查实际接续.json',record)
dump(OUT/'保留原件压缩失败实际接续.json',failure)
md='''三台新 P4 租期恢复与弱情绪损失检查实际结果

三台新服务器公共资产及私有运行环境均已真实恢复、正常退出；16公共资产SHA、25运行版本、ZIP CRC/唯一成员与完整原件SHA核验通过。B机原Torch合成检查child3567实际14:15:01自然exit0，弱惩罚梯度方向、比较项detach、强样本零额外惩罚、边界±1及三次合成更新契约通过。无真实数据训练、GPU模型前向或新VAL/TEST分数。

人类确认新租期24小时；平台到期未核，以Oct8 UTC14:00/BJ22:00保守安排。公共恢复capture实际14:22，已完整保存C工作区；B native完整原件已真正D且全SHA/CRC通过。不能称三个公共新capture已经D。

备份盘故障：压缩helper自然exit1。第一份F完整状态前后SHA相同；第二份参考100完整状态虽仍在原路径，但压缩后读取与只读重试均报ENOSPC，后SHA未复核。D当前0B，未删除文件、未重试压缩、不声称释放空间。失败及四项具体中间状态保留方案在本次C证据目录。旧禁删改冻结资料要求尚未被明确解除，故未清理旧大状态。参考前10保留作为恢复余地。

下一训练仍为单候选弱损害损失、原推理不变，尚未启动。detach只去掉通过比较器项的梯度，不冻结共享b，也不保证风险界或五项改善。所有旧VAL/TEST结果保持原样，完整五折和全面超过目标仍未完成。
'''
(OUT/'第三租期恢复与弱情绪损失原生检查实际接续.md').write_text(md,encoding='utf8')
state=OUT/'研究接续状态.md'
head=('最新第三租期实际 '+record['actual_record_utc']+'：先读《第三租期恢复与弱情绪损失原生检查实际接续.md/json》《保留原件压缩失败实际接续.json》。新A/B/C公共资产与私有runtime实际14:22均自然0/完整C capture全SHA CRC唯一通过；B新弱损失合成原Torch CPU child3567自然0，整D native capture通过，无真实训练或新VAL/TEST。人类新24h已确认，平台未核、保守Oct8UTC14:00。三个公共capture未D，不冒原GPU模型前向。D压缩helper自然1/第二大件读ENOSPC/现0B；首F完整态前后SHA一致，第二后SHA未过，不删旧冻结件、不重试压缩。新训练与完整五折未完成。具体保留候选四中间state共'+str(retention['candidate_logical_bytes'])+'B，仅提案未删；保留全部final/best/预测/源/日志/失败。以下均历史。\n\n')
state.write_text(head+state.read_text(encoding='utf8'),encoding='utf8')
archive=LOCAL/'complete_local_activation.zip'
files=[p for p in LOCAL.iterdir() if p.is_file() and p!=archive]
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
dump(LOCAL/'local_archive_receipt.json',{'actual_utc':utc(),'SHA':sha(archive.read_bytes()),'bytes':archive.stat().st_size,
                                    'scope':'Actual local activation audit and preserved native/public captures; not remote capture or full model'})
print(json.dumps({'status':record['status'],'native_D_SHA':na['sha256'],'local':str(LOCAL),
                 'candidate_bytes':retention['candidate_logical_bytes'],'D_free':failure['D_free_bytes']}))
