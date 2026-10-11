"""Local source/status preservation only. Copies already scored official metrics."""
import hashlib,json,shutil,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/flexible_official_calibration_20261008T125206Z'
SOURCE=ROOT/'work/flexible_official_calibration_v1'
D=Path('D:/CodexBackups/selective_flow_20261003_1105/flexible_official_calibration_20261008T125859Z')
CLOCK='2026-10-08 12:58:59 UTC'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    contract=json.loads((OUT/'local_contract_result.json').read_text(encoding='utf-8'))
    assert contract['status']=='LOCAL_SYNTHETIC_FLEXIBLE_OFFICIAL_CALIBRATION_COMPLETE'
    for n,h in contract['source_SHA'].items():assert sha(SOURCE/n)==h
    official=ROOT/'outputs/official_aligned_upgrade_review_20261008T111850Z/原方案改进与CaReFlow官方VAL_TEST实际结果.json'
    stored=json.loads(official.read_text(encoding='utf-8'))
    current={role:stored['roles'][role]['actual'] for role in ('VAL','TEST')}
    lines=['# 当前已完成的官方VAL/TEST五项', '',
           '仅摘录已保存原评分JSON；本轮没有重新评分或应用新校准。Acc7/Acc2/F1显示百分比。', '',
           '|划分|模型|Acc7|Acc2|F1|MAE|Corr|','|---|---|---:|---:|---:|---:|---:|']
    for role in ('VAL','TEST'):
        for key,label in (('careflow','CaReFlow'),('new','原方案改进'),('messages_off','原方案同流消息关闭')):
            m=current[role][key]
            lines.append(f'|{role}|{label}|{100*m["Acc7"]:.4f}%|{100*m["Acc2"]:.4f}%|{100*m["F1"]:.4f}%|{m["MAE"]:.9f}|{m["Corr"]:.9f}|')
    lines += ['', '新单调校准、校准后剩余增量、双方相同校准后的VAL/TEST分数：尚未产生。',
              '旧INNER/合并数据表仅留作历史机制诊断，不纳入本报告的效果验收。',
              '原结果JSON SHA：'+sha(official), '原结果来源：'+str(official)]
    (OUT/'官方五项当前结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    record=dict(status='OFFICIAL_VAL_TEST_ONLY_SCOPE_AND_FLEXIBLE_CALIBRATION_SOURCE_LOCAL_COMPLETE',
        actualclock_UTC=CLOCK,user_latest_scope='Official VAL/TEST five metrics only; TRAIN OOF is for fitting, not reported performance',
        source=str(SOURCE),source_SHA=contract['source_SHA'],
        protocol=dict(anchors='TRAIN OOF weighted p0 quantiles [0,.25,.5,.75,1], deduplicate',
                      calibration='nondecreasing piecewise-linear knot values',
                      delta_residualization='unconstrained WLS on entire same nonlinear basis B(p0)',
                      extra='q0+gamma*(delta-B*theta), rank-deficient gamma=0',
                      parties='same family/roles/weights/capacity for both; separately fitted parameters',
                      only_performance_roles=['VAL','TEST'],candidate_sweeps=False),
        existing_official_five=current,
        existing_score_source=str(official),existing_score_source_SHA=sha(official),
        existing_prediction_source=stored['frozen_prediction_source'],
        old_frozen_sources_modified=False,old_INNER_predictions_read_or_scored=False,
        user_flexible_table_independently_reproduced_this_turn=False,
        real_pair_TRAIN_OOF_calibration_qualified=False,
        new_calibrated_VAL_TEST_scores=False,new_training_or_GPU=False,original_server_validation=False,
        native_check=dict(local_only=True,child=contract['pid'],actual_UTC=contract['actual_UTC'],
                          nonlinear_p0_only_delta_exact_zero_gamma=True,KKT_error=contract['independent_convex_KKT_maxerror']),
        failure01='Missing local SciPy, preserved; corrected using installed NumPy, no new dependencies',
        report=str(OUT/'官方VAL_TEST与非线性校准说明.md'),D=str(D),
        full_crossfit_and_formal_five_metric_goal_complete=False,
        pending=['Pair complete source-qualified official TRAIN video OOF predictions',
                 'Original-runtime verification of new calibration module',
                 'Full pipeline resources and permanent checkpoint capacity',
                 'Fixed calibrators and same checkpoint official VAL/TEST evaluation'],
        resource_observation=dict(C_free=1093218304,D_free=339861504,
                                  conservative_lease_end_UTC='2026-10-08 14:00 UTC',platform_confirmed=False,
                                  required_save_reserve_seconds=7200,new_training_budget_qualified=False))
    control=ROOT/'outputs/官方VAL_TEST与灵活校准实际接续.json'
    control.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    note=('最新人类范围覆盖 '+CLOCK+'：用户明确不再看INNER性能，只看官方VAL/TEST五项及CaReFlow共同对照；TRAIN内完整视频OOF仅用于拟合校准，不当效果验收。先读《官方VAL_TEST与灵活校准实际接续.json》和flexible_official_calibration_20261008T125206Z说明/当前五项表。现官方原方案VAL MAE.597256537/TEST.650616015，C.604523826/.619535294，非新校准分数。新固定5锚点单调非线性校准+整个同B基空间delta残差化/折间稳定性记录/仅VALTEST五项源本地合成child15112自然0，非线性p0-only增量gamma0/KKT4.59e-16/视频CAL泄漏拒绝；第一次SciPy缺失自然1及源码已留，改NumPy无安装/无审批绕过。原冻源不改；本轮不读/重评INNER、不冒外部五次多项式表已复现。当前已核缺与官方模型对应的双方完整TRAIN OOF校准包，旧合并/teacher不能代替；新原runtime/真实OOF/新校准VALTEST/五指标胜未完成。小件D完整SHA CRC唯一保存，保守租期14:00已不足完整队列+2h保存，资源回复仍待，不重问密码/不续租/不删原件，原10min不改。以下历史。\n\n')
    (OUT/'state_update.md').write_text(note,encoding='utf-8')
    files=[(p,'review/'+str(p.relative_to(OUT)).replace('\\','/')) for p in OUT.rglob('*') if p.is_file()]
    files += [(p,'source/'+p.name) for p in SOURCE.iterdir() if p.is_file() and p.suffix=='.py']
    files += [(Path(__file__),'source/'+Path(__file__).name),(control,control.name)]
    names=[n for _,n in files];assert len(names)==len(set(names))
    manifest={n:sha(p) for p,n in files}
    package=OUT.with_suffix('.zip');assert not package.exists()
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for p,n in files:z.write(p,n)
        z.writestr('member_SHA.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    D.mkdir(parents=True,exist_ok=False)
    target=D/package.name;shutil.copyfile(package,target);assert sha(target)==sha(package)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    receipt=dict(actualclock_UTC=CLOCK,status='LOCAL_D_FLEXIBLE_CALIBRATION_SOURCE_AND_SCOPE_COMPLETE',
                 local_only=True,new_remote_capture=False,ZIP_SHA=sha(package),bytes=package.stat().st_size,
                 members=len(manifest)+1,all_SHA_CRC_unique=True,D=str(target))
    for p in (OUT/'D_receipt.json',D/'receipt.json'):
        p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copyfile(control,D/control.name)
    state=ROOT/'outputs/研究接续状态.md'
    state.write_text(note+state.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
