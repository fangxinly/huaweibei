"""Local small-result preservation; no remote capture or model files."""
import hashlib,json,shutil,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs/negative_gain_review_20261008T120614Z'
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/'negative_gain_addendum_20261008T121114Z'
CLOCK='2026-10-08 12:11:14 UTC'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    actual=json.loads((OUT/'actual_result.json').read_text(encoding='utf-8'))
    native=json.loads((OUT/'linear_controls_contract.json').read_text(encoding='utf-8'))
    assert actual['source_SHA']==sha(ROOT/'work/negative_gain_review_v1.py')
    source=ROOT/'work/linear_control_addendum_v1'
    for name,digest in native['source_SHA'].items():assert sha(source/name)==digest
    assert native['status']=='LOCAL_SYNTHETIC_LINEAR_CONTROLS_CONTRACT_COMPLETE'
    record=dict(status='NEGATIVE_GAIN_SAVED_CSV_REVIEW_AND_OOF_LINEAR_CONTROLS_LOCAL_COMPLETE',
        actualclock_UTC=CLOCK,report=str(OUT/'REPORT.md'),specification=str(OUT/'线性对照纳入下一实验.md'),
        source=str(source),source_SHA=native['source_SHA'],
        predecessor='outputs/下一步原流方向残差控制实际接续.json',
        original_frozen_source_modified=False,official_VAL_TEST_opened=False,
        new_OOF_train_complete=False,original_server_validation=False,new_performance_proved=False,
        whole_goal_complete=False, original_source_four_CSV_SHA_verified=True,
        metrics='Centered Pearson and uncentered second-moment correlation separately; c<a exact convex criterion',
        best36_INNER=dict(g=-.8055667766702281,label_oracle_w=-.40079258493035463,
                        label_oracle_excess_MSE=.033478769702971596,
                        FIT_fitted_blend_MAE=.6286086346085094,FIT_calibration_b_MAE=.6301674079484807,
                        original_p_MAE=.6403909964472846),
        limitations=['Historical merged FIT/INNER selected checkpoints; diagnostic only',
                     'Negative weight/high correlation do not prove no new information',
                     'Old pooled b is not text expert or true same-flow OFF p0',
                     'New controls require complete video-isolated OOF predecessors and native runtime validation',
                     'OLS and predicted nonnegative utility are not risk guarantees'],
        resources=dict(C_free=1097072640,D_free=343961600,
                       conservative_lease_end_UTC='2026-10-08 14:00 UTC',platform_confirmed=False,
                       required_save_reserve_seconds=7200,new_crossfit_budget_qualified=False),
        local_archive_and_receipt_directory=str(D))
    control=ROOT/'outputs/负gain几何与线性对照实际接续.json'
    control.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    note=('最新本地数学/原CSV实核 '+CLOCK+'：先读《负gain几何与线性对照实际接续.json》及negative_gain_review_20261008T120614Z报告/补充说明。四历史FIT/INNER CSV原SHA过，负gain四行MSE/w*复现；best36 INNER gain-.8055668 vs事后w*-.4007926，多.03347877 MSE；FIT固定无约束权重迁移INNER MAE.6286086/仅校准b.6301674/原p.640391，皆历史已选模诊断，非官方VALTEST/真实OOF/新独立确认，不择部署。MSE二阶相关与Pearson分开；c<a为凸混合精确判据，负权重不自动证明无信息，旧b非文本/同流p0。新线性对照源（无约束1参数/校准2参数/校准+剩余delta3参数）本地NumPy合成自然0，视频/CAL泄漏拒绝，纯仿射delta退回校准；未原Torch/真实OOF/新VALTEST，原residual_direction_v2冻源不改。小件D完整SHA/ZIPCRC/唯一原件保存；C约1.10GB/D.344GB，保守租期14:00已不足完整队列+2h保存，不启动训练/续租/额外删旧件；原10min及整体未完成保持。以下历史。\n\n')
    (OUT/'state_update.md').write_text(note,encoding='utf-8')
    files=[(p,'review/'+p.name) for p in OUT.iterdir() if p.is_file()]
    files += [(p,'source/'+p.name) for p in source.iterdir() if p.is_file() and p.suffix=='.py']
    files += [(ROOT/'work/negative_gain_review_v1.py','source/negative_gain_review_v1.py'),
              (Path(__file__),'source/seal_negative_gain_addendum_v1.py'),
              (control,control.name),
              (ROOT/'outputs/coupling_residual_math_review_20261008T120037Z/耦合与任务残差几何核对.md','theory/耦合与任务残差几何核对.md')]
    names=[n for _,n in files];assert len(names)==len(set(names))
    manifest={n:sha(p) for p,n in files}
    path=OUT.with_suffix('.zip')
    assert not path.exists()
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p,n in files:z.write(p,n)
        z.writestr('member_SHA.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    D.mkdir(parents=True,exist_ok=False)
    target=D/path.name;shutil.copyfile(path,target);assert sha(target)==sha(path)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    receipt=dict(actualclock_UTC=CLOCK,status='LOCAL_D_SMALL_SOURCE_RESULTS_SHA_ZIPCRC_UNIQUE_COMPLETE',
                 ZIP_SHA=sha(path),bytes=path.stat().st_size,members=len(manifest)+1,
                 all_members_SHA_CRC_unique=True,D=str(target),
                 local_only=True,new_remote_capture=False,
                 diagnostic_child_actual_UTC=actual['actual_UTC'],synthetic_child_actual_UTC=native['actual_UTC'])
    for p in (OUT/'D_receipt.json',D/'receipt.json'):
        p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copyfile(control,D/control.name)
    state=ROOT/'outputs/研究接续状态.md'
    state.write_text(note+state.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
