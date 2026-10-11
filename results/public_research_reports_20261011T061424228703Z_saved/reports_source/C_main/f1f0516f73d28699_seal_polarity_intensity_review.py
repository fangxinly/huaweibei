"""Preserve a local literature/math review; no dataset/model/remote access."""
import hashlib, json, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLOCK = '2026-10-08 13:06:09 UTC'
OUT = ROOT / 'outputs/polarity_intensity_review_20261008T130609Z'
D = Path('D:/CodexBackups/selective_flow_20261003_1105/polarity_intensity_review_20261008T130609Z')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    record = dict(status='LOCAL_MATH_AND_LITERATURE_REVIEW_ONLY', actualclock_UTC=CLOCK,
        report=str(OUT/'REPORT.md'), report_SHA=sha(OUT/'REPORT.md'),
        external_INNER_statistics_independently_recomputed=False,
        official_performance_roles=['VAL','TEST'],
        candidate='Original flow with jointly correctable polarity/intensity and auxiliary supervision',
        fixed_executable_protocol=False, candidate_code_implemented=False,
        new_training=False, new_VAL_TEST_scores=False, new_remote_capture=False,
        key_limits=['Positive magnitude cannot fix text-only sign',
                    'Product of independent conditional means generally misses covariance',
                    'Modality ownership is not established by existing diagnostics',
                    'Auxiliary polarity/intensity supervision has MOSI prior work'],
        required_evidence=['Incremental TAV magnitude prediction on official VAL',
                           'Separate auxiliary-supervision and factorized-readout effects',
                           'Same-flow message contribution beyond flexible calibration',
                           'Fixed same-checkpoint official VAL/TEST five metrics vs CaReFlow'],
        literature=['https://aclanthology.org/W18-3306.pdf','https://aclanthology.org/D17-1115/'],
        new_training_resources_qualified=False, overall_goal_complete=False)
    (OUT/'review_status.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    note = (f'最新候选审阅 {CLOCK}：极性/强度辅助监督值得筛查，模态分工尚无实证；纯文本符号×非负幅度不能纠正符号，独立条件均值相乘遗漏协方差。2018 MOSI已有极性/强度多任务原论文，创新仍须同流方向消息超过灵活校准且由折外残差效用控制。仅新文献/数学审阅，Claude INNER数未新重算，无新源码冻结/训练/VALTEST分数/远端capture；先读polarity_intensity_review_20261008T130609Z/REPORT.md，官方范围和既有五项仍按《官方VAL_TEST与灵活校准实际接续.json》。不满足完整训练+2h保存余量，不启动/续租/删除；整体未完成。以下历史。\n\n')
    (OUT/'state_update.md').write_text(note,encoding='utf-8')
    files = [(OUT/'REPORT.md','REPORT.md'),(OUT/'review_status.json','review_status.json'),
             (OUT/'state_update.md','state_update.md'),(Path(__file__),'seal_polarity_intensity_review.py')]
    manifest = {name:sha(p) for p,name in files}
    package = OUT.with_suffix('.zip')
    assert not package.exists()
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for p,name in files: z.write(p,name)
        z.writestr('member_SHA.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    D.mkdir(parents=True,exist_ok=False)
    target = D/package.name
    shutil.copyfile(package,target)
    assert sha(package)==sha(target)
    for p in (package,target):
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            assert len(z.namelist())==len(set(z.namelist()))
            for name,h in manifest.items(): assert hashlib.sha256(z.read(name)).hexdigest()==h
    receipt = dict(actualclock_UTC=CLOCK,local_only=True,new_remote_capture=False,
        all_SHA_CRC_unique=True,ZIP_SHA=sha(target),bytes=target.stat().st_size,members=5,D=str(target))
    for p in (OUT/'D_receipt.json',D/'receipt.json'):
        p.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    state=ROOT/'outputs/研究接续状态.md'
    state.write_text(note+state.read_text(encoding='utf-8-sig'),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__ == '__main__': main()
