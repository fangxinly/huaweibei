"""Preserve actual A scores and B raw-array verification, without rescoring."""
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path

def read(p): return json.loads(p.read_text(encoding='utf8'))
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def run(stamp):
    root=Path(__file__).resolve().parents[1]; e=root/'outputs/autonomous_mse16_20261008T175549Z'
    cap=read(e/'B100_independent_five_capture.json'); result=read(e/'B100_independent_five_result.json'); original=e/'B100_independent_five_original.zip'
    assert cap['natural_exit']==0 and sha(original)==cap['archive_SHA'] and original.stat().st_size==cap['archive_bytes']
    with zipfile.ZipFile(original) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        members=json.loads(z.read('member_SHA.json'))
        for n,h in members.items(): assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    assert result['independent_sklearn_scipy_maxerror']<1e-12 and result['predictions_from_frozen_original_NPZ']
    score=read(e/'A100_official_five_result.json')
    check=dict(status='LOCAL_ORIGINAL_B_RAW_ARRAY_FIVE_AUDIT_VERIFIED',actualclock_UTC=stamp,archive_SHA=sha(original),members=len(members),SHA_CRC_unique_all_members=True,independent_maxerror=result['independent_sklearn_scipy_maxerror'],CSV_numeric_rounding_maxerror=result['CSV_numeric_rounding_maxerror'],A_once_score_rerun=False)
    write(e/'B100_independent_five_C_verification.json',check)
    out=root/'outputs/MSE100官方VAL_TEST实际结果.md'
    lines=['# MSE-only 原流：官方 VAL / TEST 结果', '', '完整100轮、4000更新已完成；官方 VAL 按原128/101批次MSE均值选择第65轮。以下所有五项来自同一个选中检查点。B服务器以冻结NPZ原数组独立重算，最大差异5.55e-16。', '', '| 划分 / 模型 | Acc7 ↑ | Acc2 ↑ | F1 ↑ | MAE ↓ | Corr ↑ |', '|---|---:|---:|---:|---:|---:|']
    old={'VAL':[.5021834061135371,.875,.875084801400591,.5972565370388166,.8658304219847903], 'TEST':[.46131386861313867,.8717557251908397,.8714876426878345,.6506160153742254,.8251816144469359]}
    care={'VAL':[.4978165938864629,.8796296296296297,.8792475014697237,.6045238262758688,.8594938469285841], 'TEST':[.5065693430656935,.8763358778625954,.8761118749578298,.619535293923368,.851188467828589]}
    keys=('Acc7','Acc2','F1','MAE','Corr')
    for role in ('VAL','TEST'):
        for label,values in [('CaReFlow',care[role]),('旧原流 Huber',old[role]),('新原流 MSE',[result['independent_results'][role]['new'][k] for k in keys]),('新原流 MSE 消息关闭',[result['independent_results'][role]['messages_off'][k] for k in keys])]:
            lines.append('| '+role+' / '+label+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', '结论：单独把主损失从Huber换成MSE没有改善整体表现。新模型TEST五项全部低于CaReFlow；相比旧原流仅Corr提高。没有五项超过、稳定多种子或五折完成的结论。', '', '消息关闭是同一已训练流的功能性干预：VAL MAE下降0.004341，TEST MAE上升0.006388。它不等价于同预算重新训练的无消息对照，也不是统计显著性结论。', '', '退化诊断：第65轮之后TRAIN训练目标继续下降（0.01988→0.01470），VAL选模MSE反而上升（0.70459→0.72004）。训练目标包含context惩罚且在更新过程中统计，不能把它称作端点TRAIN任务MSE。消息在后期有真实非零梯度和扣衰减更新，不能再称为永久断路。', '', '弱样本定义为|y|≤1。VAL弱84/229（36.68%），TEST弱255/685（37.23%）；新模型VAL弱/强MAE为0.69142/0.55980，TEST为0.60636/0.68020。不同划分必须分别解释，不能拿历史INNER或201代替官方VAL/TEST。', '', '下一步沿已批准的原流结构做极性/强度监督与乘性读出，并保留同容量普通回归辅助监督对照。两臂共享订单、种子、预算及VAL选模规则；不根据本次TEST挑结构。候选已通过早先CPU合成集成，但真实完整编码器训练尚未开始，需要适配完整恢复、审计及保存。', '', '核验中的首次B独立评分审计失败来自CSV的float32短十进制导出与1e-12容差不匹配。修复只改变B读数来源为冻结NPZ；A推理及一次评分没有重跑，失败原件保留。', '', '完整训练原状态已保存到D盘、GitHub Release，B完成4000步Adam/调度器/RNG/订单/选择及真实CPU流重放。预测和A评分已Release保存；本报告及B独立评分完整原件正在追加保存。TEST此前已经查看，不能称新盲测。', '', f'记录时间：{stamp}', f'预测SHA：{result["prediction_SHA"]}', f'选中状态SHA：{result["selected_state_SHA"]}', f'B原件SHA：{cap["archive_SHA"]}']
    out.write_text('\n'.join(lines)+'\n',encoding='utf8')
    state=read(root/'outputs/自主优化实际接续.json');state.update(status='REAL_TRAIN100_OFFICIAL_VAL_TEST_FIVE_B_INDEPENDENT_COMPLETE_NO_WIN',updated_at_utc=stamp,new_VAL_TEST_scores=True,goal_complete=False)
    state['current_execution'].update(phase='MSE100_FINAL_METRICS_COMPLETE; PREPARING_PREVIOUSLY_APPROVED_POLARITY_INTENSITY_MATCHED_CONTROL',score_result=str((e/'A100_official_five_result.json').relative_to(root)),independent_result=str((e/'B100_independent_five_result.json').relative_to(root)),selected_epoch=65,new_scores=result['independent_results'],B_score_independent_natural_exit=0,B_first_score_failure_preserved=str((e/'B100_score_first_audit_capture.json').relative_to(root)),new_candidate_real_training_started=False)
    state['sessions_current'].update(A_SSH_foreground_waiting_child=False,B_SSH_foreground_waiting_child=False,A_SSH_foreground_waiting_capture_wrapper=False)
    state['official_MSE100_current']=result['independent_results'];state['formal_all_five_exceeded']=False
    write(root/'outputs/自主优化实际接续.json',state)
    brief=f'{stamp} 官方新五项已完成：MSE100/4000选65，VAL .502183/.875000/.874909/.608080/.860817；TEST .451095/.862595/.862229/.652712/.835848；CaReFlow TEST .506569/.876336/.876112/.619535/.851188，新模型五项全落后。B child1990自然0原NPZ独立重算max5.55e-16/C完整SHA CRC唯一全成员过；首CSV精度审计1857自然1保留，A未重推理/评分。TRAIN完整D+Release/B4000CPU过，预测/A评分Release过。下一原流极性强度与同容量辅助回归对照仅准备，真实候选未启动，不按TEST改结构。A53451/SFTP63220 B61821/19718 idle有效；旧closedID禁复用。整体/正式五项胜/全数据五折未完成。详MSE100官方VAL_TEST实际结果.md及自主优化JSON。\n\n'
    short=root/'outputs/研究接续状态.md';short.write_text(brief+short.read_text(encoding='utf8'),encoding='utf8')
    (root/'NEXT_EXECUTION.md').write_text(brief+'保持原10分钟。先完成B独立五项及失败原件Release/D小件保存，再接原流极性强度候选：真实16步完整前缀→保存与B原CPU→前16计入总4000续训；禁止重复MSE/旧已完成实验。新方案真实执行前fresh UUID/compute/argv/source/assets/空间/预算与至少2小时保存余量。\n',encoding='utf8')
    names=['A100_official_five_result.json','A100_inference_result.json','A100_score_capture.json','A100_score_publication.json','B100_audit_result.json','B100_independent_five_result.json','B100_independent_five_capture.json','B100_independent_five_exit.json','B100_independent_five_original.zip','B100_independent_five_C_verification.json','B100_prediction_audit_original.zip','B100_score_first_audit_original.zip','B100_score_first_audit_capture.json','B100_score_first_audit_exit.json','B100_score_first_audit_stderr.log']
    target=Path('D:/CodexBackups/selective_flow_20261003_1105')/('MSE100_official_five_'+stamp.replace('-','').replace(':','').replace(' ','_'))
    assert shutil.disk_usage(target.parent).free>200_000_000+sum((e/n).stat().st_size for n in names)
    target.mkdir();zpath=target/'full_small_proofs.zip'
    with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(e/n,n)
        z.write(out,out.name);z.writestr('member_SHA.json',json.dumps({n:sha(e/n) for n in names}|{out.name:sha(out)}))
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for n,h in json.loads(z.read('member_SHA.json')).items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    write(e/'MSE100_five_D_small_verification.json',dict(actualclock_UTC=stamp,path=str(zpath),SHA=sha(zpath),bytes=zpath.stat().st_size,SHA_CRC_unique_all_members=True,old_frozen_originals_deleted=False))
    print(json.dumps(dict(report=str(out),B_verified=check,D=str(zpath))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);run(p.parse_args().stamp)
