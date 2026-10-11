"""Publish verified official-role metrics without training or selection."""
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def run(stamp,clock):
    ws=Path(__file__).resolve().parent.parent;p=read(ws/'work/anchored_roles_pointer.json');backup=Path(p['D'])
    chain={}
    for stage in ('A_infer','B_audit','B_score'):
        root=backup/stage;r=read(root/'actual_capture_receipt.json');v=read(root/'actual_D_verification.json')
        assert r['child_natural_exit']==v['natural_exit']==0
        assert sha(root/'complete_actual_capture.zip')==r['archive_SHA']==v['archive_SHA']
        assert sha(root/'actual_capture_receipt.json')==v['receipt_SHA']
        chain[stage]=v
    raw=backup/'B_score/extracted/out/actual_VAL_TEST_five_metrics.json';result=read(raw)
    assert result['status']=='FIXED_NEW_UPGRADE_VAL_TEST_FIVE_METRICS_COMPLETE'
    assert result['prediction_SHA']==read(backup/'A_infer/extracted/out/prediction_result.json')['prediction_SHA']
    assert result['all_five_same_fixed_checkpoint'] and result['no_training_or_checkpoint_selection']
    folder=ws/'outputs'/('anchored_VAL_TEST_review_'+stamp);folder.mkdir()
    for n in ('actual_VAL_TEST_five_metrics.json','VAL_fixed_predictions_and_labels.csv','TEST_fixed_predictions_and_labels.csv'):
        shutil.copy2(backup/'B_score/extracted/out'/n,folder/n)
    lines=['# 原 AnchoredFlow 消息增量改进：VAL / TEST 五项实际结果','',
           '评测已完成：原 best36 骨干与固定第20轮消息增量适配器。此次只推理、复核和评分，没有训练、校准或更换检查点。',
           '', '## 五项指标','', '| 数据角色 | 模型 | Acc7 (%) | Acc2 (%) | F1 (%) | MAE | Corr |','|---|---|---:|---:|---:|---:|---:|']
    names={'new_fixed20_upgrade':'固定消息增量改进','same_flow_messages_off':'同一个流关闭消息'}
    differences={}
    for role in ('VAL','TEST'):
        r=result['roles'][role];m=r['metrics']
        for model in names:
            t=m[model]['overall'];lines.append(f"| {role} ({r['rows']}条/{r['videos']}视频) | {names[model]} | {t['Acc7']*100:.4f} | {t['Acc2']*100:.4f} | {t['F1']*100:.4f} | {t['MAE']:.9f} | {t['Corr']:.9f} |")
        differences[role]={k:m['new_fixed20_upgrade']['overall'][k]-m['same_flow_messages_off']['overall'][k] for k in ('Acc7','Acc2','F1','MAE','Corr')}
    lines+=['','## 强弱情绪分组','', '弱情绪定义为 |y|≤1，强情绪为 |y|>1。分组只用于报告，没有参与这次推理或模型选择。','',
            '| 数据角色 | 组别 | 条数 | 比例 (%) | 消息关闭 MAE | 改进 MAE |','|---|---|---:|---:|---:|---:|']
    for role in ('VAL','TEST'):
        r=result['roles'][role]
        for subset,name in [('weak','弱'),('strong','强')]:
            n=r[subset+'_count'];o=r['metrics']['same_flow_messages_off'][subset]['MAE'];u=r['metrics']['new_fixed20_upgrade'][subset]['MAE']
            lines.append(f"| {role} | {name} | {n} | {100*n/r['rows']:.4f} | {o:.9f} | {u:.9f} |")
    lines+=['','## 数据划分与结论边界','',
            '本骨干是在合并数据的视频分组实验中训练的；官方 VAL229 中135条属于 FIT、34条 INNER、60条 OUTER；官方 TEST685 中436条属于 FIT、88条 INNER、161条 OUTER。因此以上按原官方角色给出的分数不是独立官方保留测试结果，不能与无 TEST 训练重叠的 CaReFlow 成绩作正式胜负判断。',
            'INNER264/11 的上次改进 MAE 为0.622458224，属于另一数据角色，不与本次 VAL/TEST混用。此次未完成全部五折，也不宣称五项全面超过CaReFlow。',
            '', '下一优化判断应以当前视频隔离的留出结果及合法重新划分的 TRAIN/VAL 流程为依据，不使用本次 TEST 分数选择结构或检查点。',
            '', '## 验证与保存','',
            'A原完整编码器实际推理914条自然退出0；零标签、首批完整模型与显式前向一致、dummy标签0/7与重放一致、所有参数/缓冲/RNG不变。B用保存的100维源状态重放完整流及消息分支，最大误差9.54e-7；这是尾部CPU复核，不是CPU编码器前向。',
            'B一次读取对应VAL/TEST标签并同时评分；五项独立混淆矩阵/回归算术复核通过。三份完整远端捕获均已D保存，逐成员SHA、ZIPCRC、唯一成员、源码、完整argv、PID及自然退出联合通过。原完整父模型与小变更训练状态继续通过SHA引用保留，没有重复下载大模型。',
            f"评测完成UTC：{result['actual_UTC']}。本地报告记录actualclock：{clock}。"]
    report=folder/'原方案改进VAL_TEST五项实际报告.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    evidence={'actualclock_UTC':clock,'D_root':str(backup),'actual_chain':chain,'prediction_SHA':result['prediction_SHA'],'adapter_state_SHA':result['adapter_state_SHA'],'differences_new_minus_off':differences,'remote_execution_complete':True,'whole_goal_complete':False}
    write(folder/'actual_evidence_link.json',evidence)
    manifest={f.name:sha(f) for f in folder.iterdir() if f.is_file()};write(folder/'report_member_SHA.json',manifest)
    zp=backup/('final_VAL_TEST_report_'+stamp+'.zip')
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
        for f in folder.iterdir():z.write(f,f.name)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    continuation={'status':'CURRENT_FIXED_UPGRADE_VAL229_TEST685_FIVE_COMPLETE_D_OTHER_NODE_CPU','actualclock_UTC':clock,'report':str(report),'result':str(folder/'actual_VAL_TEST_five_metrics.json'),'D_root':str(backup),'report_ZIP':str(zp),'report_ZIP_SHA':sha(zp),'SHA_CRC_unique':True,'roles':{k:{'rows':v['rows'],'training_overlap':v['training_overlap'],'new_five':{m:v['metrics']['new_fixed20_upgrade']['overall'][m] for m in ('Acc7','Acc2','F1','MAE','Corr')}} for k,v in result['roles'].items()},'not_independent_official_benchmark':True,'whole_goal_complete':False,'session_closure_pending':True}
    target=ws/'outputs/原方案改进VAL_TEST五项实际接续.json';target.write_text(json.dumps(continuation,ensure_ascii=False,indent=2),encoding='utf8')
    write(backup/'final_VAL_TEST_report_seal.json',continuation)
    print(json.dumps(continuation,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);p.add_argument('--actualclock',required=True);a=p.parse_args();run(a.stamp,a.actualclock)
