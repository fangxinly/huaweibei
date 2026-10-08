import json,datetime,hashlib,shutil,zipfile
from pathlib import Path
R=Path(__file__).parent.parent
D=Path('D:/CodexBackups/selective_flow_20261003_1105/five_metric_DEV_actual_20261006T183811Z')
j=json.loads((D/'actual_five_metric_result.json').read_text());plan=json.loads((D/'frozen_metric_plan.json').read_text())
text='CaReFlow五指标统一评价与实际DEV结果\n\n'
text+='实际本地CPU计算UTC '+j['actual_utc']+'。既有官方DEV229，13个真值0；Acc2/F1用216非零行。预固定旧A/B/C2的default预测与CaReFlow seed128缓存，未更新模型、读取TEST或挑seed/读出。原数组及原回执SHA、作者Acc7函数、独立F1与Pearson核验全部通过；原10成员ZIP真实D保存，非新GPU或异节点CPU。\n\n'
text+='| 既有DEV229 | Acc7↑ | Acc2↑ | F1↑ | MAE↓ | Corr↑ |\n|---|---:|---:|---:|---:|---:|\n'
names={'CaReFlow_cached_seed128_DEV':'CaReFlow复现seed128','Own_A_saved_default_DEV':'旧A','Own_B_saved_default_DEV':'旧B','Own_C2_saved_default_DEV':'旧C2'}
for name,row in j['rows'].items():
 text+='| '+names[name]+' | '+f"{100*row['Acc7']:.4f}% | {100*row['Acc2']:.4f}% | {100*row['F1']:.4f}% | {row['MAE']:.8f} | {row['Corr']:.8f}"+' |\n'
text+='\n三旧臂分别对该预固定CaReFlow缓存DEV五项全部更好；这是已选且反复探索DEV的描述性观察，不是相同seed/训练容量预算的公平因果比较、跨seed稳定性、论文TEST超过或新完整流100成绩。A的MAE最低；B其余四项较高；不据此拼接成不存在的单模型结果。新完整流使用FIT695/30与INNER153/4，不能把它的INNER选模数与这229行DEV或论文Test横比。\n\n'
text+='评价口径与作者发布代码一致：Acc7裁剪两数组到[-3,3]后NumPy取整（半整数到偶数）；Acc2/F1排除真值恰0，预测≥0为正，F1按真实类别支持数加权；MAE/Corr用全部原始未裁剪回归值，Corr是Pearson。额外Has0指标、混淆矩阵与未定义处理见JSON。用float64聚合，原float32微小舍入差单列，不算模型变化。来源：https://github.com/TmacMai/CaReFlow/blob/main/train_reflow_new.py 。\n\n'
text+='作者论文Table1仅外部目标参照：MOSI Acc7/Acc2/F1/MAE/Corr=50.6%/89.8%/89.7%/.616/.858；MOSEI=55.7%/87.9%/88.0%/.504/.799。来源：https://arxiv.org/html/2602.19140v1#S4.T1 。DEV与Test不同不能直接判胜。\n\n'
text+='人类目标是同一模型、同一合格评价协议五项同时超过CaReFlow：四项更高、MAE更低，按未舍入数判定；不平均掉失败项，不为各指标单独选checkpoint/阈值。当前冻结100源及INNER视频等权MSE earliest-strict-min选模保持。结束完整D/异节点CPU/严格重放保存通过后，给该唯一best补五项INNER选择用途指标；更后续development/head角色协议先冻，五项统一报告。正式论文比较另需同官方数据划分/标签尺度/训练与选模协议，锁定方案后一次最终Test评价；禁TEST挑结构。尚未实施新fullTRAIN/matched消融/MOSEI/finalTest。\n\n'
text+='原D证据：'+str(D)+'；frozen plan SHA '+j['frozen_plan_sha256']+'。\n'
p=R/'outputs/CaReFlow五指标统一评价与实际DEV结果.md';p.write_text(text,encoding='utf-8')
(R/'outputs/CaReFlow五指标统一评价与实际DEV结果.json').write_text(json.dumps({'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'report':str(p),'result':j,'plan':plan,'complete_goal_achieved':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
dest=D/'report_and_protocol';dest.mkdir(exist_ok=False)
for x in [p,R/'outputs/CaReFlow五指标统一评价与实际DEV结果.json',Path(__file__)]:shutil.copyfile(x,dest/x.name)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mf={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
(dest/'manifest.json').write_text(json.dumps({'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'members':mf,'C_free':shutil.disk_usage('C:/').free,'D_free':shutil.disk_usage('D:/').free},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with zipfile.ZipFile(dest/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in mf:z.write(dest/n,n)
 z.write(dest/'manifest.json','manifest.json')
with zipfile.ZipFile(dest/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in mf.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
print(json.dumps({'report':str(p),'report_SHA':sha(p),'D':str(dest),'ZIP_SHA':sha(dest/'records.zip')}))
