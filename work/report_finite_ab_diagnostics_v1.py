from pathlib import Path
import json,datetime
o=Path(__file__).parent.parent/'outputs';r=json.loads((o/'有限任务风险两臂20条件诊断及C2启动快照独立核验.json').read_text(encoding='utf-8'))
t='''\n\n最新真实UTC%s：A/B预定20条件冻结干预与另外7次raw校准前向均实际完成、原receipt/229数组永久D，独立NumPy核验通过。default重放、换标签、最终上下文重放均0，参数SHA前后不变；只冻结DEV诊断，无参数更新或TEST访问。v1/v2收尾原float32平方差恒等式相减误差2.265e-6/2.734e-6超过原2e-6阈值，两失败日志/输出保留。v3使用同一float32预测转float64核验恒等式，采用更严1e-12；原科学预测/utility数组/条件/源参数未改变，两臂自然exit0，独立恒等式误差<1e-12。不可把首两版称通过。

raw风险估计对象是所有未控制供体共同输入时，删除一个原始供体的损失变化；最终风险估计对象是控制完成后删除实际传输消息、不重新优化剩余消息的损失变化。对于fixed A两者重合；对于scalar B两者不同，不能拿raw估计冒最终收益。换供体仅以固定作者128/101批次内循环移位替换单方向供体表示，再重新做无标签控制。
'''%datetime.datetime.now(datetime.timezone.utc).isoformat()
for n,x in r['rows'].items():
 raw=[c['raw_sign_agreement'] for c in x['channels']];fin=[c['final_sign_agreement_with_raw_proxy'] for c in x['channels']];rs=[c['raw_spearman'] for c in x['channels']];fs=[c['final_spearman_with_raw_proxy'] for c in x['channels']]
 t+=f"\n{n.upper()}：六方向raw符号一致率{min(raw):.4f}–{max(raw):.4f}、raw Spearman {min(rs):.4f}–{max(rs):.4f}；与最终传输收益符号一致率{min(fin):.4f}–{max(fin):.4f}、Spearman {min(fs):.4f}–{max(fs):.4f}。逐样本最优raw方向匹配{x['raw_best_direction_agreement']:.4f}，最优最终方向匹配{x['final_best_direction_agreement']:.4f}。\n"
 t+='\n|冻结条件|DEV229 MAE|MSE|相对默认平方风险变化|\n|---|---:|---:|---:|\n'
 for k,v in x['metrics'].items():t+=f"|{k}|{v['mae']:.8f}|{v['mse']:.8f}|{v['risk_change_vs_default']:+.8f}|\n"
t+='''\n这些结果不支持当前残差代理已可靠校准逐样本通道收益。alloff虽MAE低，但MSE更高；两种风险指标不能混用。删除某单通道的平均符号与整组删除效果也不能混同，通道共同作用与分布/选模影响仍存在。DEV已用于选模，当前只是单seed探索诊断；没有共享/补充/干扰语义真值或稳定提升/SOTA结论。

只有C2修订向量臂正式新100运行：seed91817前10fixed后90finite_vector，初始化/after20/100订单/400共享期及原输入/二阶隔离机制已通过，原v1失败永久保留、A/B不重训。启动真实UTC16:02:12，wrapper3571/child3572，真实16:12联合快照中60轮live；不是100结果。计划有限任务风险C2修订100轮冻结计划.json；二阶证据单token解析分支专门二阶与梯度隔离独立核验.json。真实快照C2_root/data/coding/finite_task_risk_c2_deployment_20261005T1600Z，capture12已三节点部署整SHA/原子complete/exit0后下载，全ZIP/member唯一/SHA通过，并分别保留旧C失败与新C2 live证据。未来C2完成才严格整模型磁盘重载、官方229原输入重放、永久完整保存与异训练异组装CPU原回执。
'''
with (o/'有限任务风险实验分析.md').open('a',encoding='utf-8') as f:f.write(t)
(o/'有限任务风险20条件冻结诊断分析.md').write_text('# 有限任务风险20条件冻结诊断\n'+t,encoding='utf-8')
print('REPORT_WRITTEN')
