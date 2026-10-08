"""Analyze only available audited A/B final predictions; C is not completed here."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from analyze_utility_completed_v1 import digest,NAMES
out=Path('outputs');base=Path('D:/CodexBackups/selective_flow_20261003_1105/counterfactual_v5_completed_20261005');rows=[];arrays={}
for node in ['a','b']:
 cp_audit=out/('主任务反事实效用'+node.upper()+'完整权重核验.json');audit=json.loads(cp_audit.read_text(encoding='utf-8'));assert audit['stage']=='complete' and audit['status']=='COUNTERFACTUAL_SNAPSHOT_AUDITED';r=audit['rows'][0];assert r['node']==node and r['epochs']==100 and r['full_checkpoint_verified']
 cp=base/node/'run/best.pt';assert cp.stat().st_size==r['checkpoint_bytes'] and digest(cp)==r['checkpoint_sha256']
 manifest=json.loads((base/node/'independent_manifest.json').read_text());assert len(manifest['files'])==7
 for n,e in {**manifest['files'],**manifest['extra_metadata_saved']}.items():
  p=cp.parent/n;assert p.stat().st_size==e['bytes'] and digest(p)==e['sha256']
 dp=out/('主任务反事实效用'+node.upper()+'29条件诊断核验.json');d=json.loads(dp.read_text(encoding='utf-8'));assert d['status']=='COUNTERFACTUAL_29_CONDITION_FROZEN_DEV_ARRAYS_AUDITED';diag=d['rows'][0];assert diag['node']==node and diag['diagnostic']['checkpoint_sha256']==r['checkpoint_sha256']
 with np.load(cp.parent/'predictions.npz',allow_pickle=False) as f:z={k:f[k] for k in f.files}
 y=z['valid_y'].astype(float);p=z['valid_pred'].astype(float);off=z['condition_off_pred'].astype(float);assert y.shape==p.shape==off.shape==(229,);arrays[node]=z
 u=z['utility'].astype(float);w=z['predicted_weights'].astype(float);history=json.loads((cp.parent/'history.json').read_text());assert len(history)==100
 stats=[]
 for i,name in enumerate(NAMES):stats.append({'direction':name,'u_mean':float(u[:,i].mean()),'u_std':float(u[:,i].std()),'u_positive_fraction':float((u[:,i]>0).mean()),'predicted_weight_mean':float(w[:,i].mean()),'predicted_weight_std':float(w[:,i].std()),'predicted_weight_min':float(w[:,i].min()),'predicted_weight_max':float(w[:,i].max()),'predicted_weight_near_boundary_fraction':float(((w[:,i]<.05)|(w[:,i]>.95)).mean())})
 off_gain=np.square(off-y)-np.square(p-y)
 rows.append({'node':node,'mode':r['mode'],'selection':r['selection'],'metrics':r['results']['valid'],'true_229_MSE':float(np.square(p-y).mean()),'off_true_229_MSE':float(np.square(off-y).mean()),'default_retention_gain_vs_all_off':float(off_gain.mean()),'default_positive_retention_fraction':float((off_gain>1e-10).mean()),'default_negative_retention_fraction':float((off_gain< -1e-10).mean()),'utility_statistics':stats,'channels':diag['channels'],'joints':diag['joints'],'risks':diag['risks'],'diagnostic':diag['diagnostic'],'history_last':history[-1],'checkpoint':str(cp),'checkpoint_bytes':r['checkpoint_bytes'],'checkpoint_sha256':r['checkpoint_sha256'],'core_seven_local_sha_verified':True,'shared_phase_metadata_sha_verified':True,'completed_audit_sha256':digest(cp_audit),'diagnostic_audit_sha256':digest(dp)})
assert np.array_equal(arrays['a']['valid_y'],arrays['b']['valid_y']);y=arrays['a']['valid_y'].astype(float);a=arrays['a']['valid_pred'].astype(float);b=arrays['b']['valid_pred'].astype(float)
compare={'fixed_minus_none_MAE':float((np.abs(b-y)-np.abs(a-y)).mean()),'fixed_minus_none_true_229_MSE':float((np.square(b-y)-np.square(a-y)).mean()),'fixed_lower_absolute_error_fraction':float((np.abs(b-y)<np.abs(a-y)).mean()),'prediction_displacement_rms':float(np.sqrt(np.square(b-a).mean()))}
c_live=json.loads((out/'主任务反事实效用C运行核验_1410保存时.json').read_text(encoding='utf-8'));cr=c_live['rows'][0];assert cr['node']=='c' and cr['epochs']==81
r={'status':'AVAILABLE_A_B_COMPLETED_COUNTERFACTUAL_RESULTS_ANALYZED_C_FINAL_UNAVAILABLE','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'paired_dev_comparison':compare,'C_last_verified':{'epochs':81,'captured_at':cr['captured_at'],'full_final_checkpoint_available':False,'provisional_checkpoint_epochs_observed':31,'provisional_best_epoch':30},'source_sha256':digest(__file__),'limits':['Only A/B final results; C last observed 81 epochs and separate 31-epoch provisional full weight.','Common first ten epochs fixed; seed91814 only.','DEV used for selection and analysis; no TEST access.','Mode and GPU node are bound; no stable/SOTA or semantic truth claim.','14:30 real deadline capture not available; later recovery cannot fulfill its timestamp.']}
target=out/'主任务反事实效用已保存结果分析.json'
with target.open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=['# 主任务反事实效用 v5：已保存结果与缺口','', '本地已核验 A（none）与 B（fixed）各100轮完整权重、官方DEV229预测及预定29条件冻结诊断。C在14:10实际快照中为81轮；没有取得其100轮最终权重或最终预测，不能完成三臂比较。C的31轮临时完整权重和当时best30仅作为恢复材料。','', '|策略|选中轮|MAE|作者批次等权MSE|229例等权MSE|','|---|---:|---:|---:|---:|']
for v in rows:lines.append(f"|{v['mode']}|{v['selection']['best_epoch']}|{v['metrics']['MAE']:.8f}|{v['metrics']['author_batch_mse']:.8f}|{v['true_229_MSE']:.8f}|")
lines += ['',f"fixed相对none的MAE差 {compare['fixed_minus_none_MAE']:+.8f}，229例MSE差 {compare['fixed_minus_none_true_229_MSE']:+.8f}。这是一个探索seed下各自DEV选模的比较，不能解释为稳定提升。两臂先共同固定反馈10轮后切换；none不是从初始化开始始终无反馈。",'', '冻结诊断默认与all-off逐点重放均为0误差，四原方法/包装基准通过，整模型参数SHA保持一致；零optimizer、零TEST访问。natural与fixed参考收益分别审核，不混用。','', '|臂/方向|固定参考平均保留收益g|u符号均衡召回|u与g Spearman|预测门均值±标准差|','|---|---:|---:|---:|---:|']
for v in rows:
 for ch,st in zip(v['channels'],v['utility_statistics']):
  cal=ch['fixed_reference_calibration'];br=cal['balanced_sign_recall'];sp=cal['spearman'];lines.append(f"|{v['mode']} / {ch['direction']}|{cal['mean_gain']:+.8f}|{br:.5f}|{sp:.5f}|{st['predicted_weight_mean']:.5f} ± {st['predicted_weight_std']:.5f}|")
lines += ['', 'g=(关闭后预测−y)²−(参考预测−y)²；正值表示当前参考下保留通道降低风险。none的自然权重全0，因此自然关闭/换供体不改变默认预测；这不能证明其潜在消息无价值，fixed参考诊断才描述强制进入路径的影响。', '', f"B默认相对all-off的229例平均保留收益为 {rows[1]['default_retention_gain_vs_all_off']:+.8f}；其六个单通道fixed参考平均收益均为负。B六方向u均衡符号召回约0.44–0.50；目标从双模态任务头改成同一主任务终端参考后，已保存结果仍未显示可靠逐样本收益判断。该结论仅限A/B已观察DEV；C预测策略的表现未知。",'', '双关闭风险残差与预测非加性分开保存，float64恒等式独立通过。反馈供体替换仅改变消息生成中的供体，固定参考、自然u和权重保留；它不是完整模态置换或语义干扰真值。', '', '13:40与14:10旧/新组合capture均有永久本地和独立CPU证明。14:30没有实际capture证据；不能补写为已完成。UTC07:25恢复时原三个会话被远端关闭，C新连接拒绝原凭据并退出，未重试。A/B完整权重仍在永久D目录；B独立CPU文件回执已下载，A独立CPU通过输出在本轮终端中收到，原文件回执尚未下载。', '', '后续先解决有效节点与C最终材料的恢复依赖，再考虑新候选。数学目标、监督尺度、交互边界及固定教师候选见《主任务反事实效用监督边界与后续设计.md》。不因空闲GPU而重复旧实验，不在TEST挑结构，不提前开展五seed稳定结论。', '', '证据：主任务反事实效用A/B完整权重核验.json；主任务反事实效用A/B29条件诊断核验.json；主任务反事实效用已保存结果分析.json；租期1340/1410动态保存完成记录.json；C临时完整权重本地/独立保存核验_1342.json。']
(out/'主任务反事实效用实验分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':r['status'],'comparison':compare}))
