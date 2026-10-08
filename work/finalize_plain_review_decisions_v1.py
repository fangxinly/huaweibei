"""Post-experiment interpretation from frozen JSON aggregates only; no refitting."""
import datetime as dt, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np

b=Path.cwd();D=Path('D:/CodexBackups/selective_flow_20261003_1105/plain_scalar_residual_actual_20261006T115633Z');now=dt.datetime.now(dt.timezone.utc).isoformat();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=json.loads((D/'original/execute/results.json').read_text(encoding='utf-8'));c=json.loads((D/'original/execute/A_frozen_rule.json').read_text(encoding='utf-8'))['constant']
qa=np.array([v['q'] for v in result['A']['constant']['per_video']]);means=(c*c-qa)/(2*c);G=len(means);m=float(means.mean());V=float(np.mean((means-m)**2));qB=-m*m+(2*G-1)/(G-1)**2*V
constant_other=(G*m-means)/(G-1);individual=constant_other**2-2*means*constant_other
stored=np.array([v['q'] for v in result['B']['constant']['per_video']]);assert np.max(abs(individual-stored))<1e-12
proof=dict(status='FROZEN_JSON_ONLY_ALGEBRA_DECOMPOSITION_PASSED',utc=now,source_sha256=sha(__file__),fit_constant=c,inner_video_residual_means_derived_from_existing_Q=means.tolist(),mean=m,variance=V,negative_global_bias_term=-m*m,finite_three_video_estimation_cost=(2*G-1)/(G-1)**2*V,rebuilt_B_Q=qB,original_B_Q=result['B']['constant']['video_q'],individual_B_Q_error=float(np.max(abs(individual-stored))),largest_improvement_video=str(result['A']['constant']['per_video'][int(np.argmin(qa))]['video']),fraction_of_total_net_improvement=float(qa.min()/qa.sum()),no_new_labels_read=True,no_new_fitting_or_predictions=True,derived_mean_for_explanation_only_not_deployed=True,new_GPU=False)
(D/'post_result_algebra.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
report=f'''独立审视与停止决策，UTC {now}。

第七完整报告已全文阅读并保存；同意保留T_0原预测和FIT常数为简单对照，停止本批仿射扩容、换fold/seed/岭系数追分与重训交叉拟合。未启动新GPU或消息100。未来新teacher/fullflow必须在自己的合法FIT角色估计常数，不能移植T_0的c。

对正结果的解释收窄：常数完全不依赖p_F，只显示整体偏差修正，未证明逐样本残差可预测。最大改善视频占四视频净改善总量{100*proof['fraction_of_total_net_improvement']:.2f}%；删它后平均Q仍负但仅-0.002423514。通过事前成本门槛不等于幅度稳健、显著或风险保证。INNER曾用于teacher选模、全TRAIN此前探索，不能当新确认集。

B失败不是角色迁移唯一原因的证据。仅由已冻JSON的c与每视频Q，主聊天独立反解每视频残差均值，再核三视频常数的每视频Q，误差{proof['individual_B_Q_error']:.3g}。恒等式mean(Q_B)=-m²+(2G-1)/(G-1)²*V，在G=4时得到{proof['negative_global_bias_term']:.9f}+{proof['finite_three_video_estimation_cost']:.9f}={qB:.9f}。它支持视频均值异质性/小组估计代价的解释，仍不证明泛化原因。没有新标签读取、拟合、预测或分数；不部署反解均值或扫收缩救分。

当前唯一合理正结论是：固定T_0、固定fold0的FIT常数得到有限INNER探索收益，仿射无额外收益，INNER三视频校准诊断失败。下一步若确认常数，需透明披露历史的独立角色协议；若研究逐样本纠错或Q，需事先指定可推理额外输入与匹配直接校正对照。现批到此结束，不把这些候选自动变成新训练授权。GPU最新实际保存仍UTC11:39，估计期限不等于已释放。
'''
(b/'outputs/同折标量残差独立审视与停止决策.md').write_text(report,encoding='utf-8')
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第七次独立审视_标量残差实际结果与停止决策.md')
lp=b/'outputs/研究建议交流接续.json';ledger=json.loads(lp.read_text(encoding='utf-8'));ledger['updated_at_utc']=now
ledger['review_thread'].update(status='seventh_complete_full_report_read_no_pending_review',last_wait_cursor='5c299d87-8d70-412a-94b9-bdc45953b3ac:27',last_completed_turn='01a11113-4958-7c40-ba49-7e96834a31db',current_turn=None)
ledger['received_reviews'].append(dict(path=str(review),sha256=sha(review),completed_utc='2026-10-06T11:59:58Z',read_and_considered=True,accepted='保留FIT常数简单偏差对照，停止仿射/挑fold/消息100；收益幅度集中，不冒逐样本纠错。B有限三视频估计代价代数独立核过，不拟合新参数。',deferred='独立确认/新额外输入/Q/完整flow另立协议，当前无新GPU。'))
for batch in ledger['evidence_batches']:
    if batch['batch']=='plain_scalar_fold0_actual_20261006T115633Z':batch['status']='sent_once_completed_full_report_read_stop_decision_accepted'
ledger['scientific_state']='固定fold0 CPU与D/独立本地审核完成，第七完整审视已读，纯JSON代数解释已核；保留常数简单偏差基线，停止本批扩容/救分，无新GPU，最新远端状态保存UTC11:39，研究/租期未整体完成。'
lp.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
sp=b/'outputs/研究接续状态.md';state=sp.read_text(encoding='utf-8');old='turn01a11113-4958-7c40-ba49-7e96834a31db，cursor26当前inProgress，仅commentary非完整新建议。';assert old in state
state=state.replace(old,'turn01a11113-4958-7c40-ba49-7e96834a31db已真实completed，cursor27，第七完整报告已全文审阅并D保存，无待返回审视。')
state+=f'\n补充UTC {now}：第七审视支持停止扩容，并指出常数仅整体偏差修正、约90.54%净收益来自一视频。主聊天仅从原JSON独立核B常数有限三视频估计代价恒等式，未读新标签/拟合/预测/部署反解均值；见同折标量残差独立审视与停止决策.md及D post_result_algebra.json。当前无新远端查询/训练，平台租期未核。\n'
sp.write_text(state,encoding='utf-8')
folder=D/'final_research_records';folder.mkdir(exist_ok=False)
sources=[sp,lp,b/'outputs/同折标量残差固定fold0实际探索结果.md',b/'outputs/同折标量残差独立审视与停止决策.md',review,b/'work/finalize_plain_review_decisions_v1.py',D/'post_result_algebra.json',D/'review_completed_status_snapshot.json',D/'official_automation_update.json',D/'preservation_receipt.json',D/'post_review_failed_v1.py',D/'post_review_helper_failure.json']
manifest={}
for p in sources:
    q=folder/p.name;assert not q.exists();shutil.copy2(p,q);manifest[q.name]=dict(sha256=sha(q),bytes=q.stat().st_size)
with zipfile.ZipFile(D/'final_research_records.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
    for name in manifest:z.write(folder/name,name)
with zipfile.ZipFile(D/'final_research_records.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest)
    for name,entry in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256']
(D/'final_research_records_receipt.json').write_text(json.dumps(dict(status='POST_RESULT_REVIEW_FINAL_SHORT_STATE_AND_JSON_ONLY_ALGEBRA_D_SHA_ZIP_CRC_MEMBER_PASSED',utc=now,members=manifest,package_sha256=sha(D/'final_research_records.zip'),remote_capture=False,new_scientific_fitting=False,whole_research_complete=False),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'REVIEW_AND_JSON_ALGEBRA_SAVED','B_Q':qB,'individual_Q_error':proof['individual_B_Q_error'],'concentration':proof['fraction_of_total_net_improvement'],'members':len(manifest)},ensure_ascii=False))
