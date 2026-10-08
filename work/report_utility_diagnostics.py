from pathlib import Path
import datetime,hashlib,json
r=Path(__file__).resolve().parent.parent;out=r/'outputs'
complete=json.loads((out/'逐样本效用三组完整权重核验.json').read_text(encoding='utf-8'))
diag=json.loads((out/'逐样本效用24条件诊断核验.json').read_text(encoding='utf-8'))
cal=json.loads((out/'逐样本效用教师校准分析.json').read_text(encoding='utf-8'))
assert diag['status']=='ALL_THREE_24_CONDITION_GPU_DIAGNOSTICS_ARRAYS_AND_RISK_IDENTITIES_VERIFIED'
receipts=[]
for row in complete['rows']:
    node=row['node'];folder=r/'work/utility_completed_20261005'/node
    manifest=json.loads((folder/'independent_manifest.json').read_text(encoding='utf-8'))
    receipt=json.loads((folder/'destination_verification.json').read_text(encoding='utf-8'))
    assert receipt['status']=='NEW_INFLOW_INDEPENDENT_LEASED_COPY_ALL_SEVEN_FILES_SHA_VERIFIED'
    assert receipt['manifest_sha256']==hashlib.sha256((folder/'independent_manifest.json').read_bytes()).hexdigest()
    assert receipt['source_gpu_uuid']==manifest['source_gpu_uuid'] and receipt['source_gpu_uuid']!=receipt['target_gpu_uuid']
    assert receipt['files']==manifest['files'] and receipt['selection']==row['selection'] and receipt['mode']==row['mode']
    assert receipt['local_audit_sha256']==manifest['local_audit_sha256']
    assert receipt['files']['best.pt']['sha256']==row['checkpoint_sha256']
    receipts.append({'node':node,**receipt})
proof={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'THREE_NEW_COMPLETE_LOCAL_AND_ROTATED_INDEPENDENT_SEVEN_FILE_COPIES_VERIFIED','receipts':receipts,'limits':'Current three new weights only. No old weights retransferred, no claim of full research or lease-final completion.'}
with (out/'逐样本效用三份权重独立保存核验.json').open('x',encoding='utf-8') as f:json.dump(proof,f,ensure_ascii=False,indent=2)
lines=['# seed91813逐样本有方向效用实验','',
    '三组100轮训练、三份完整权重、100最小dev选模、官方229预测与六方向任务/效用数组、独立轮转七文件SHA及24条件冻结诊断均已通过实际审核。当前预测效用反馈没有优于匹配无反馈控制，冻结推断保留反馈增加了平均DEV风险。只有一个探索seed，结果不支持稳定提升或信息语义真值结论。','',
    '|模式|最佳轮|DEV MAE|作者批次MSE|Non0 acc2|','|---|---:|---:|---:|---:|']
for row in complete['rows']:
    v=row['metrics']['valid'];lines.append(f"|{row['mode']}|{row['selection']['best_epoch']}|{v['MAE']:.8f}|{v['author_batch_mse']:.8f}|{v['Non0_acc2']:.8f}|")
lines+=['','作者MSE为128/101两个批次均值的等权平均，用于选模。下文风险均为229样本等权，不能直接混用。所有模式的初始整模型和100轮订单相同，新任务/效用辅助目标也一致；模式绑定节点，跨模式差异不直接构成稳定因果结论。','','|训练模式|冻结all-off MAE|样本MSE：默认 / all-off|保留反馈收益：正值改善|默认与all-off预测RMS|','|---|---:|---:|---:|---:|']
for row in cal['rows']:
    b=row['default_retention_benefit_vs_all_off'];off=next(v for v in complete['rows'] if v['node']==row['node'])['metrics']['frozen_condition_off']
    lines.append(f"|{row['mode']}|{off['MAE']:.8f}|{row['main_default']['MSE']:.8f} / {row['main_all_off']['MSE']:.8f}|{b['mean']:+.8f}|{b['prediction_displacement_rms']:.8f}|")
lines+=['','这次最终直接反馈有可测的预测影响，但两个有反馈臂的平均风险均恶化。关闭反馈后的训练模型与重新训练的none臂是不同模型，不据此把训练路径差异归因于最终直接反馈。','','效用目标是q=(单模态任务头误差)−(双模态任务头误差)。预测臂的文本单模态样本MSE约.767，音频和视觉各约2.71/2.70；包含文本的双模态头明显好于弱音频/视觉任务头。它们的q高正均值和高权重主要反映任务头误差差距，不能单凭此认定语义互补。','','|方向（预测臂）|教师q均值|q正比例|效用符号准确率|正/负均衡召回|u与q Spearman|SmoothL1 / 零预测|','|---|---:|---:|---:|---:|---:|---:|']
pred=next(v for v in cal['rows'] if v['mode']=='predicted')
for v in pred['directions']:
    t=v['calibration'];lines.append(f"|{v['direction']}|{t['q_mean']:+.6f}|{t['positive_fraction']:.6f}|{t['sign_accuracy_nonzero_q']:.6f}|{t['balanced_sign_recall']:.6f}|{t['spearman_u_q']:.6f}|{t['smooth_l1']:.6f} / {t['zero_smooth_l1']:.6f}|")
lines+=['','A←T与V←T的符号准确率约.77，但均衡召回只有.5，表明这些方向没有辨别负效用子集。T←A/T←V的排序相关为负；A←V/V←A的预测损失劣于零预测。高总体正比例或平均权重不能代替逐样本效用校准。三个接收者的两个供体排序一致率（预测臂）为'+', '.join(f"{v['receiver']}={v['agreement_excluding_target_ties']:.6f}" for v in pred['donor_ordering'])+'。','','24预定推断条件覆盖默认、全部关闭、强制.5/预测权重、六个单通道关闭以及反馈内容/完整路径的逐通道与全供体置换。默认、off、强制fixed与predicted包装对原方法最大差异均为0；默认/off对已保存结果最大差异也为0。模型state与整weight SHA前后不变、0优化更新、未访问TEST。none臂所有干预预测保持不变。','','|预测臂通道|保留该通道的样本平均收益|u与实际收益Spearman|教师q与实际收益Spearman|实际收益符号均衡召回|','|---|---:|---:|---:|---:|']
pred_diag=next(v for v in diag['rows'] if v['mode']=='predicted')
def fmt(v):return '未定义' if v is None else f'{v:+.6f}'
for v in pred_diag['channels']:
    name=pred['directions'][v['channel']]['direction'];t=v['u_vs_actual_retention'];lines.append(f"|{name}|{v['mean_actual_retention_benefit']:+.8f}|{fmt(t['spearman_u_q'])}|{fmt(v['teacher_q_vs_actual_spearman'])}|{fmt(t['balanced_sign_recall'])}|")
lines+=['','单通道收益比较默认与关闭该通道后的主任务风险，不是双模态任务头q。每例均验证收益=−2eδ−δ²；完整JSON保留全部方向与全部24条件，不筛选显著结果。供体置换在固定128/101批次内循环1，仅反馈内容路径保留接收者原权重，完整路径同时重算任务头与效用；这些特定扰动不等价于重新训练、自然缺失或共享/补充/干扰的真值。','','|训练模式|全反馈供体置换的保留收益|全路径供体置换的保留收益|','|---|---:|---:|']
for v in diag['rows']:
    lookup={x['condition']:x for x in v['risks']};lines.append(f"|{v['mode']}|{lookup['shift_feedback_all']['default_retention_benefit']:+.8f}|{lookup['shift_full_all']['default_retention_benefit']:+.8f}|")
lines+=['','三新完整best.pt各745,113,122bytes，均保存当前work/utility_completed_20261005/{a,b,c}/run；独立轮转目标为A←B fixed、B←C predicted、C←A none，目标CPU核验全部七文件SHA。原seed91811/v3和旧84权重保持原保存，不重复传输。','','证据：逐样本效用三组完整权重核验.json、逐样本效用三份权重独立保存核验.json、逐样本效用教师校准分析.json、逐样本效用24条件诊断核验.json。新诊断源work/diagnose_utility_v4_v1.py SHA37b0df32b26c2e413543eb7551a15d0dbf55303c822a7397d646e1dfb29c8511，远端根utility_diagnostics_20261005T0439Z。','',
    '下一步优先改进效用监督与最终反馈的对应：教师q需与实际单通道主任务净收益区分，并检验错误供体及负效用识别。任何新结构先固定预算、匹配控制、标签隔离/梯度/初始化与订单检查，再据证据启动。当前本地空间不足以再保存三份同体积新完整权重，下一轮保存预算必须先解决；不删旧权重。13:40/14:10/14:30租期最终动态保存仍未执行，整体研究未完成。']
with (out/'逐样本效用实验分析.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({'status':proof['status'],'channels':pred_diag['channels'],'all_shift':[{ 'mode':v['mode'],'risks':[x for x in v['risks'] if x['condition'] in ['shift_feedback_all','shift_full_all']]} for v in diag['rows']]}))
