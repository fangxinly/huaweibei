import datetime, json, os, pathlib, shutil, sys
P=pathlib.Path
sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import seal,sha,raw
from raw_TRAIN_save_and_publish_v1 import checkzip
r=P(sys.argv[1]);dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
proof=json.loads((r/'support_D_preservation_receipt.json').read_bytes());pub=json.loads((r/'Release_receipt.json').read_bytes());summary=json.loads((r/'saved_support_independent_summary.json').read_bytes())
assert proof['archive_SHA']==pub['source_SHA']==sha(r/'complete_actual_TRAIN_support_original.zip') and pub['remote_digest_verified']
checkzip(r/'complete_actual_TRAIN_support_original.zip','capture_member_manifest.json')
result=summary['result'];assert result['TRAIN_labels_decoded']==0 and not result['new_solve_fit_score'] and result['max_original_prediction_reconstruction_error']==0
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
report='''# 原 TRAIN 拟合状态支持范围诊断

原冻结协议已实际执行一次，child1785自然退出0；1281 TRAIN行、52视频、原五折、四路径20个拟合状态全部保留。只读取原特征和已保存的拟合状态，不读取标签或VAL/TEST数值，不求解、重训、重新评分或修改原预测。

原预测逐项重构最大误差为0。完整原件SHA、全部member SHA/CRC、唯一成员及exactset已在D通过，Release远端digest已通过。

| 路径 | 最大单项贡献视频 | 最大单项贡献绝对值 | 全部留出行最大绝对Z |
|---|---|---:|---:|
'''
for name,s in summary['paths'].items():
 t=s['max_term_video'];report+=f"| {name} | {t['video_id']} | {t['max_abs_coefficient_term']:.9g} | {s['max_abs_standardized_feature']:.9g} |\n"
report+='''
视觉及联合路径的最大项都位于Iu2PFX3z_1s[11]，原视觉summary通道42的均值特征。原fit尺度1.6870982678569644e-7，held值-0.0006349086761474609，Z=-3763.350839625711；其单项贡献分别104.03647756784403和123.7047754650107。音频路径最大项位于2WGyTLYerpo[46]的原音频summary通道33均值特征，fit尺度5.3196230215487434e-8，Z=-1208.407083454356，单项贡献-41.84833946559604。

这证实冻结估计器中存在小fit尺度放大held输入和预测项的具体数值现象，并与此前已保存的误差集中视频对应。单项贡献不是总预测或误差的因果分解；其余项可能抵消。不能据此认定模态无信息、MI/PID、因果机制、普遍分布漂移或稳定显著收益，也不能据此更改原fold/lambda/视频/预测来救分。

全部TRAIN音频和视觉原数值非有限计数为0。有限零计数已逐行逐通道保存，但零是否代表缺失哨兵未知。原通道单位、上游处理配方及预训练曝光范围仍未知，不从维度推断通道语义。

此前A/B官方结果和原raw有限ridge负结果保持。完整神经视频五折及整体研究仍未完成。此次原协议成功执行，不证明底层审批问题已修复；旧20:12拒绝及旧SSH10986 Unknown仍保留，10986永久不复用。新同endpoint认证会话19826仅为本次实际接续。
'''
(r/'TRAIN原拟合状态支持诊断结果.md').write_text(report,encoding='utf8')
shutil.copyfile(__file__,r/P(__file__).name)
metadata=r/'completion_metadata';metadata.mkdir()
for name in ['retry_session_observation.json','capture_receipt.json','natural_exit.json','support_D_preservation_receipt.json','Release_receipt.json','saved_support_independent_summary.json','TRAIN原拟合状态支持诊断结果.md',P(__file__).name]:shutil.copyfile(r/name,metadata/name)
closure=seal(metadata,'complete_support_completion_metadata_original.zip');(r/'completion_metadata_D_receipt.json').write_bytes(raw(closure))
state=json.loads((dc/'D_current_research_state.json').read_bytes())
state['TRAIN_support_previous_policy_rejection_history']=state.get('current_research_execution_blocker')
state['current_research_execution_blocker']=None
state['latest_TRAIN_support_diagnostic_complete']={'actual_UTC':now,'root':str(r),'D_receipt':proof,'Release_receipt':pub,'completion_metadata_D_receipt':closure,'result':result,'summary':summary,'new_same_endpoint_SSH_session':19826,'old_unknown_SSH_session':10986,'old_session_permanently_unusable':True,'once_consumed':True,'no_rerun':True,'underlying_approval_repair_not_claimed':True}
state['latest_TRAIN_support_diagnostic_prepared']['reference']['remote_runtime_not_yet_qualified']=False
state['latest_TRAIN_support_diagnostic_prepared']['reference']['real_diagnostic_not_executed']=False
state['updated_at_utc']=now
state['next_gate']='Independently review the saved support-only result, publish exact source/summary blobs, and send this new SHA once for analysis. No rerun, fitting, scoring, TEST-based rescue, or claims of upstream cause.'
for dest in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
 tmp=dest.with_name(dest.name+'.supportcomplete.tmp');tmp.write_bytes(raw(state));os.replace(tmp,dest)
assert (dc/'D_current_research_state.json').read_bytes()==(c/'自主优化实际接续.json').read_bytes()
with (c/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n\n'+now+' 人类继续后原SSH10986 Unknown永久禁复用，新同N2 SSH19826真实认证/UUID/runtime/源资产SHA/合成资格过。TRAIN支持诊断child1785自然0，全部1281行/52视频/20fit、标签0/VALTEST数值0/新fit评分0，原预测重构误差0，once已消费。视觉held最大|Z|3763.35、单项贡献104.036/联合123.705；音频1208.41/单项41.848，小fit尺度放大held现象真，上游原因未知，不改原结果。完整原ZIP0f45efd12ff9a52361727018cd12c5aa4e08236aaea6c04b5be1910e791309ba D全member SHA/CRC/unique/exactset及Release远端digest过，D/C同步；旧审批失败仍历史保留，本次成功不冒底层修复。\n')
print(json.dumps({'root':str(r),'closure':closure,'result':result,'D_C_same_SHA':sha(dc/'D_current_research_state.json')}))
