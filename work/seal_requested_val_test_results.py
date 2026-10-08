"""Validate saved real GPU/CPU capsules and report all available VAL/TEST five metrics."""
import json,hashlib,zipfile,shutil,sys
from pathlib import Path
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105');OUT=Path('outputs');CLOCK=sys.argv[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def audit_capsule(root,archive,receipt,capsule):
 r=read(root/receipt);assert sha(root/archive)==r['sha256']
 with zipfile.ZipFile(root/archive) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 d=root/capsule;m=read(d/'manifest.json')
 for n,h in m['member_sha256'].items():assert sha(d/n)==h
 g=read(d/'original/actual_stage_receipt.json');n=read(d/'execution/natural_exit.json');c=read(d/'execution/actual_child.json')
 assert n['natural_exit']==0 and n['pid']==g['pid']==c['pid'] and n['fullargv']==g['fullargv']==c['fullargv']
 return g,dict(archive=str(root/archive),sha256=r['sha256'],pid=g['pid'],actual_exit_utc=n['actual_exit_utc'],fullargv_linked=True)
old=BASE/'old598_requested_VAL_TEST_source_20261007T112358Z';new=BASE/'selected20_requested_VAL_TEST_20261007T111903Z';rows={};evidence={}
for name,folder in [('A','finite_risk_completed_20261005/a'),('B','finite_risk_completed_20261005/b')]:
 g,ge=audit_capsule(old,name+'_capture.zip',name+'_capture_receipt.json',name+'_capsule');c,ce=audit_capsule(old,name+'_CPU_capture.zip',name+'_CPU_capture_receipt.json',name+'_CPU_capsule')
 assert c['scores']==g['scores'] and c['original_receipt_sha256']==sha(old/(name+'_capsule/original/actual_stage_receipt.json'))
 cp=BASE/folder/'full_checkpoint.pt';assert sha(cp)==g['checkpoint_sha256']
 rows['old_'+name]=dict(VAL=g['scores']['val'],TEST=g['scores']['test'],scope='Original fullTRAINfit/DEVselected saved model; TEST not used in model training/selection',checkpoint_sha256=g['checkpoint_sha256'],DEV_replay_max_error=g['original_DEV_replay_max_error'])
 evidence[name]=dict(GPU=ge,other_node_CPU=ce,D_complete_original=str(cp))
g,ge=audit_capsule(new,'A_complete_capture.zip','A_capture_receipt.json','A_original_capsule');c,ce=audit_capsule(new,'B_CPU_complete_capture.zip','B_CPU_capture_receipt.json','B_CPU_original_capsule')
assert c['original_receipt_sha256']==sha(new/'A_original_capsule/original/actual_stage_receipt.json') and c['saved_metrics_exact_equal'] and c['predictions']==g['prediction_sha256']
rows['anchored_selected20']=dict(VAL=g['scores']['val']['p'],TEST=g['scores']['test']['p'],INNER=g['scores']['inner']['p'],scope='Descriptive official-role scores of pooled video-fold model; includes FIT/INNER examples; not independent official TEST',official_role_overlap=g['official_role_overlap'])
evidence['anchored_selected20']=dict(GPU=ge,other_node_CPU=ce,checkpoint_SHA_ref=g['checkpoint_sha256'],parent_complete_joint_sha256='45a68a9aaed2136ed5849d43a46bce1cbcfe876b646fa96aec6e8722b05bae0c')
dev=read(OUT/'正式双方fullTRAIN100固定best一次DEV五项实际结果.json');test=read(OUT/'正式双方TEST685一次五项实际结果.json')
for name in ['minimal_fixed_F','careflow']:rows[name]=dict(VAL=dev['methods'][name]['fixed_DEV_five'],TEST=test['metrics'][name],scope='Matched fixed official fullTRAIN100; saved official DEV selection and once TEST evaluation',best_epoch=dev['methods'][name]['best_epoch'])
historic=read(OUT/'CaReFlow五指标统一评价与实际DEV结果.json')['result']['rows'];rows['old_C2']=dict(VAL=historic['Own_C2_saved_default_DEV'],TEST=None,scope='VAL229 already scored; requested TEST not executed')
failure=read(old/'C2_failed_execution/natural_exit.json');assert failure['natural_exit']==1
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第十七次独立审视_合法任务教师与冻结范围的最小比较.md')
exchange=read(OUT/'研究建议交流接续.json');exchange['pilot40_actual_batch'].update(status='COMPLETE_REVIEW17_READ_INDEPENDENT_DECISION_RECORDED',complete_report_read_or_adopted=True,report_path=str(review),report_sha256=sha(review),independent_main_decision='Accept fixed-model b/f/p diagnosis, legal same-T ownership/dropout controls and counting teacher cost; diagnosis actually complete. Defer new S/E training due storage/time; never infer freeze advantage or negative gain causality. New human VAL/TEST request permits descriptive official-role scoring; full grouped OOF remains incomplete.',actualclock_decision_UTC=CLOCK);write(OUT/'研究建议交流接续.json',exchange)
report=dict(status='AVAILABLE_FIXED_VAL_TEST_FIVE_COMPLETE_D_OTHER_CPU_C2_TEST_PENDING_HUMAN_BUDGET_CHANGE',actualclock_record_UTC=CLOCK,models=rows,evidence=evidence,C2_failure=failure,C2_failure_before_model_forward=True,C2_explicit_budget_change_question_pending=True,no_fivefold_complete=True,no_five_metric_CaReFlow_superiority=True,selected20_same_checkpoint_diagnosis=g['scores'],selected20_OUTER_full437_not_scored=True,selected20_outer_subset_labels_in_requested_official_scores={'VAL':60,'TEST':161},TEST_scores_not_used_for_tuning=True,metric_definition='Acc7 clip-round all; Acc2/support weighted F1 exclude true0 and pred>=0; raw MAE/Pearson',review17_sha256=sha(review))
write(OUT/'所有固定模型VAL_TEST五项实际对齐结果.json',report)
md=['固定模型 VAL / TEST 五项实际结果','',f'记录：{CLOCK}。VAL=官方DEV229；TEST=官方685。Acc7/Acc2/F1以下用百分比。全五项来自同一个固定checkpoint。','', '| 模型 | 集合 | Acc7 % | Acc2 % | F1 % | MAE | Corr |','|---|---|---:|---:|---:|---:|---:|']
for name,label in [('careflow','CaReFlow best93'),('minimal_fixed_F','固定F best89'),('old_A','旧A best40'),('old_B','旧B best11'),('old_C2','旧C2 best37'),('anchored_selected20','新20轮（含训练/选模样本）')]:
 for role in ['VAL','TEST']:
  v=rows[name][role]
  md.append('| '+label+' | '+role+' | '+(' | '.join([f"{100*v[k]:.4f}" for k in ['Acc7','Acc2','F1']]+[f"{v[k]:.8f}" for k in ['MAE','Corr']]) if v else '未测 | 未测 | 未测 | 未测 | 未测')+' |')
md+=['','旧A/B本次新测试自然0，原完整VAL预测重放及参数/RNG不变、dummy换标签一致、原源码/整模型SHA、D完整capture和B独立保存数组算术均通过。旧A的原VAL重放误差0；B也须以原GPU回执为准。B没有做模型前向。固定F/CaReFlow沿用已完成正式分数，未重跑。','', '旧A VAL MAE .59844172→TEST .64265227；旧B .59862398→.64640454。旧0.598确属完整VAL229，但改善未保留到TEST。旧A TEST Acc2/F1胜新CaReFlow，Acc7/MAE/Corr差；没有五项全面胜。新F TEST五项都差。旧A/B与CaReFlow训练seed/容量/任务教师预算不同，表中对齐评价集不是新匹配训练实验。','', '新20轮VAL229含FIT135/INNER34/OUTER60；TEST685含FIT436/INNER88/OUTER161。其.54210749/.54422463是混合训练/开发/外层样本的描述分数，不能称独立TEST改善或完整五折。请求前原OUTER未评分；本次明确请求后这些外层子集标签进入官方角色汇总，不能继续声称所有OUTER标签未评分。完整OUTER437/五折OOF仍未评分，不用这些分数选新结构。','', '同固定20轮FIT/INNER eval中b/f/p实测：FIT b MAE.568942→p .411217；INNER b .773530→p .679452，INNER五项均提高。INNER弱情绪111条 b .490774→p .638971；强情绪153条 b .978666→p .708821。修正实际帮助整体和强情绪，却损害弱情绪。负gain本身不是坏机制的证据；这不是重新训练的去流消融。后续方向需基于FIT/INNER预冻结弱情绪误差控制与合法同教师冻结范围对照，尚无新训练收益。','', 'C2 TEST启动前被剩余时间<90秒执行+7200秒保存门拦下，child5494自然1，未模型前向/TEST预测或评分。原失败完整保留。已向人类询问仅C2把本次保存预留改为1小时；未回复前不执行候选例外源码，不删旧原件。','', '第17建议完整已读，独立采纳固定模型诊断/合法共同起点/精确ownership与dropout一致/总教师成本；新S/E训练因空间及时间暂缓。历史第16失败不抹去。']
(OUT/'所有固定模型VAL_TEST五项实际对齐结果.md').write_text('\n'.join(md),encoding='utf8')
seal=new/'result_seal';seal.mkdir()
for p in [OUT/'所有固定模型VAL_TEST五项实际对齐结果.json',OUT/'所有固定模型VAL_TEST五项实际对齐结果.md',OUT/'研究建议交流接续.json',review,Path(__file__)]:shutil.copy2(p,seal/p.name)
write(seal/'manifest.json',dict(actualclock_UTC=CLOCK,sha256={p.name:sha(p) for p in seal.iterdir() if p.is_file()}))
with zipfile.ZipFile(new/'completed_available_results_seal.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in seal.iterdir():z.write(p,p.name)
print(json.dumps(dict(status=report['status'],result_sha256=sha(OUT/'所有固定模型VAL_TEST五项实际对齐结果.json'),seal_sha256=sha(new/'completed_available_results_seal.zip'))))
