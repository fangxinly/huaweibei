"""Metadata-only future development roles, no predictions or task labels."""
import datetime,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];parent=root/'work/second_lease_fixed_execution_v1_20261006T1349Z'
utc=datetime.datetime.now(datetime.timezone.utc);stamp=utc.strftime('%Y%m%dT%H%M%SZ')
folder=root/'work'/('future_fold0_head_role_reservation_'+stamp);folder.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mapping=json.loads((parent/'train_row_video_mapping.json').read_text(encoding='utf-8'))
ids={key:np.load(parent/(key+'_0.npy'),allow_pickle=False) for key in ('fit','inner','outer')}
vids=[x['video_id'] for x in mapping]
assert len(vids)==1281 and len(ids['outer'])==433
videos=sorted({vids[int(i)] for i in ids['outer']},key=lambda v:hashlib.sha256(('MINIMAL_FIXED_FOLD0_HEAD_ROLE_RESERVATION_V1_20261006:'+v).encode()).hexdigest())
assert len(videos)==18
head_train_videos=videos[:9];head_eval_videos=videos[9:]
head_train=np.asarray([i for i in ids['outer'] if vids[int(i)] in head_train_videos],dtype=np.int64)
head_eval=np.asarray([i for i in ids['outer'] if vids[int(i)] in head_eval_videos],dtype=np.int64)
assert set(head_train)|set(head_eval)==set(ids['outer']) and not set(head_train)&set(head_eval)
assert not (set(head_train)|set(head_eval))&(set(ids['fit'])|set(ids['inner']))
np.save(folder/'head_development_fit_0.npy',head_train);np.save(folder/'head_development_eval_0.npy',head_eval)
plan={'status':'FUTURE_HEAD_ROLE_RESERVATION_ONLY_NO_HEAD_EXPERIMENT_AUTHORIZATION_GATE','actual_frozen_utc':utc.isoformat(),'reference_runtime_plan_sha256':sha(parent/'runtime_candidate_plan.json'),'mapping_metadata_sha256':sha(parent/'train_row_video_mapping.json'),'original_reference_roles':{r:{'rows':len(a),'videos':len({vids[int(i)] for i in a}),'role_array_sha256':sha(parent/(r+'_0.npy'))} for r,a in ids.items()},'rule':'SHA256 salt:video sorted; first9 head FIT, last9 development EVAL. No head selection role, no labels/predictions/Q used.','head_development_fit':{'videos':head_train_videos,'rows':len(head_train),'array_sha256':sha(folder/'head_development_fit_0.npy')},'head_development_eval':{'videos':head_eval_videos,'rows':len(head_eval),'array_sha256':sha(folder/'head_development_eval_0.npy')},'head_hyperparameter_selection_on_eval_allowed':False,'reference_FIT_INNER_unchanged':True,'precheck_OUTER_still_disabled':True,'past_T0_and_old_allTRAIN_exploration_retained':True,'fresh_confirmation_or_whole_pipeline_crossfit':False,'reference_training_already_complete':False,'head_architecture_candidate_budget_and_stop_protocol_complete':False,'actual_task_label_or_prediction_read_this_freeze':False,'actual_GPU':False}
(folder/'head_role_reservation_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
report='固定fold0未来头角色预留\n\nUTC '+utc.isoformat()+f'''。仅原row/video元数据与role整数数组，未读官方输入/标签/预测/Q。参考FIT695/30和INNER153/4不动；原OUTER433/18按预先盐SHA视频顺序9/9预留head FIT {len(head_train)}行、开发EVAL {len(head_eval)}行，没有head选择组，不在EVAL挑岭、权重、幅度、候选或配方。该批OUTER存在旧T0/全TRAIN开发历史，保留而非新确认或全流程crossfit。

这是未来角色预留，未完成头输入/结构/合法candidate/成本/停止协议，不授权直接启动头实验。新参考预检OUTER仍完全禁用；未来reference预测先冻结，头只在自己的9视频FIT训练，固定对照再于另9视频开发测量。没有新模型、GPU、成绩或真实标签拟合。正式参考100前把本预留与最终候选科学协议一致联结，若研究问题需另方案，先重新记录元数据协议及理由，不能据成绩重划角色。原角色、科学源与标签守卫不改。
'''
(root/'outputs/固定fold0未来头角色预留接续.md').write_text(report,encoding='utf-8')
base=Path('D:/CodexBackups/selective_flow_20261003_1105').resolve();assert shutil.disk_usage(base).free>6*1024**3
dest=(base/('future_fold0_head_role_preparation_'+stamp)).resolve();assert dest.is_relative_to(base);dest.mkdir(exist_ok=False)
members={}
for p in list(folder.glob('*'))+[Path(__file__),root/'outputs/固定fold0未来头角色预留接续.md']:
 q=dest/p.name;shutil.copy2(p,q);members[p.name]={'bytes':q.stat().st_size,'sha256':sha(q)}
(dest/'member_manifest.json').write_text(json.dumps(members,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with zipfile.ZipFile(dest/'original_package.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in list(members)+['member_manifest.json']:z.write(dest/n,n)
with zipfile.ZipFile(dest/'original_package.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
receipt={'actual_utc':utc.isoformat(),'permanent_D':str(dest),'plan_sha256':sha(folder/'head_role_reservation_plan.json'),'archive_sha256':sha(dest/'original_package.zip'),'full_SHA_CRC_member_verified':True,'official_task_data_labels_or_scores_read':False,'actual_GPU':False,'formal100_or_head_started':False,'role_rows':[len(head_train),len(head_eval)]}
(dest/'preservation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');(root/'outputs/固定fold0未来头角色预留最新.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
