"""Record this turn's real recovery and independent review, without a TRAIN claim."""
import ast, hashlib, json, shutil, sys, zipfile
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
stamp = sys.argv[1]
ROOT = BASE/'outputs/autonomous_followup_20261008T175110Z'
ROOT.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
write = lambda p, value: Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
review = Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第二十次独立审视_MSE健康预检与消息增量识别.md')
assert sha(review)=='e2c25e87bbd0c17e37e67b424547783b614d7067c342b6f61f5777642e4062fe'
shutil.copyfile(review,ROOT/review.name)
helper=BASE/'work/training_health_components_v2.py'
ast.parse(helper.read_text(encoding='utf-8'))
shutil.copyfile(helper,ROOT/helper.name)
capture=BASE/'outputs/fourth_public_restore_actual_20261008T174420Z'
receipt=json.loads((capture/'local_verification.json').read_text())
assert receipt['archive_CRC_unique_member_SHA'] and receipt['launcher_exit']==0
agreement=dict(review_number=20,report=str(review),report_SHA=sha(review),full_report_read=True,
  delivery='authorized advice chat event plus independent local file read; read_thread remained on historical failed19',
  accepted=['fixed true TRAIN health precheck before performance claims','task versus context gradient decomposition','actual updates compared to rounded pure decay','detach and Adam caveats','fixed official VAL selection/all five VAL TEST metrics'],
  deferred=['frozen-teacher donor increment without trainable gain','matched non-flow and simple calibration controls','module LR or initialization changes after genuine TRAIN diagnosis'],
  rejected=['claiming first-step zeros imply permanent inactivity','claiming MSE alone fixes learning','claiming proposed frozen-teacher experiment already implemented or proved'],
  actual_new_training=False)
write(ROOT/'independent_review_decision.json',agreement)
proposal=dict(status='PENDING_HUMAN_EXPLICIT_REPLACEMENT_OF_D_COMPLETE_STATE_REQUIREMENT',
  original_rule='Complete states to D and independent node CPU; frozen original deletion forbidden',
  alternative=['Remote complete model/selected model/Adam/scheduler/RNG/orders and complete source/argv/exit capture',
               'Other currently authorized node independently validates complete state and original files on CPU',
               'Complete archive: SHA256, ZIP CRC, unique member paths and every original member SHA',
               'Upload bounded parts to experiment-specific GitHub Release; verify server digest and original stream SHA',
               'Keep source, protocol, receipts, full logs and frozen predictions on C; use large SHA refs for earlier preserved assets'],
  approval_received=False,execution_enabled=False,remote_lease_end='2026-10-09T15:00:00+00:00',
  lease_end_platform_confirmed=False,minimum_saving_margin_seconds=7200,
  small_local_evidence_capacity_must_be_rechecked=True,new_release_upload_not_started=True)
write(ROOT/'storage_alternative_proposal.json',proposal)
state=dict(status='A_PUBLIC_RUNTIME_RESTORED_REVIEW20_READ_TRAIN_BLOCKED_BY_UNRESOLVED_PRESERVATION',
  actualclock_before_snapshot_UTC=stamp,report=str(ROOT/'independent_review_decision.json'),
  A_public_restore_receipt=str(capture/'capture_receipt.json'),A_restore_archive_SHA=receipt['remote_receipt']['sha256'],
  A_restore_child_natural_exit=0,A_runtime_assets_C_verified=True,D_new_capture_saved=False,
  A_SSH_session=88736,A_SFTP_session=93872,
  client_closure=dict(SSH_logout_and_connection_closed_observed=True,SFTP_bye_observed=True,
                      client_process_exit_codes_not_returned_by_tool=True,old_sessions_must_not_be_reused=True),
  current_capacity={n:dict(free=shutil.disk_usage(n+':/').free) for n in ('C','D')},
  backup_capacity_solved=False,pip_cleanup=dict(automatic_approval='REJECTED',reason='blocked by policy',deleted=False,bypass_attempted=False),
  compressed_old_state_content_SHA_still_unknown=True,old_frozen_files_deleted=False,
  storage_alternative=str(ROOT/'storage_alternative_proposal.json'),approval_pending=True,
  advice_review=agreement,health_v2=dict(source=str(helper),SHA256=sha(helper),AST_pass=True,native_execution_passed=False),
  full_encoder_16_step_entrypoint_frozen=False,actual_new_TRAIN_started=False,new_VAL_TEST_scores=False,
  official_current_MAE=dict(VAL=.5972565370388166,TEST=.6506160153742254,CaReFlow_TEST=.619535293923368),
  automation=dict(id='automation',existing_task_resumed=True,status='ACTIVE',interval_minutes=10,new_task_created=False,latest_lease_prompt_updated=True),
  next_gate='Human storage disposition; real preservation capacity; qualify v2 and fixed 16-step entrypoint; fresh UUID/source/assets/budget before GPU',
  goal_complete=False)
write(ROOT/'actual_followup_receipt.json',state)
write(BASE/'outputs/自主优化实际接续.json',state)
exchange=BASE/'outputs/研究建议交流接续.json'
old=exchange.read_bytes();(ROOT/'original_exchange_state.json').write_bytes(old)
j=json.loads(old.decode('utf-8-sig'));j['review20_actual']=agreement
j['review20_actualclock_recorded_UTC']=stamp
write(exchange,j)
for n in ('actual_followup_receipt.json','storage_alternative_proposal.json','independent_review_decision.json'):
 assert json.loads((ROOT/n).read_text(encoding='utf-8'))
status=BASE/'outputs/研究接续状态.md'
prefix=(stamp+' 自主优化实际接续：人类再次授权持续优化与建议窗口讨论；第20完整报告已读/SHA核过，独立采纳真实TRAIN任务/惩罚梯度与扣衰减更新检查，教师零增量暂缓。新health_v2仅AST未执行，16步入口/完整编码器资格未冻，未启动新训练/VALTEST。A公共assets+私有runtime恢复child629自然0，完整capture SHA0b96331342cf8a4da741f3cd976346db2446b3edbfb2c850533b0910f2c8ecb6，C完整CRC/unique/memberSHA通过；D满无新D保存。SSH88736 logout/closed、SFTP93872 bye已观察，工具未回显客户端exit码不冒0，旧ID禁复用。缓存清理自动审批blocked by policy拒绝，未删除或绕过。D0/C不足1GB、旧压缩状态SHA仍未知；GitHub Release+异节点CPU替代D大状态方案等待人类明确回复，没回复不当批准。原automation从PAUSED恢复ACTIVE，原10min不改，最新第四租期/门槛接续替换旧过期服务器指令，未建新任务。既有官方VAL .5972565/TEST .6506160、CaReFlow TEST .6195353不变；正式五项超过、整体研究/全数据五折未完成。详见《自主优化实际接续.json》。\n\n')
tmp=status.with_suffix('.new.tmp');tmp.write_text(prefix+status.read_text(encoding='utf-8-sig'),encoding='utf-8');tmp.replace(status)
members={p.name:sha(p) for p in ROOT.iterdir() if p.is_file()}
write(ROOT/'member_SHA.json',members)
with zipfile.ZipFile(ROOT/'complete_local_followup.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in ROOT.iterdir():
  if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
with zipfile.ZipFile(ROOT/'complete_local_followup.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
print(json.dumps(dict(receipt=str(ROOT/'actual_followup_receipt.json'),archive_SHA=sha(ROOT/'complete_local_followup.zip'),D_saved=False,new_training=False)))
