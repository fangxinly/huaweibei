"""Short continuation from actual immutable observations; no remote claims invented."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

clock=sys.argv[1];outputs=Path('outputs');pilot=Path('D:/CodexBackups/selective_flow_20261003_1105/anchored_flow_pilot40_20261007T091746Z')
dynamic=Path('D:/CodexBackups/selective_flow_20261003_1105/lease_dynamic_actual_20261007T093008Z')
obs=pilot/'actual_live_observation_20261007T093008Z.json'
if sha(obs)!='f541104e2e5ad6ba593ddee56615f1ca5c3aec1483a06c959f10941bf0f046a0':raise ValueError('Actual live observation differs')
live=json.loads(obs.read_text());joint=dynamic/'actual_D_dynamic_joint.json'
if sha(joint)!='ba3a81fa4313cc1986be3101aac1a53f3ae7461ee50984d85f143df7e4ef54e5':raise ValueError('Actual snapshot proof differs')
state={'status':'ANCHORED_FLOW_FIXED_PILOT40_ACTUALLY_RUNNING_NOT_COMPLETE','actualclock_local_record_utc':clock,
    'actual_last_live_observation_utc':live['actual_utc'],'actual_completed_epochs':live['actual_completed_epoch'],
    'actual_step_prefix':live['actual_step_prefix'],'child_pid':live['pid'],'fullargv':live['fullargv'],'uuid':live['uuid'],
    'stderr_Traceback':live['stderr_has_Traceback'],'natural_exit_exists_at_observation':live['natural_exit_exists'],
    'live_observation':{'path':str(obs),'sha256':sha(obs)},'D':str(pilot),
    'protocol':{'path':str(pilot/'bundle/pilot40_execution_protocol.json'),'sha256':sha(pilot/'bundle/pilot40_execution_protocol.json')},
    'scope':{'method':'anchored_flow','fold':0,'epochs':40,'updates':1880,'fit':1494,'inner':264,'outer':437,
             'only_INNER_MSE_selects':'strictmin earliest','full_fivefold_complete':False,'CaReFlow_matched_comparison_performed':False,
             'public_encoder_new_initialization':True,'precheck_or_old_task_weights_reused':False,'OUTER_labels_scored':False,
             'pooled_pickle_materializes_original_label_objects':True,'only_FIT_training_supervision':True},
    'latest_progress_in_development_not_final_score':live['progress'],
    'diagnosis_report':{'path':str((outputs/'TRAIN_DEV退化定位与下一步训练方案.md').resolve()),'sha256':sha(outputs/'TRAIN_DEV退化定位与下一步训练方案.md')},
    'real_0930_dynamic_save':{'path':str(joint),'sha256':sha(joint),'whole_new_prefix_D_bytes':2966418867,
        'whole_new_prefix_SHA':'6f14aca10622edd0007c6c77ecad2355e122fd1c945dfc94de0271316290d1552',
        'prefix_not_training_completion':True,'B_CPU_audit_of_new_prefix_or_final_not_completed':True},
    'remaining_final_state_max_reservation_bytes':4000000000,'actual_D_free_bytes':shutil.disk_usage('D:/').free,
    'conservative_lease_end_UTC':'2026-10-07T13:30:00+00:00','not_platform_lease_confirmation':True,
    'remaining_dynamic_save_windows_UTC':['11:30','13:00'],
    'closed_interactive_session_ids_no_reuse':[47434,75135,35317,76271,30442,40926],
    'all_six_current_connections_explicit_exit_or_bye_natural_zero':True,'detached_training_continues':True,
    'original_failures_preserved':{'first_operator_relative_parent_path':{'natural_exit_observed':1,'training_child_not_started':True,
        'limited_cwd_operator_supplement_source':str(pilot/'pilot40_start_operator_cwd_v2.py'),'training_frozen_bundle_changed':False},
        'unused_dependency_delete_rejected':{'reason':'blocked by policy','deletion_retried':False,'dependency_directory_not_deleted':True}},
    'next':['Do not redispatch or stop healthy A child4695. Fresh original connection only after real password prompt using original human credentials.',
        'After actual natural0: frozen fold_evidence capture using fresh existing-parent output root, complete final full.pt D SHA/CRC/unique.',
        'Transfer exact final original/fullsource/protocol to B; natural fold_cpu_audit then original capture and D join; no CPU model forward claim.',
        'Report all five of the single INNER-MSE-selected checkpoint. Development only, no OUTER or TEST rescue selection.',
        'Full100/ten-model fivefold remains unexecuted; new complete storage and whole queue+2h budget required.',
        'Next actual dynamic saves11:30 and13:00UTC, do not backfill timestamps or static source into remote capture.'],
    'overall_research_five_metric_superiority_and_full_fivefold_complete':False}
write(outputs/'新方案40轮实际训练接续.json',state)
md=f'''新方案固定40轮已实际运行，尚未完成。

原A child{live['pid']}，实际观察{live['actual_utc']}：完成{live['actual_completed_epoch']}/40轮、物理更新前缀{live['actual_step_prefix']}/1880，stderr无Traceback、natural_exit尚不存在。原公共编码器重新初始化，未继承预检或旧任务权重。仅第0折FIT1494训练、INNER264按MSE strictmin earliest选模；OUTER437未评分。合并pickle整库已解码，标签访问角色守卫不等于原TEST对象未解码。

UTC09:30三机真实动态保存完成D：A/B/C capture实际09:30:55/21/27；joint ba3a81fa4313cc1986be3101aac1a53f3ae7461ee50984d85f143df7e4ef54e5，A新第10轮完整状态2966418867B SHA6f14aca10622edd0007c6c77ecad2355e122fd1c945dfc94de0271316290d1552已D全SHA/ZIPCRC/唯一，原旧完整大件只SHArefs不重传。这是运行前缀和实际动态快照，非40完成、最终新原CPU或五折成绩。

单次冻结plan82d2de9439766b2f981f92703d1c34ac6b30c2ef52911e603bf71c22c2a7330a，源D {pilot}。第一operator相对父路径错误自然1，未启动训练；独立cwd修正启动器保持训练冻结源不变。依赖目录清理被策略拒绝，没有重试或删除；当时按现有D核算单次整状态4GB、额外6GB保存余量；额外余量已部分用于人类指定09:30动态整前缀，现D余量{state['actual_D_free_bytes']}B仍保留最终整状态4GB预算。

6个当前SSH/SFTP均明确exit/bye自然0关闭，ID47434/75135/35317/76271/30442/40926禁复用；detached健康训练继续。完成立即完整D及异节点B原TorchCPU/capture/joint，再报告固定selected INNER五项，不能称匹配CaReFlow比较。下一11:30/13:00租期保存、正式全五项超过和完整五折仍未完成。
'''
(outputs/'新方案40轮实际训练接续.md').write_text(md,encoding='utf-8')
headline=f'''最新实测{live['actual_utc']}：新anchored_flow固定第0折40轮已A child4695实际运行，完成{live['actual_completed_epoch']}/40轮、物理前缀{live['actual_step_prefix']}/1880，原source/fullargv/UUID及初态门过，stderr无Traceback/natural_exit未存在，严禁重派/停健康。先读《新方案40轮实际训练接续.md/json》《TRAIN_DEV退化定位与下一步训练方案.md》。计划82d2de...保持既定loss/gain/source不变，仅固定40预算；不是100或完整五折。UTC09:30三机真实动态保存D joint ba3a81fa...，新第10轮2966418867B完整状态D SHA/ZIPCRC/唯一过，旧大件SHArefs未重传；新最终CPU/40完成/新selected五项仍待，OUTER不评分。6会话47434/75135/35317/76271/30442/40926全exit/bye0关闭禁复用，原人类凭据不索重复。初operator相对路径natural1保留、cwd独立修正未改冻结训练源；依赖删除策略拒绝未重试/未删。D现{state['actual_D_free_bytes']}B保留最终整状态4GB；09:30已完成，11:30/13:00仍待，整体正式超过/完整五折未完成。以下全部旧状态为历史，旧“训练0”不代表最新单次40。

'''
status=outputs/'研究接续状态.md';status.write_text(headline+status.read_text(encoding='utf-8-sig'),encoding='utf-8')
for name in ['新方案40轮实际训练接续.md','新方案40轮实际训练接续.json','研究接续状态.md']:
    shutil.copy2(outputs/name,pilot/name)
seal=pilot/'started_pilot_operator_seal';seal.mkdir()
for p in [pilot/'actual_live_observation_20261007T093008Z.json',pilot/'actual_pilot_source_freeze_receipt.json',
          pilot/'actual_local_pilot_capacity.json',joint,outputs/'新方案40轮实际训练接续.json',outputs/'TRAIN_DEV退化定位与下一步训练方案.md']:
    shutil.copy2(p,seal/p.name)
write(seal/'large_frozen_original_refs.json',{'new_prefix_path':str(dynamic/'A_frozen_prefix_full.pt'),
    'new_prefix_sha256':'6f14aca10622edd0007c6c77ecad2355e122fd1c945dfc94de0271316290d1552','new_prefix_bytes':2966418867,
    'large_refs_are_not_repeated_downloads':True,'not_a_new_remote_CAPTURE':True})
archive=pilot/'started_pilot_operator_seal.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(seal.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(archive) as z:
    if z.testzip() or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Local operator seal failed')
print(json.dumps(dict(status=state['status'],continuation_sha256=sha(outputs/'新方案40轮实际训练接续.json'),
    local_operator_seal_sha256=sha(archive),actual_prefix_epochs=live['actual_completed_epoch'],fivefold_complete=False),ensure_ascii=False))
