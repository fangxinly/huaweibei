"""Compact current state update using already obtained actual receipts only."""
import argparse,json,shutil,zipfile,hashlib
from pathlib import Path
base=Path(__file__).resolve().parents[1];ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);p.add_argument('--phase',choices=['peer_pending','recovery_running'],required=True);a=p.parse_args()
def read(f):return json.loads(f.read_text(encoding='utf-8-sig'))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
statefile=base/'outputs/自主优化实际接续.json';q=read(statefile);pair=q['polarity_intensity_pair_v2'];snapshot={}
for node in ('A','B'):
 cap=read(ev/(node+'_interrupted_capture.json'));pub=read(ev/(node+'_interrupted_publication.json'));client=read(ev/(node+'_interrupted_publication_client_exit.json'));cost=read(ev/(node+'_interrupted_compute_cost.json'))
 assert cap['capture_process_natural_exit']==client['natural_exit']==0 and pub['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED'
 row=dict(interrupted_capture=cap,Release_publication=pub,recorded_extra_replayed_updates=cost['original_completed_extra_replayed_updates'],original_last_recorded_step=cost['original_last_recorded_step'],no_new_final_VAL_TEST_five=True)
 for suffix in ('interrupted_peer400_dispatch.json','interrupted_peer400_preflight.json'):
  if (ev/(node+'_'+suffix)).exists():row[suffix]=read(ev/(node+'_'+suffix))
 if a.phase=='recovery_running':
  for suffix in ('interrupted_peer400_seekfix_result.json','interrupted_peer400_seekfix_capture.json','recovery400_actual_dispatch.json','recovery400_fresh_preflight.json','recovery400_restore.json','recovery400_progress.json','recovery400_detached_dispatch.json'):
   row[suffix]=read(ev/(node+'_'+suffix))
  assert row['interrupted_peer400_seekfix_capture.json']['natural_exit']==0 and row['recovery400_progress.json']['epoch']>=11
 snapshot[node]=row
pair['recovery400_actual']=snapshot;pair['full100_complete']=False;pair['no_new_final_VAL_TEST_five']=True;pair['fixed_algorithm_order_selection_unchanged']=True
q.update(status='MATCHED_PAIR_DETACHED_RECOVERY400_RUNNING' if a.phase=='recovery_running' else 'MATCHED_PAIR_INTERRUPTED_ORIGINAL_RELEASE_COMPLETE_PEER400_CPU_RUNNING',actualclock_before_snapshot_UTC=a.stamp,updated_at_utc=a.stamp,github_source=read(ev/'candidate_continuation_GitHub_receipt.json'),next_gate='Complete original peer400 CPU -> detached exact recovery400; native400->440 both nodes passed. After real100 natural0 complete original Release + peer full4000 CPU -> fixed selected official VAL/TEST prediction save -> all5 once + independent NPZ audit. No TEST structure selection or INNER acceptance.',goal_complete=False,formal_all_five_exceeded=False)
q['current_capacity']={str(n):shutil.disk_usage(str(n)+':/').free for n in ('C','D')};statefile.write_text(json.dumps(q,ensure_ascii=False,indent=2),encoding='utf8')
short=(a.stamp+' 两臂中断完整原件actual A00:09:51/B00:09:39 SHA/CRC/唯一全成员通过，Release两分片各actual00:16:36/42自然0完成。原最后记录A3893/B3873步，恢复需重放3493/3473条已记录更新，未知未落盘末更新不排除；不冒同预算额外算力为0。400→440两模式两节点合成自然0与D小原件核过。'+('新跨节点公开下载完整400状态CPU执行中，未冒通过；真实恢复未启动。' if a.phase=='peer_pending' else '跨节点完整400状态CPU自然0通过；新脱离SSH恢复实际训练已开始，11轮原参数/预测SHA与旧轨迹完全一致；每10轮原子可变完整状态。')+' 新最终VAL/TEST五项未产生，不择TEST改结构。GitHub源码main '+q['github_source']['commit'][:8]+'；旧冻源不删、租期保守Oct9UTC15:00+2h保存门保持。整体未完成。\n\n')
status=base/'outputs/研究接续状态.md';status.write_text(short+status.read_text(encoding='utf8'),encoding='utf8')
(base/'NEXT_EXECUTION.md').write_text('# 当前接续\n\n'+short+'\n只读取研究接续状态最新短段、自主优化JSON及交流JSON。新会话SSH A42487/B4798，SFTP A72818/B55421均idle；旧53451/61821/63220/19718 Unknown process永久禁复用。当前真实执行根及PID见自主优化JSON。未完整400原件公开保存+异节点CPU不得开始恢复；完成后必须保存100全状态及原CPU审核再计算官方VAL/TEST五项。已完成MSE100、201/FIT232/reference100/232禁止重跑。\n',encoding='utf8')
(ev/('current_'+a.phase+'_snapshot.json')).write_text(json.dumps(dict(actualclock_UTC=a.stamp,status=q['status'],nodes=snapshot,capacity=q['current_capacity']),indent=2),encoding='utf8')
print(json.dumps(dict(status=q['status'],capacity=q['current_capacity']),ensure_ascii=True))
