"""Preserve actual prefix/peer/native/dispatch evidence without touching frozen originals."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path

base=Path(__file__).resolve().parents[1]
ev=base/'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8')
def seal(tag,stamp):
 snapshot=base/'work'/('candidate_pair_actual_resume_snapshot_'+tag);snapshot.mkdir()
 arms={}
 for n,m in [('A','factorized_aux'),('B','regression_aux')]:
  prefix=read(ev/(n+'_prefix16_result.json'));dispatch=read(ev/(n+'_resume100_dispatch.json'));pre=read(ev/(n+'_resume100_preflight.json'))
  restore=read(ev/(n+'_resume100_restore.json'));progress=read(ev/(n+'_resume100_progress.json'))
  assert restore['input_SHA']==prefix['checkpoint_SHA'] and restore['restored_optimizer_steps']==16 and restore['first16_counted_in4000'] and restore['full_RNG_restored']
  assert progress['steps']>=40
  arm=dict(mode=m,remote_root='/data/coding/'+n+'_polarity_intensity_'+m+'_resume100_actual_20261008T205936Z',dispatch=dispatch,fresh_preflight=pre,complete_prefix_restore=restore,latest_actual_progress=progress,real100_started=True,full100_complete=False)
  arms[n]=arm
  for suffix in ('resume100_dispatch.json','resume100_preflight.json','resume100_restore.json','resume100_progress.json'):
   shutil.copy2(ev/(n+'_'+suffix),snapshot/(n+'_'+suffix))
 for f in ev.iterdir():
  if f.is_file() and f.suffix in ('.json','.zip','.jsonl','.npy') and f.name!='pair_qualified_small_Release_plan.json':shutil.copy2(f,snapshot/f.name)
 for n,m in [('A','factorized_aux'),('B','regression_aux')]:
  q=base/'work'/('polarity_intensity_'+m+'_resume100_qualified_20261008T205936Z')
  for name in ('qualified_resume_plan.json','qualified_resume_source.zip'):shutil.copy2(q/name,snapshot/(n+'_'+name))
 files=[f for f in snapshot.iterdir() if f.is_file()];members={f.name:sha(f) for f in files};write(snapshot/'member_SHA.json',members)
 archive=snapshot/'complete_qualified_small_originals.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for f in files+[snapshot/'member_SHA.json']:z.write(f,f.name)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 assert archive.stat().st_size<40*1024**2
 dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/snapshot.name;dest.mkdir()
 target=dest/archive.name;shutil.copy2(archive,target);assert sha(target)==sha(archive)
 receipt=dict(actualclock_UTC=stamp,status='BOTH_REAL_RESUME100_STARTED_PREFIX16_COUNTED_ONCE_SMALL_ORIGINALS_C_D_VERIFIED',arms=arms,archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,archive=str(archive),D_original=str(target),full_prefix_primary_Release_and_other_node_originals=True,no_full_prefix_D_claim=True,no_old_deletion=True)
 write(snapshot/'actual_receipt.json',receipt)
 state=read(base/'outputs/自主优化实际接续.json');state['status']='POLARITY_INTENSITY_MATCHED_PAIR_REAL_100_RUNNING';state['actualclock_before_snapshot_UTC']=stamp
 pair=state['polarity_intensity_pair_v2'];pair.update(status='BOTH_PREFIX16_RELEASE_OTHER_NODE_CPU_PASSED_REAL_RESUME100_RUNNING',actualclock_UTC=stamp,full100_started=True,full100_complete=False,peer_CPU_complete=True,resume_native_both_modes_two_nodes_natural0=True,resume100_arms=arms,small_evidence_preservation=receipt)
 state['current_execution']=dict(kind='polarity_intensity_matched_pair_resume100',actualclock_UTC=stamp,arms=arms,TEST_not_training_or_selection=True)
 state['next_gate']='Healthy fixed100 continuation; capture400/complete originals -> Release + other-node CPU -> selected official VAL/TEST predictions freeze/preserve -> same-checkpoint five once per arm. No TEST method selection.'
 state['goal_complete']=False;state['formal_all_five_exceeded']=False
 state['sessions_current'].update(A_SSH=53451,A_SFTP=63220,B_SSH=61821,B_SFTP=19718,A_SSH_foreground_waiting_child=True,B_SSH_foreground_waiting_child=True)
 write(base/'outputs/自主优化实际接续.json',state)
 short=stamp+' 原流极性强度/同容量辅助回归两臂真实100续训已开始，完整prefix16自然0/Release两份原件+交叉公开下载CPU345Adam16/同流ONOFF通过；两节点续训CPU合同前16到40状态/Adam/sched/RNG一致自然0，实际GPU完整恢复16计一次/4000，当前A/B进度见自主优化JSON。相同初始状态/订单/预算，官方VAL128/101批MSE选模，TEST不选结构。小原件C/D完整SHA CRC唯一过，完整前缀Release+异节点原件保留，无全前缀D声明/无旧删除。A53451/B61821 foreground健康续训，SFTP63220/19718有效，禁额外SSH命令。完成/400保存->Release+原CPU->官方VALTEST同checkpoint全五项。MSE100既有TEST MAE.652712 vs CaReFlow .619535未胜；新候选尚无最终五项，整体/正式超过/全数据五折未完成。\n\n'
 note=base/'outputs/研究接续状态.md';note.write_text(short+note.read_text(encoding='utf8'),encoding='utf8')
 (base/'outputs/NEXT_EXECUTION.md').write_text(short,encoding='utf8')
 report=base/'outputs/极性强度匹配两臂实际续训.md'
 report.write_text('# 原流极性与强度升级：实际匹配续训\n\n'+short+'两臂都保留原流与六方向消息。factorized_aux 显式输出极性×非负强度；regression_aux 是同容量、同辅助监督的回归对照。新增2002参数，固定Huber主损失、强度回归与符号辅助监督。此候选是普通监督结构升级，尚不构成折外效用控制或共享/互补语义证明。\n\n'+ '\n'.join(n+': child '+str(a['dispatch']['pid'])+'，完整恢复16步，实际已到 '+str(a['latest_actual_progress']['steps'])+' 步。' for n,a in arms.items())+'\n\n最终只用官方VAL选定同一checkpoint，分别报告VAL与TEST的Acc7/Acc2/F1/MAE/Corr及CaReFlow对照。TEST已经看过，不能称新盲测，也不用于选结构。\n',encoding='utf8')
 pub=read(base/'work/publish_polarity_intensity_A_prefix16_plan.json');pub['assets']=[dict(path='/data/coding/pair_qualified_small_originals_'+tag+'.zip',name='polarity-intensity-qualified-prefix-peer-resume-'+tag+'-originals.zip',bytes=archive.stat().st_size,sha256=sha(archive))]
 write(snapshot/'Release_plan.json',pub)
 print(json.dumps(dict(snapshot=str(snapshot),archive=str(archive),SHA=sha(archive),bytes=archive.stat().st_size,real100_started=True)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args();seal(a.tag,a.stamp)
