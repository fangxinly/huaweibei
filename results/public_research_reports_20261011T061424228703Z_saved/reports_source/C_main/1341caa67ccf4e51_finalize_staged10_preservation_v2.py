"""Join actual original GPU/D/other-node CPU evidence; preserve short state."""
import datetime,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference_first10_actual_20261006T151954Z')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
local=read(BASE/'local_D_joint_audit_v2.json');cpu=read(BASE/'B_full_state_CPU_original_receipt.json');ex=read(BASE/'B_full_state_CPU_original_exit.json')
bcap=read(BASE/'B_capture_local_joint_audit.json')
assert bcap['status']=='ACTUAL_B_FULL_CPU_ROOT_CAPTURE_SHA_CRC_UNIQUE_AND_D_WHOLE_FILE_ASSOCIATION_PASSED'
raw=BASE/'a/original_small_files/run';r=read(raw/'out/actual_training_receipt.json')
assert ex['exit_code']==0 and ex['natural_wait_verified'] and ex['child_pid']==cpu['pid']
assert ex['child_full_argv'][0]=='/data/coding/multimodal_flow_public_20261006T1341Z/.venv/bin/python' and ex['child_full_argv'][1:]==cpu['argv']
assert cpu['source_sha256']==sha(BASE/'a/original_small_files/source/audit_staged_reference_CPU_v1.py')
assert cpu['original_training_receipt_sha256']==sha(raw/'out/actual_training_receipt.json')==local['original_training_receipt_sha256']
assert cpu['original_natural_exit_sha256']==sha(raw/'natural_exit.json') and cpu['steps']==220 and not cpu['GPU_used'] and not cpu['CPU_model_forward']
for n in ('complete_resume_full.pt','selected_best_full.pt'):
 f=BASE/'a'/n;assert sha(f)==local['full_files'][n]['sha256']==cpu['states'][n]['file_sha256']
 assert f.stat().st_size==local['full_files'][n]['bytes'] and cpu['states'][n]['tensors']==374
assert cpu['complete_Adam_parameter_states']==364
now=datetime.datetime.now(datetime.timezone.utc);assert shutil.disk_usage('D:/').free>12*1024**3
dest=BASE/('research_records_'+now.strftime('%Y%m%dT%H%M%SZ'));dest.mkdir(exist_ok=False)
for n in ('研究接续状态.md','研究建议交流接续.json','完整流单参考分段训练实际接续.json','完整流单参考分段训练实际接续.md'):
 q=dest/'preceding_state'/n;q.parent.mkdir(exist_ok=True);shutil.copy2(OUT/n,q)
joint={'status':'ACTUAL_SHARED10_GPU_D_B_CPU_NEXT_UPDATE_AND_FRESH_REPLAY_JOINT_PASSED',
 'actual_utc':now.isoformat(),'run_root':local['original_run'],'source_root':local['original_source'],
 'plan_sha256':r['plan_sha256'],'epochs':10,'formal_updates':220,
 'resume_full_sha256':local['full_files']['complete_resume_full.pt']['sha256'],
 'selected_best_full_sha256':local['full_files']['selected_best_full.pt']['sha256'],
 'local_D_audit_sha256':sha(BASE/'local_D_joint_audit_v2.json'),
 'original_other_node_CPU_receipt_sha256':sha(BASE/'B_full_state_CPU_original_receipt.json'),
 'original_other_node_CPU_exit_sha256':sha(BASE/'B_full_state_CPU_original_exit.json'),
 'original_training_receipt_sha256':sha(raw/'out/actual_training_receipt.json'),
 'original_B_actual_capture_audit':bcap,'original_B_capture_audit_sha256':sha(BASE/'B_capture_local_joint_audit.json'),
 'original_CPU_receipt':cpu,'original_CPU_exit':ex,'GPU_D_local_original_audit':local,
 'CPU_model_forward':False,'formal100_complete':False,'OUTER_used':False,'head_training_started':False,
 'continuation_started':False,'exact_next_update_complete_state_passed':True,'fresh_disk_replay_error':0.0}
(BASE/'complete_stage10_GPU_D_B_CPU_joint_audit.json').write_text(json.dumps(joint,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
record=read(OUT/'完整流单参考分段训练实际接续.json')
record.update({'status':joint['status'],'updated_actual_utc':now.isoformat(),'epochs_completed':10,'formal_updates':220,
 'stage10_natural_exit_utc':read(raw/'natural_exit.json')['actual_exit_utc'],'complete_joint_audit':str(BASE/'complete_stage10_GPU_D_B_CPU_joint_audit.json'),
 'complete_joint_audit_sha256':sha(BASE/'complete_stage10_GPU_D_B_CPU_joint_audit.json'),
 'final_budget':r['final_budget'],'INNER_best_epoch':r['best_epoch'],'INNER_video_MSE_selection_only':r['INNER_video_equal_MSE_selection_only'],
 'next_update_check':r['next_update_continuation_check'],'donor_mechanism':r['donor_mechanism'],
 'formal100_complete':False,'continuation_started':False,'OUTER_or_head_labels_used':False,
 'stage10_CPU_and_D_complete':True})
(OUT/'完整流单参考分段训练实际接续.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc=f'''完整流单参考100计划：前10轮实际完成与完整保存

联合保存实际UTC {now.isoformat()}。A child1173 UTC15:19:19.953311自然exit0，10轮/220次正式更新完整执行，所有正常32及每轮尾23均真实optimizer/scheduler；全185402807参数/364张量，FIT695/30-only监督与统计，INNER153/4每轮视频等权float64MSE选模。源plan {r['plan_sha256']} 和完整100订单/2200日程不改。不是100完成、共享10比较控制结果、OUTER评价或整体crossfit。

第10轮为该前缀earliest best，INNER选择MSE {r['INNER_video_equal_MSE_selection_only']:.12f}；只选模，不冒独立泛化或与旧TRAIN.014直接比较。原逐轮数组预测SHA先写盘，再访问对应INNER标签统计。OUTER/CAL/EVAL/DEV/TEST和新head未使用。

阶段总374.684秒；累计allocated3924471808bytes约3.655GiB/reserved4406116352bytes约4.104GiB，含构造/全量状态存盘/下一update双支路/fresh实例，无中途peak重置，低于原6GiB。每轮训练约19.3秒，INNER前向约.85秒，仅工程预算信息。

完整resume2966997912bytes SHA{joint['resume_full_sha256']}，含last10完整model/Adam364参数两矩及step220/scheduler220/全部真实RNG/全history和累计best。selectedbest741731206bytes SHA{joint['selected_best_full_sha256']}。两份均完整下载D并转存B，非大SHA引用冒完整下载。B独立原TorchCPU对两整文件374状态/完整元素/元数据/FITstats/Adam与RNG及153行数组核过，原PID/argv/自然exit0联结；不是CPU模型前向。原件 {BASE.as_posix()}。

下一batch两孤立update精确比较：同原11轮首32FIT数据，连续与fresh磁盘恢复的完整model/Adam/scheduler/起终RNG以及目标2.354990959完全相同，各一诊断step之后原220状态恢复，正式221未提交；73.18秒。不仅梯度L1相同。自己的完整selectedbest同实例和fresh公共实例strict INNER原输入重放误差0，dummy标签0/7误差0，FITstats不变。

预声明同四FIT行供体置零：feedback.0680423/context改变.0142784/第二Euler状态.000123739/终端scalar改变9.10461e-6/恢复0，3前向2.465秒。初始和原两步终端0证据保留；这里真实终端作用不证明预测收益，不因此增step、择新行或选checkpoint。

A capture25 UTC15:20:09.822294实际COMPLETE，自然exit0/原receipt后53成员ZIP SHA/CRC/唯一/新源/完整argv和两大稳定inode引用通过。B初次snapshot手抄摘要多一字符导致展开前AssertionError；随后用原capture_receipt摘要比对通过。A→B直接SFTP无提示连接取消，未认证/传件；实际完整B原件由D传送并独立重新审核，不回填初次成功。

续11–100待fresh UUID/空compute/fullargv/资产源/remote和D12GiB/原保守时间投影+2h保存核实；从完整last10恢复，不从best或预检两步恢复，不重复epoch10或重置日程。当前未启动续训，整体研究未完成。未来head 232/201视频角色已无标签预留，实际候选/标签/成本协议另冻结，当前仍禁OUTER。
'''
(OUT/'完整流单参考分段训练实际接续.md').write_text(doc,encoding='utf-8')
ledger=read(OUT/'研究建议交流接续.json');ledger['second_lease_new_assets']['staged_reference_training']=record
ledger['scientific_state']='单fixed参考前10实际自然exit0/220正式更新/D两完整state/B原TorchCPU/完整下一update精确续训/fresh重放0全过；INNER选择2.185963仅选模，11-100待fresh门控，不冒100/外折/供体收益。'
ledger['updated_at_utc']=ledger['updated_utc']=now.isoformat()
(OUT/'研究建议交流接续.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=(OUT/'研究接续状态.md').read_text(encoding='utf-8').splitlines();lines[0]=f'更新UTC {now.isoformat()}。继续主动研究/真实优化与第二租期保存，整体未完成。'
for i,s in enumerate(lines):
 if s.startswith('4) '):lines[i]='4) 单fixed参考前10 child1173 UTC15:19:19自然exit0，220真updates/tail23；phase total374.7s/peak allocated3.655 reserved4.104GiB含精确下一update与fresh重放，低于原6GiB。INNER选模best10 MSE2.1859632038非泛化。D/B两完整状态与原CPU/53成员capture25/下一batch完整model-Adam-scheduler-RNG一致/原220恢复/strict153重放0通过。先读完整流单参考分段训练实际接续.md/json。11-100待fresh门控，不能冒100/OUTER/对照训练。供体同四FIT终端9.10e-6作用非收益；完整源计划不改。'
 if s.startswith('9) '):lines[i]='9) 人类明确要求重试后，UTC16:52新A SSH75480已认证/UUID匹配/空compute/原前10exit0和checkpoint存在实核；UTC16:53微型Torch CUDA点积结果11/child exit0无stderr。旧SSH79535已Unknown process，不能冒旧会话exit0。新B SSH62058/SFTP85817真实认证并UUID/空compute/原capture回执核过。恢复是这次实际发送与GPU运算证据，不称底层代码修复。此前UTC16:32审批拒绝原件保留。控制session退出仍须实际核验，不停健康训练；其它旧ID不复用。'
(OUT/'研究接续状态.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
files=[OUT/n for n in ('研究接续状态.md','研究建议交流接续.json','完整流单参考分段训练实际接续.md','完整流单参考分段训练实际接续.json')]
files += [BASE/n for n in ('complete_stage10_GPU_D_B_CPU_joint_audit.json','local_D_joint_audit.json','local_D_joint_audit_v2.json','local_D_joint_audit_v2_original_receipt_before_source_versioning.json','local_audit_version_correction.json','local_audit_exact_source_restoration.json','B_full_state_CPU_original_receipt.json','B_full_state_CPU_original_exit.json')]
files += [Path(__file__),ROOT/'work/audit_staged10_D_local_v1.py',ROOT/'work/audit_staged10_D_local_v2.py',ROOT/'work/audit_staged_reference100_CPU_v2.py',ROOT/'work/preserve_staged10_local_audit_versions_v1.py',ROOT/'work/run_staged_CPU_original_wrapper_v1.py']
files += [BASE/'B_capture_local_joint_audit.json',ROOT/'work/audit_staged10_B_capture_local_v1.py',ROOT/'work/capture_staged_CPU_audit_v1.py',BASE/'B_synthetic_parameter_identity_original.json']
members={}
for p in files:
 n=('outputs/' if p.parent==OUT else 'evidence/')+p.name;q=dest/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);members[n]={'sha256':sha(q),'bytes':q.stat().st_size}
for p in (dest/'preceding_state').iterdir():members[p.relative_to(dest).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
(dest/'member_manifest.json').write_text(json.dumps(members,indent=2)+'\n')
with zipfile.ZipFile(dest/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in list(members)+['member_manifest.json']:z.write(dest/n,n)
with zipfile.ZipFile(dest/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
receipt={'actual_utc':now.isoformat(),'status':joint['status'],'records_dir':str(dest),'records_sha256':sha(dest/'records.zip'),'members':len(members),'SHA_CRC_unique_passed':True,'formal100_complete':False,'continuation_started':False}
(dest/'preservation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
