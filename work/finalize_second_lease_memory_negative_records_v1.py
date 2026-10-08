"""Persist real v4/v5 failures and original other-node audits, no fabricated success."""
import datetime,hashlib,json,shutil,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'outputs';now=datetime.datetime.now(datetime.timezone.utc)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
old=Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_precheck_failure_actual_20261006T141445Z');v4=Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_budget_trace_actual_20261006T143256Z');v5=Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_cache_release_actual_20261006T143905Z')
assert shutil.disk_usage('D:/').free>6*1024**3
cp=read(old/'B_initial_cpu_audit_receipt.json');ce=read(old/'B_initial_cpu_audit_actual_exit.json')
assert ce['CPU_auditor_actual_exit_code']==0 and cp['checkpoint_file_sha256']==sha(old/'a/clean_initial_full.pt') and cp['complete_state_tensors']==374 and cp['public_encoder_exact_matched_tensors']==198
results=[]
for base in (v4,v5):
 a=read(base/'local_independent_audit.json');b=read(base/'B_raw_CPU_audit_receipt.json');ex=read(base/'B_raw_CPU_audit_actual_exit.json');assert ex['exit_code']==0 and ex['natural_wait']
 assert all(a[k]==b[k] for k in ('snapshot_sha256','stage_trace','step_times','initial_full_sha256','initial_CPU_receipt_sha256'))
 assert a['snapshot_sha256']==sha(base/'a/snapshot.zip') and a['initial_full_sha256']==cp['checkpoint_file_sha256']
 results.append({'permanent_D':str(base),'original_local_audit_sha256':sha(base/'local_independent_audit.json'),'original_B_audit_sha256':sha(base/'B_raw_CPU_audit_receipt.json'),'original_B_exit_sha256':sha(base/'B_raw_CPU_audit_actual_exit.json'),'audit':a,'original_other_node_CPU_pass':True,'new_complete_initial_weight_download':False})
record={'actual_utc':now.isoformat(),'new_GPU_scientific_precheck_attempts_real':True,'GPU_precheck_passed':False,'ordinary_v4':results[0],'cache_release_v5':results[1],'initial_full_D_and_B_CPU_complete':True,'initial_full_sha256':cp['checkpoint_file_sha256'],'full_INNER_replay_completed':False,'after_two_steps_checkpoint_available':False,'formal100_started':False,'new_performance_scores':False,'next_candidate_only':'Text nonreentrant activation checkpoint v6, public text source inspected, actual short GPU run in progress; outcome not claimed.'}
archive=v5/('research_records_'+now.strftime('%Y%m%dT%H%M%SZ'));archive.mkdir(exist_ok=False)
for n in ('研究接续状态.md','研究建议交流接续.json','第二租期固定流首次预检实际失败与接续.md','第二租期固定流首次预检失败最新.json'):
 q=archive/'preceding_state'/n;q.parent.mkdir(exist_ok=True);shutil.copy2(out/n,q)
(out/'第二租期显存优化负结果最新.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc=f'''第二租期完整流显存门控与缓存释放实际负结果

实际记录UTC {now.isoformat()}。新三P4公共资产及24包依赖在各节点安装/原exit0核过。旧租期到期限制不延伸到人类新24h授权；正式100仍未启动。

首次v3自然失败后，完整clean initial 741731078bytes已实际D保存，B TorchCPU独立加载374状态张量及198公共编码器张量exact，零初始化六供体末层；不是CPU模型前向或两步后权重。完整初始SHA {cp['checkpoint_file_sha256']}，v4/v5 fresh原大文件SHA均相同，联结现有原件，不冒新下载。

v4 child651 UTC14:31:59自然exit1，24.47s门控时分配峰值5811790848bytes(5.412GiB)，保留6757023744bytes(6.293GiB)>6GiB；不是OOM。原两步目标2.843935728/1.458111644，每步364个保留参数张量有限非None，实际singleton行454/505/620长度1；主辅路径符合断言，零梯度诚实记录。未尾批、完整INNER、后状态权重或formal100。

v5 child806 UTC14:37:59自然exit1。只释放每批之前未使用缓存，模型/目标/容量/init/标签/订单/LR/批32/6GiB门槛不改且不重置历史峰值。两步目标和每个梯度L1范数与v4差恰0(不冒完整梯度逐字节相等)。分配5814365184bytes(5.415GiB)，保留7098859520bytes(6.611GiB)，反而更高。缓存释放不能解决本例门槛，停止此方案，不扫allocator参数或改门槛救通过。

v4 capture21 UTC14:33:04实际CAPTURE_COMPLETE/exit0、79成员ZIP全SHA/CRC/unique/source-role/assets/argv核过，D {v4.as_posix()}，B原stdlibCPU审核exit0已下载联合核过。v5 capture22 UTC14:39:14实际CAPTURE_COMPLETE/exit0、88成员同核，D {v5.as_posix()}，B原CPU审核exit0同样保存。两次初始大SHA引用非重下载，原B全初始权重可用；不是异节点模型前向。所有原失败/source/receipt时间保留。

下一v6已另冻结并真实启动仅public DebertaV2文本编码器非重入激活检查点(use_reentrant=False,preserve_rng_state=True)，源码已在Torch2.1/Transformers4.37.2实际环境查过。新目标值/梯度范数对照、尾23不加step、整新pt/完整INNER153 dummy换标签0/strict重放<=1e-6及全程6GiB/20min仍需实际通过；当前无v6成功结论或新分数。官方原始实现：https://github.com/pytorch/pytorch/blob/v2.1.0/torch/utils/checkpoint.py 和 https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py 。缓存语义：https://github.com/pytorch/pytorch/blob/v2.1.0/torch/cuda/memory.py 。
'''
(out/'第二租期显存门控与优化实际接续.md').write_text(doc,encoding='utf-8')
failure=read(out/'第二租期固定流首次预检失败最新.json');failure.update({'other_node_CPU_initial_audit_complete':True,'other_node_CPU_initial_original_receipt':str(old/'B_initial_cpu_audit_receipt.json'),'other_node_CPU_initial_receipt_sha256':sha(old/'B_initial_cpu_audit_receipt.json'),'update_actual_utc':now.isoformat()});(out/'第二租期固定流首次预检失败最新.json').write_text(json.dumps(failure,indent=2)+'\n')
f=out/'第二租期固定流首次预检实际失败与接续.md';f.write_text(f.read_text(encoding='utf-8')+f'\n最新实际UTC {now.isoformat()}：B clean初始CPU whole/state/public198核验及exit0原回执已D保存，首次失败仍未通过。后续v4/v5真实峰值及负结果以第二租期显存门控与优化实际接续.md为准。\n',encoding='utf-8')
ledger=read(out/'研究建议交流接续.json');assets=ledger['second_lease_new_assets'];assets.update({'asset_restore_pending':False,'other_node_initial_CPU_preservation_pending':False,'first_GPU_precheck':failure,'ordinary_v4_actual_failed':results[0],'cache_release_v5_actual_failed':results[1],'actual_GPU_precheck_attempted':True,'new_GPU_precheck':False,'new_training':False,'activation_checkpoint_v6':{'source':read(out/'第二租期激活检查点v6执行包最新.json'),'actual_GPU_started':True,'status':'RUNNING_OUTCOME_UNKNOWN','remote_run':'/data/coding/minimal_fixed_fold0_activation_checkpoint_v6_actual_20261006T144244Z','formal100_started':False}})
assets.pop('budget_trace_v4_unexecuted',None)
for n in ('a','b','c'):
 if n in assets.get('deployment_nodes',{}):assets['deployment_nodes'][n]['dependency_install_complete_verified']=True
ledger['review_thread'].update({'status':'NINTH_COMPLETE_FULL_REPORT_READ_NO_PENDING','last_completed_turn':'01a11170-52b3-7b01-9299-4d3d7a6f81ed','current_turn':None})
ledger['scientific_state']='新P4公共环境恢复已过，完整流v3/v4/v5真实GPU显存门控失败并D/原B CPU保存；缓存释放两步数值一致但保留峰值更高，停止。v6文本激活检查点短GPU预检真实进行，无新分数/100。'
ledger['updated_at_utc']=ledger['updated_utc']=now.isoformat();(out/'研究建议交流接续.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state=f'''更新UTC {now.isoformat()}。主聊天继续自主研究/实际优化与新旧租期保存，整体未完成。按最新短证据推进，不重注入长历史。
1) 人类新三P4 24h已实际接入/独立UUID/公共assets与24 Linux deps恢复并原exit0/import/version核过，旧到期不阻断新资产。仅新A REDACTED_SERVER_HOST.invalid:53314 GPU-53696803-875e-eec8-2231-29db63579891；B REDACTED_SERVER_HOST.invalid:53332 GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f；C REDACTED_SERVER_HOST.invalid:53517 GPU-417d3577-0525-788b-7296-0808a0f52012。其它所有历史地址/N/R永禁连。人类24h约Oct7UTC13:35/BJ21:35，平台未核，执行保守Oct7UTC13:30非平台确认。fresh<=5min UUID/fullargv/compute/source/assets/空间/20min+2h保存仍必须过。
2) 最新GPU实质结果：v4两真步364参数张量finite非None、singleton真实mask过，25秒natural1，仅reserved峰值6.293GiB超6(allocated5.412)。v5释放闲缓存两步目标/全梯度L1与v4差0，但reserved6.611GiB超6/allocated5.415，natural1。不是OOM/完整GPU预检通过；未完整INNER/后两步pt/100或新性能。原D snapshot21 79成员/22 88成员与原B stdlibCPU全SHA/CRC/source/role/argv/原退出和初始full联结过。详见第二租期显存门控与优化实际接续.md及最新JSON。
3) 完整初始741731078bytes SHA38d589de501ddaf7caa9f5328db421f3406b017b1753bfef19ebb62d71adca57已真实D，B原TorchCPU374state/198public exact/六donor末层零/exit0。v4/v5同初始freshSHA引用非新下载或后两步CPU。D原件 {old.as_posix()}、{v4.as_posix()}、{v5.as_posix()}。原v3没有峰值数，保留不回填；A原在线ReadTimeout/B-C自己的下载中断换同版本离线均保留不冒旧exit0。
4) 下一v6已真实短GPU启动且结果未知：work/second_lease_activation_checkpoint_v6_20261006T144158Z，plan6499ee8fbacb203b89eed327b3ee0ae6f9eb429b208fb6b5be4e1d26636dcc34。仅text非重入checkpoint/rng保留，不用失败v5逐批cacheclear，不改batch32/loss/容量/init/角色/订单/LR/6GiB门槛、不重置构造后peak；以runtime e4d0c71a...为模型数据来源。下轮先核原natural_exit/预算/完整INNER/后状态pt，不重复启动。formal100仍无源/init/完整订单/shared10/容量/预算保存冻结，不填卡跳依赖。HVP未实现。
5) fixed参考全序列一pass两Euler，MSE+.02FM+.01cycle+.05unimodal+.01variance，删unusedutility/pair/10支路。cycle终端/context梯度、donor输入detach/首阶段.125保留；公共DeBERTa+随机任务FIT695/30-only，INNER153/4预检只dummy标签，OUTER433/18禁用；seed91819独立100x695订单21x32+23尾共2200正式更新仅准备，precheck2步不冒100。未来头OUTER9/9视频232/201仅无标签元数据预留，合法结构/输入/成本停止协议未全冻，不启动头。新参考不搬旧T0常数/旧A-C2-teacher任务weight或全TRAIN统计。
6) 新科学最近同T0锁定c=-.0741191 OUTER433/18开发视频MSE.551755514→.534415092降3.14%但9/18<预声明12成本门槛，停止标量扩张；旧FIT-INNER A2.56%/B失败保留不择fold/seed/岭或同读EVAL救分。旧全TRAIN探索/INNER选模/T0与历史重叠不冒新确认/wholecrossfit。Q=delta²-2delta*r，完整Z须含生成delta，普通/4delta²加权残差匹配对照尚未训，h(pF)失败不否定所有Q。数学/原文献非新成绩。
7) 九旧流100/full/20/D/原CPU(只520506donor头)、三91818老师100/best85/45/68/full/D/原CPU(184749003全微调)、OOF1281/52 MSE.6293898611/MAE.5965080822仅teacher质量及所有已完成短实验不重跑重传/改冻结源。旧C仅10失败/最终和旧14:30缺口不回填。固定EVAL863/34 F/native/CAL_OT/CAL_Opm MSE.014974953/.014086934/.015010218/.015021068八臂负结果停止该版；CAL覆盖96→418仅机制；旧F fullTRAINfit/DEVselected，Oracle有限已知标签65.11%非严格上界/泛化。18CAL418不是418iid，cap/epsilon经验非真风险界。
8) 老租期三capture11:33/11:39以及teacher full/原CPU D保留，08:08/10:08未执行不回填。新root capture20/21/22分别实CAPTURE_COMPLETE/natural0→receipt→ZIP全SHA/CRC/member，source新root/全argv覆盖；大freshSHA引用与真正full原件分清，旧capture19保留SHA6852df98...未在新root冒执行。D优先逐次fresh查C/D禁删或改旧冻结资料。
9) SSH79535(A当前v6进程占用)/62639/17537和SFTP6854/47542/93212当前活动；关闭后必须exit/bye实际0再禁复用。password只实际提示后人类新凭据，禁写文件/命令/自动/猜测或发送审视。新审批拒绝不换工具/命令/连接绕过，当前工具可用不称底层修复；正常重启偏好不是强杀Codex许可。
10) 同建议聊天01a10fcb-6663-70a2-9a76-60e5634d0c03用户授权双向、只分析不GPU/改主源/训练/子代理/凭据/新聊天。续第六/第七/第八/第九全文已读，原第六usage失败保留；无待回复，新memory结果尚未发。重大实测先D/准确原CPU保存后按交流JSON批次SHA去重，数学/文献批说明非实验；独立采纳/暂缓/拒绝与成本，不冒建议为事实。
11) 保持10min原automation，不新建/续租/释放/关机/停健康训练/subagents/设置/浏览器UI/TEST择结构/SOTA/稳定5seed/语义真值；未变/不可行动/仅准备静默，只有实质结果、新失败、完成、必要人类行动通知。整体研究/后续实验/保存未完成。
'''
(out/'研究接续状态.md').write_text(state,encoding='utf-8')
files=[out/'研究接续状态.md',out/'研究建议交流接续.json',out/'第二租期显存优化负结果最新.json',out/'第二租期显存门控与优化实际接续.md',out/'第二租期固定流首次预检实际失败与接续.md',out/'第二租期固定流首次预检失败最新.json',out/'第二租期缓存释放v5执行包最新.json',out/'第二租期激活检查点v6执行包最新.json',root/'work/audit_budget_trace_capture_cpu_v1.py',root/'work/audit_budget_trace_capture_cpu_v2.py',root/'work/audit_cache_release_capture_cpu_v3.py',root/'work/prepare_second_lease_cache_release_v5.py',root/'work/prepare_second_lease_activation_checkpoint_v6.py',root/'work/deberta_checkpoint_source_inspection_20261006T1438Z.json',Path(__file__)]
members={}
for p in files:
 n=p.relative_to(root).as_posix();q=archive/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);members[n]={'bytes':q.stat().st_size,'sha256':sha(q)}
for p in (archive/'preceding_state').iterdir():members[p.relative_to(archive).as_posix()]={'bytes':p.stat().st_size,'sha256':sha(p)}
(archive/'member_manifest.json').write_text(json.dumps(members,indent=2)+'\n')
with zipfile.ZipFile(archive/'records.zip','x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for n in list(members)+['member_manifest.json']:z.write(archive/n,n)
with zipfile.ZipFile(archive/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
r={'actual_utc':now.isoformat(),'permanent_D':str(archive),'records_sha256':sha(archive/'records.zip'),'members':len(members),'full_SHA_CRC_unique_verified':True,'v4_v5_original_D_and_other_node_CPU_complete':True,'v6_started_no_success_claim':True};(archive/'preservation_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
