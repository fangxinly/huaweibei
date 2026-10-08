import hashlib,json,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
PRE=ROOT/'work/minimal_fixed_precheck_local_20261006T1240Z'
now=datetime.now(timezone.utc);utc=now.isoformat()
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/('minimal_fixed_precheck_local_candidate_'+now.strftime('%Y%m%dT%H%M%SZ'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
space={drive:shutil.disk_usage(drive).free for drive in ('C:/','D:/')};assert space['D:/']>100*1024*1024
plan_sha=sha(PRE/'precheck_candidate_plan_v2.json');source_sha=sha(PRE/'precheck_minimal_fixed_v2.py')
audit=json.loads((PRE/'local_static_gate_receipt_v2.json').read_text())
assert audit['candidate_plan_sha256']==plan_sha and audit['actual_execution_gate_passed'] is False
D.mkdir(exist_ok=False);out=D/'original';out.mkdir()
state=ROOT/'outputs/研究接续状态.md';shutil.copyfile(state,D/'preceding_short_state.md')
report=(f'完整流fixed v2预检与全状态重放本地源接续\n\n实际UTC {utc}。source-only目录{PRE.as_posix()}，D原件{D.as_posix()}。新runner v2 SHA {source_sha}，candidate plan v2 SHA {plan_sha}。本轮未导入Torch/连GPU/读真实输入或标签，没有新分数或真实预算、参数梯度、磁盘重放证据。\n\n'
 '候选预检先检查可信人类/平台来源、确证租期、预检20min+2h保存余量、5min内UUID/空compute/完整argv/主机身份/源资产完成空间证据，再导入Torch。JSON字段不验证来源或替代授权；合成正门控不是真实GPU门控。当前平台未核且估计已过，执行门槛关闭。空间方案远端4GiB、永久D6GiB用于两份整新状态及归档，未在真实GPU执行环境验证；6GiB Torch分配/保留峰值不冒硬件全局显存测量。\n\n'
 '正常32和mixed-singleton32各1真optimizer/scheduler步；23尾批只前后向计时，参数SHA必须不变，禁止第三步。实际主/辅目标分别求探针梯度：backward字段应仅辅助有路径，decoder仅任务有路径；最终聚合所有保留参数finite非None，零值准确记录。不用总损失减MSE制造抵消路径，不保留不用utility/pair/反事实decoder。所有这些都是尚未执行的源码断言。\n\n'
 'clean initial全参数/缓冲在步前保存，Torch/CUDA和Python/NumPy RNG分别记录；两步状态另整pt保存，原INNER153零和7 dummy标签重放0，参数/FIT统计SHA不变，自己的新pt文件/张量SHA与source/fold/seed/scope/steps核对后另公共初始化实例strict加载，原输入差<=1e-6。全过程不用真实INNER标签，不采OUTER。正式训练重置全部状态/RNG的runner仍未实现。\n\n'
 'v1从未Torch/GPU运行，原源/plan/AST回执全部保留。局部源码审阅发现terms局部变量可能在删除首模型后仍保留autograd图，新增v2每步释放terms/names，并在fresh重载后释放不用clean CPU克隆；不是实测OOM或GPU故障，也未改旧完成源/模型/目标。\n\n'
 '本地AST检查gate在Torch导入前、仅两步循环、tail无额外step、strict=True、无INNER真标签调用；JSON fixtures拒绝14项不合法条件及clean initial冒两步状态。正fixture只是合成输入；真实执行门控未过。原输入单token454/505/620 mask、主辅梯度、实际时间显存/完整重载全待真实验证。HVP/完整encoder二阶未实现，未来控制求解另验，不冒固定参考预检已覆盖二阶。\n\n'
 '下一依赖：自然退出wrapper、独立CPU整新checkpoint/数组/原回执审核、实际capture新源/root/argv覆盖、真实资产和预算/保存，以及formal100与OUTER完成授权。当前不启动GPU/100，不重试A/B或向建议聊天发送小准备，10min静默规则维持；整体研究/租期未完成。\n')
(ROOT/'outputs/完整流fixed_v2预检与全状态重放本地接续.md').write_text(report,encoding='utf-8')
short=(f'更新UTC {utc}。只读短接续/按需原证据，不重注入历史。研究、后续实验和租期保存未整体完成。\n'
 '1) 最近真实研究仍同T0固定fold0本地CPU残差：FIT695/30、INNER153/4，仅pF，零/常数/岭.01仿射与预声明INNER留一视频。A视频MSE.750301859/.731085080/.731849024，常数降2.56%但约90.54%净收益集中一视频，仿射未胜常数；B常数.756955123/仿射.783655131失败。INNER曾教师选模且A已统计，不冒新确认/整体crossfit。保留常数简单基线，停止本批扩容/挑fold seed岭。D plain_scalar_residual_actual_20261006T115633Z 17原件与独立正规方程/14代数见证保留，不重跑。\n'
 '2) 九91815/16/17流100/full/20/D/异节点CPU、三91818老师100/best85/45/68/full/D/原CPU、1281/52OOF、Oracle/同折FIT_INNER与短诊断保留不重跑重传/改冻结源。旧流仅520506供体头，老师184749003保留参数全微调。旧C仅10失败，旧C最终/14:30缺口不回填。\n'
 '3) 原固定EVAL863/34 F/native/CAL_OT/CAL_Opm MSE.014974953/.014086934/.015010218/.015021068，八臂负结果停止此版扩张，CAL半径96→418仅机制不在已读EVAL救分。teacher OOF MSE.6293898611/MAE.5965080822只teacher质量；标签已知有限Oracle降65.11%非严格上界/泛化。旧F fullTRAINfit/DEVselected，非全新确认/全流程crossfit；18CAL视频418行非418独立样本，经验epsilon/cap非真风险界。\n'
 '4) 最新真实三remote capture19仍UTC11:33、11:39；D lease_final_window_20261006T113906Z SHA/CRC/旧科学/大引用不变，完整D teacher full/原异节点CPU联结fresh核过。大SHA引用非新下载。UTC08:08/10:08未执行，缺口不回填。估UTC12:08:17/BJ20:08:17已过但平台未核，不证明释放/重建；无2h余量，禁止新GPU预检/100/学生100。\n'
 '5) 12:06:36只有C实际UUID/空compute/capture19 SHA/空间并exit0；所查run完成路径空，不冒新100完成。A key改变认证前exit1，候选ED25519 SHA256:+vqL/Af8vr0nRkVUGfj6KNUvQzZ8lpHGszpp2gGglIo未平台核实，禁改known_hosts/禁用校验/换路；B凭据一次拒绝取消exit1。A/B当前状态未知，核对问题已一次问用户，无新可信信息不重复试/通知。所有旧会话已终止，B9330/C29197及旧SSH/SFTP禁复用。D lease_expiry_readonly_20261006T120417Z 本地12:08:02三既有ZIP freshSHA/CRC过，不是新remote capture。\n'
 '6) 新fixed flow/runtime/guard/seed91819 100×695独立PCG64订单均仅本地候选：全序列一pass两Euler，MSE+.02FM+.01cycle+.05unimodal+.01variance；删utility/pair/10参考支路，保留cycle终端/context、donor输入detach、首阶段每donor.125。新构造RNG不同，不冒旧同init；公共预训练+随机任务/FIT-only，禁A/C2/teacher任务weight初始化。每轮21×32+23尾，共2200更新，完整匹配控制init/capacity/shared10仍待。\n'
 f'7) 本轮新precheck runner v2/完整状态保存strict重放源候选，plan {plan_sha}、runner {source_sha}，D {D.as_posix()}。AST/JSON fixtures14负门控通过，仅本地源码，不导入Torch/真实梯度/预算/重放。v1未执行且完整保留，v2释放遗留图/不用CPU克隆。正常/mixed两真步、tail无step、主辅探针+全保留finite非None、INNER153 dummy标签/整新ptstrict重放<=1e-6只是尚未运行断言。实际GPU gate=false，HVP未实现。详见完整流fixed_v2预检与全状态重放本地接续.md。下一wrapper/独立CPU checkpoint数组审核/capture覆盖/真实资产预算保存；formal100/OUTER完成授权仍未实现。\n'
 '8) 第六原usage失败保留，续第六与第七完整报告均读完，无待返回审视。同建议聊天01a10fcb-6663-70a2-9a76-60e5634d0c03人类授权双向，分析无GPU/改主源/训练/子代理，不发凭据/新建聊天。实质结果或文献数学决策按交流JSON去重；小准备/健康不重发。适当原始文献、允许消息不胜简单对照时简化，Q条件信息须包含delta。\n'
 '9) 只新A REDACTED_SERVER_HOST.invalid:53449 GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7；B REDACTED_SERVER_HOST.invalid:53416 GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067；C REDACTED_SERVER_HOST.invalid:53458 GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa，旧址/N/R永禁连。需合法恢复后fresh真实password仅人类原凭据，禁文件/命令/自动/猜测，先UUID/fullargv/compute/completed/source/space，退出核实际结果。新审批拒绝不换工具/连接绕过；不沿用历史阻塞/称底层修复。正常重启偏好记录但未重启/不强杀。\n'
 '10) capture19 SHA6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad，公共soft_vector_20261005T1220Z与Jacobian1327 deployment，实际clock唯一stamp/newroot源argv覆盖/CAPTURE_COMPLETE自然exit0→receipt→完整ZIP SHA/CRC/member。D优先逐次fresh查C/D禁删/改旧源。禁subagents/设置/浏览器UI/续租/释放/关机/停健康训练/TEST选结构/SOTA/稳定5seed/语义真值。10min heartbeat保留，未变/无行动/仅准备静默，只实质结果/新失败/完成/必要用户行动通知。\n')
state.write_text(short,encoding='utf-8')
ledger_path=ROOT/'outputs/研究建议交流接续.json';ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
ledger['updated_at_utc']=utc;ledger['updated_utc']=utc
ledger['local_fixed_precheck_candidate']={'directory':str(PRE),'permanent_D':str(D),'version':2,'plan_sha256':plan_sha,'source_sha256':source_sha,'phase':'UNEXECUTED_PRECHECK_SOURCE_AST_JSON_FIXTURE_ONLY','actual_GPU_gate_passed':False,'real_Torch_or_GPU':False,'new_scores':False,'sent_to_review':False,'report':str(ROOT/'outputs/完整流fixed_v2预检与全状态重放本地接续.md')}
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
paths=[p for p in PRE.iterdir() if p.is_file()]+[ROOT/'work'/n for n in ('prepare_minimal_fixed_precheck_local_v1.py','prepare_minimal_fixed_precheck_local_v2.py','audit_minimal_fixed_precheck_local_v1.py','audit_minimal_fixed_precheck_local_v2.py')]+[Path(__file__),state,ledger_path,ROOT/'outputs/完整流fixed_v2预检与全状态重放本地接续.md']
members={}
for p in paths:
 assert p.name not in members
 q=out/p.name;shutil.copyfile(p,q);assert sha(p)==sha(q)
 members[p.name]={'bytes':q.stat().st_size,'sha256':sha(q)}
package=D/'original_package.zip'
with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(out.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(package) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(members)
 for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
receipt={'status':'LOCAL_UNEXECUTED_PRECHECK_SOURCE_PLAN_AST_FIXTURE_D_SHA_ZIP_CRC_PASSED','actual_utc':utc,'free_bytes_before':space,'members':members,'package_sha256':sha(package),'actual_GPU_gate_passed':False,'new_GPU_or_scores':False,'remote_capture_or_other_node_CPU':False,'whole_research_or_lease_complete':False}
(D/'preservation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(PRE/'permanent_D_location.json').write_text(json.dumps({'directory':str(D),'receipt_sha256':sha(D/'preservation_receipt.json')},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'directory':str(D),'plan_sha256':plan_sha,'source_sha256':source_sha,'members':len(members),'new_GPU':False,'new_scores':False}))
