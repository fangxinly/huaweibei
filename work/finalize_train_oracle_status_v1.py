from pathlib import Path
import datetime,hashlib,json,shutil
work=Path(__file__).resolve().parent;outputs=work.parent/'outputs'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=datetime.datetime.now(datetime.timezone.utc);stamp=utc.strftime('%Y%m%dT%H%M%SZ')
state=outputs/'研究接续状态.md';archive=outputs/('研究接续状态_Oracle完成前历史_'+stamp+'.md');assert not archive.exists();shutil.copy2(state,archive)
closure=json.loads((outputs/'TRAIN_Oracle运行会话退出核验.json').read_text(encoding='utf-8'));assert closure['all_exit_zero']
joint=json.loads((outputs/'TRAIN_Oracle完成后三节点动态保存联合核验.json').read_text(encoding='utf-8'))
assert joint['status']=='THREE_ATOMIC_ORACLE_COMPLETE_SNAPSHOTS_PRIOR_EVIDENCE_AND_ORIGINAL_CPU_COPY_AUDITED'
status=f'''最新真实UTC {utc.isoformat()}。

# 多模态流研究短接续

本轮：用户明确重试；旧SSH38957实际Unknown process。正常fresh三SSH/SFTP已成功，真实password提示后用本聊天人类凭据，GPU/命令发送恢复。未改设置/安全规则/客户端，不称底层工具代码已修复或原拒绝根因已定位。三实际UUID/空compute核实。UTC02:03:28本轮SSH24204/48078/7150、SFTP91421/93769/77340均明确exit/bye且真实exit0，旧ID禁止write_stdin。其他历史旧session不据此回填退出证明。

已完成：九个新100轮91815/91816/91817A/B/C2的完整weight、20诊断、永久D和独立CPU全部保留不重跑重传；原C v1仅10失败。旧84/v2/v3/v4和旧v5A/B不重复。旧v5C最终未取，最后81/临时31不是100；旧14:30捕获缺口不回填，旧A独立C回执只有终端证据不能冒已下载。

新实验实际完成：固定C2best37 TRAIN Oracle诊断，root /data/coding/train_oracle_diagnostic_20261006T014605Z。C PID10681 UTC01:57:09至01:58:09正常exit0/退休，1281行/52视频/41批含尾1，59.80秒/峰值151230976bytes。原尺度β/三步.25/信赖域.25，不训练或改参数。学习分支换标签0/原控制重放0；p0误差9.54e-7，double目标相对误差6.81e-17/梯度HVP0，参数SHA前后相同/参数grad None。diag v2 SHA9df96fc68214c1006eb2bccd279fdbfa8e74ae351cd73a528fed507a84fa3ed1，plan_v2 SHAbe7292966e15b8fec70dcda0a349cc953b8e8f346d5fc2a7782a00366b3792ef，原NPZ SHA92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658。21pins已实读，C视频映射原缺失由永久A快照相同SHA补齐。只读TRAINcache，不读DEV/TEST。

结果：TRAIN raw/learned/oracle/zero/mismatched MSE .01612769/.01526792/.00562681/.02165725/.13377095，学习下降5.33%/标签Oracle65.11%，oracle1281逐例改善。当前消息空间/求解预算有标签已知改善；不是严格上界/未知条件均值/泛化或语义真值。此模型与参考已全TRAINfit/DEVselected，不能把低TRAIN误差当DEV收益。既有DEV A/B/C2 MAE .59844172/.59862399/.59932536，C2未优于对照的结论保留。报告TRAIN_Oracle可达性实验分析.md与原始数组独立核验.json。

新保存完成：永久D train_oracle_completed_20261006T0158Z/c原数组/源/计划/日志/exit0九文件SHA，CPU B原副本在同新root/cpu_preservation/c，完整九文件与NumPy独立数组审核/原CPU回执已下载审核。训练诊断C/CPU B，不冒重新GPU执行。保存根同新root，避免遗漏CPU新根。原模型没有新weight或重复旧weight传输。

最新原子capture16已三节点上传核SHA 0f9452b2aeabfd93eda63d57eb833f5fb4ef560dff5935ad86e1f37e5b60faaf；--root公共 /data/coding/soft_vector_research_20261005T1220Z --deployment旧Jacobian /data/coding/jacobian_aligned_v2_deployment_20261005T1327Z --stamp真实唯一ID。原capture14/15和科学源保留。capture16新增Oracle/CPU全部source/npz/json/log/txt和实际inventory/process匹配，实际CAPTURE_COMPLETE/exit0后receipt再ZIP下载。三真UTC02:00:50快照永久D train_oracle_completed_snapshots_20261006T020034Z，全SHA/ZIPCRC/成员唯一/原九训练与CPU证据未变/Oracle完成/独立CPU原回执联合核验通过。目录020034是ID而非实际时刻。旧大weight只是fresh整体SHA引用，不冒再次下载。审计源audit_soft_snapshot_v16.py、audit_train_oracle_completed_snapshots_v1.py；proof TRAIN_Oracle完成后三节点动态保存联合核验.json。

下一优先：实现已准备按视频隔离标量教师的真实GPU机制预检，再据实测预算与保存空间正式冻结100。目前teacher仅本地plan_v2/runtime/precheck/训练器草稿，不是已部署/预检/正式100/OOF。三折fit/inner/outer695/153/433、715/142/424、714/143/424；公共预训练+随机任务初始化，禁止全TRAINfit A/C2初始化，fit-only归一化、inner选模、outer标签隔离、全保留参数梯度/跨折非统计init SHA/原输入重放/峰值显存/预算/三完整weight空间必须实核。跨折只转μ_oof标量，现有A特征仍全TRAINfit/DEVselected，禁止全流程crossfit误称。新teacher根/CPU根必须扩展新capture后部署；不为填卡跳依赖。报告按视频分组TRAIN内教师隔离设计.md；work/group_teacher_plan_20261005T1650Z/group_teacher_plan_v2.json；train_group_teacher_v1_draft.py要求实际三机制回执与formalplan/十分钟空间证据才启动。

唯一授权新P4：A REDACTED_SERVER_HOST.invalid:53449 GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7；B REDACTED_SERVER_HOST.invalid:53416 GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067；C REDACTED_SERVER_HOST.invalid:53458 GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa。旧址/N/R永不连。freshSSH/SFTP，实际password提示后仅人类原凭据，禁文件/命令/自动提示/猜测，exit/bye核exit0。每轮fresh实际UUID/完整argv/compute/history或完成/源协议/空间。D优先fresh查盘禁删，禁止设置/浏览器/续租/关机/停健康训练/subagents/改旧冻结源。

24h首实际Oct5UTC12:08:17只估Oct6UTC12:08:17/BJ20:08:17，平台未核；UTC08:08/10:08/11:38强化实际动态保存，完成立即永久D和独立CPU，不伪称期限捕获或回填。官方TRAIN1281/dev229探索，禁TEST选结构/五seed稳定/SOTA/语义真值。原十分钟监管，健康未变/无行动静默，实质结果/新失败/完成/必要行动才通知。全部研究与租期保存未完，不能称整体完成。本轮前历史已完整归档{archive.name}，禁止重新注入长历史。
'''
state.write_text(status,encoding='utf-8')
done=Path('D:/CodexBackups/selective_flow_20261003_1105/train_oracle_completed_20261006T0158Z/c')
assert shutil.disk_usage(done).free>1073741824
reports=done/('analysis_and_state_'+stamp);reports.mkdir()
files=[state,archive]+[outputs/n for n in ['TRAIN_Oracle可达性实验分析.md','TRAIN_Oracle原始数组独立核验.json','TRAIN_Oracle异节点CPU原回执核验.json','TRAIN_Oracle完成后三节点动态保存联合核验.json','TRAIN_Oracle运行会话退出核验.json','TRAIN_Oracle完成后既有根独立核验.json','TRAIN_Oracle完成捕获16修订冻结.json']]
files += [work/n for n in ['report_train_oracle_completed_v1.py','audit_train_oracle_completed_snapshots_v1.py','capture_soft_vector_v16.py','audit_soft_snapshot_v16.py','finalize_train_oracle_status_v1.py']]
manifest={}
for path in files:
 target=reports/path.name;shutil.copy2(path,target);assert sha(path)==sha(target);manifest[path.name]=dict(bytes=target.stat().st_size,sha256=sha(target))
(reports/'preservation_manifest.json').write_text(json.dumps(dict(utc=utc.isoformat(),files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='ORACLE_COMPLETE_REPORT_AND_SHORT_STATE_D_SHA_VERIFIED',destination=str(reports),files=len(files))))
