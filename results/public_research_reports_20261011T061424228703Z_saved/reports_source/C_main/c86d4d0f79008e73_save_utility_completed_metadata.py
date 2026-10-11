from pathlib import Path
import datetime,hashlib,json,shutil,tomllib,zipfile
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs'
DEST=Path('D:/CodexBackups/selective_flow_20261003_1105/utility_completed_metadata_20261005T0454Z')
assert shutil.disk_usage('D:/').free>50000000
complete=json.loads((out/'逐样本效用三组完整权重核验.json').read_text(encoding='utf-8'))
copies=json.loads((out/'逐样本效用三份权重独立保存核验.json').read_text(encoding='utf-8'))
diag=json.loads((out/'逐样本效用24条件诊断核验.json').read_text(encoding='utf-8'))
assert len(complete['rows'])==len(copies['receipts'])==len(diag['rows'])==3
automation=tomllib.loads(Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'))
assert automation['status']=='ACTIVE' and automation['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58' and automation['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=10'
assert '100轮均已完成' in automation['prompt'] and 'capture_utility_followup_v1_20261005T0443Z.py' in automation['prompt']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
auto={k:automation[k] for k in ('id','name','status','rrule','target_thread_id','updated_at')}
auto.update(verified_at=now,prompt_chars=len(automation['prompt']),prompt_sha256=hashlib.sha256(automation['prompt'].encode()).hexdigest(),phase='v4 completed and diagnostics verified; next design and saving dependency; lease saves pending')
with (out/'逐样本效用完成阶段监管核验.json').open('x',encoding='utf-8') as f:json.dump(auto,f,ensure_ascii=False,indent=2)
state=out/'研究接续状态.md';old=state.read_text(encoding='utf-8')
legacy=old[old.index('最新状态优先：2026-10-05北京时间11:40。'):]
free={d:shutil.disk_usage(d+'/').free for d in ['C:','D:','E:']}
lines=['# 多模态情感流模型研究接续','',f'## 最新优先：{now}，seed91813三组100轮及冻结诊断已完成','',
    'A none6143/B fixed6464/C predicted10379训练PID全部退休，不能再套live或重启。三组初始整模型、100订单与目标匹配；100dev选模、官方229预测/六方向own-pair-u-weight数组、三份新完整.pt全SHA及独立轮转七文件CPU SHA全部实际通过。旧seed91811/v3/旧84weights与测试诊断保留不重传。新根 /data/coding/selective_flow/inflow_utility_v4_deployment_20261005T0346Z；冻结四源 currentwork/inflow_utility_v4_20261005T0346Z，计划与检查证据保留。','',
    '结果none最佳55 MAE.61269742/作者batchMSE.71936572；fixed最佳74 MAE.61672485/batchMSE.72806916；predicted最佳55 MAE.62106884/batchMSE.74743333。Non0acc .875/.88425926/.86111111。预测反馈未优于匹配无反馈。fixed/predicted冻结all-off样本等权风险减少.02779747/.02925369，预测RMS.10432393/.12606482。不得混用作者128/101批次等权MSE与229样本等权风险。','',
    '三新best.pt各745113122bytes，全部本地currentwork/utility_completed_20261005/{a,b,c}/run/best.pt。A SHAbcb0e11795877e4995257dec1c2eaa9bf3c2f3728c5724ba8c4ad7caf17ede3e；B SHA67c7ba76da722fb2be9ac834e9e4f0f5375468eadbc45b4e029a8c35a567e204；C SHAe3d8a29ee0c9512e4f1e50cd9fa5347599979d193281517650ad661942ecba63。每node另6metadata、independent_manifest、destination_verification。独立根 /data/coding/selective_flow/utility_preservation_20261005T0439Z：A/run_fixed←B、B/run_predicted←C、C/run_none←A，CPU七文件全SHA实际04:48:02/04:48:02/04:46:48Z。','',
    '三GPU24条件冻结诊断已实际完成，PID A6508/B6830/C10862退休；根 /data/coding/selective_flow/utility_diagnostics_20261005T0439Z。源work/diagnose_utility_v4_v1.py SHA37b0df32b26c2e413543eb7551a15d0dbf55303c822a7397d646e1dfb29c8511；预先计划 outputs/逐样本效用冻结诊断计划.md。默认/off与保存结果0差，默认/off/forced_fixed/forced_predicted包装与原方法0差，张量state和整CP前后SHA相同，零优化更新无TEST。三组全部24×229数组/形状/权重/关闭与置换关系及逐例收益恒等式通过独立审核。none所有干预0预测变化。','',
    '教师q校准不足：预测臂T←A/T←V Spearman负，A←T/V←T虽总体符号约.77但均衡召回.5，A←V/V←A劣于零目标预测。六通道实际平均保留收益全负且u与实际收益排序相关全负，教师代理不能当最终反馈收益或共享/补充/干扰真值。全供体置换净保留收益fixed−.00308692/predicted−.00465427（仅反馈）和−.00460739（完整路径）。单seed、模式绑定节点、已观测DEV，无稳定/SOTA结论。','',
    '人类报告 outputs/逐样本效用实验分析.md；证据逐样本效用三组完整权重核验.json、逐样本效用三份权重独立保存核验.json、逐样本效用教师校准分析.json、逐样本效用24条件诊断核验.json。最新work/utility_followup_202610050448Z/{a,b,c}实际captured04:48:14Z，各112成员、全部ZIP/memberSHA，GPU空/训练及诊断PID退休。当前新capture work/capture_utility_followup_v1.py SHA7eaed4d81e7e3a502c6a8c0e66e4fc47af45017bf4b73f620c119e8a39d1979a，远端 /data/coding/selective_flow/capture_utility_followup_v1_20261005T0443Z.py --stamp新唯一；新增两新根并保留此前8根，不含整.pt。新独立审核work/audit_utility_diagnostics_v1.py；不能用旧live审核退休训练。','',
    '下一优先 outputs/主任务反事实效用下一步设计.md：效用目标对齐TRAIN内固定参考后续流主任务单通道关闭收益，确定性参考路径/缓存/标签隔离/梯度及启动零目标须真实检查，匹配协议预算尚待实现冻结，不直接跳100训练。当前保存依赖实际不足：'+', '.join(f'{d}={v/1e9:.3f}GB' for d,v in free.items())+'；再三同体积weights需约2.24GB+证据。已用文本问题请求至少3GB永久保存目录，待用户回答；不删除旧文件、不为填卡跳依赖。数学/实现准备和租期保存可继续。','',
    '本轮完成metadata独立小包正在保存 D:/CodexBackups/selective_flow_20261003_1105/utility_completed_metadata_20261005T0454Z，完成receipt及六连接exit0将另附。当前本turn六SSH/SFTP尚开放；关闭前不要假称已exit0。本轮smallpack不含完整weights，其全SHA与CPU副本另已验证。监管原十分钟已工具更新并实际TOML验证ACTIVE/currenttarget，新prompt明确v4全部完成，证据逐样本效用完成阶段监管核验.json。','',
    'Oct5北京时间13:40/14:10/14:30仍需实际最终动态保存旧namedcapture+当前新followup及后续新根，14:40只是人类估计。先读原旧namedcapture实际CLI；当前新capture不能替代旧研究根。全部授权研究/报告和期限保存完成前不称整体完成。仅当前三P4可连；N/R final及旧地址禁重试。SSH/SFTP只在实际password提示后用原聊天人类turn01a105a5-9780-72d3-80b4-0ba2bd29eed0两消息凭据；禁密码文件/自动提示/猜测，明确exit/bye核对exit0。禁subagents、浏览器/设置、续租/关机/停止健康任务/改冻结协议。','',
    '以下是旧已完成研究及连接/期限背景；以上当前状态优先。','',legacy]
state.write_text('\n'.join(lines),encoding='utf-8')
DEST.mkdir(exist_ok=False)
folders=[out,ROOT/'work/inflow_utility_v4_20261005T0346Z',ROOT/'work/utility_checks',ROOT/'work/utility_completed_20261005',ROOT/'work/utility_followup_202610050448Z',ROOT/'work/utility_completed_snapshots_20261005T0439Z']
files={}
for folder in folders:
    for p in sorted(folder.rglob('*')):
        if p.is_file() and p.suffix not in ('.pt','.tmp','.pyc') and '__pycache__' not in p.parts:files[p.relative_to(ROOT).as_posix()]=p
for p in (ROOT/'work').glob('*.py'):files[p.relative_to(ROOT).as_posix()]=p
def sha(b):return hashlib.sha256(b).hexdigest()
members=[]
with zipfile.ZipFile(DEST/'metadata.zip','x',zipfile.ZIP_DEFLATED) as z:
    for name,p in files.items():
        b=p.read_bytes();z.writestr(name,b);members.append({'name':name,'source':str(p),'bytes':len(b),'sha256':sha(b)})
with zipfile.ZipFile(DEST/'metadata.zip') as z:
    assert len(z.namelist())==len(set(z.namelist()))==len(members)
    for m in members:
        b=z.read(m['name']);assert len(b)==m['bytes'] and sha(b)==m['sha256']==sha(Path(m['source']).read_bytes())
p=DEST/'metadata.zip'
manifest={'created_at':now,'archive_bytes':p.stat().st_size,'archive_sha256':sha(p.read_bytes()),'members':members,'limits':'No .pt in metadata archive. Current new weights preserved separately. Late close/status/receipt records appended separately.'}
(DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
receipt={k:manifest[k] for k in ['created_at','archive_bytes','archive_sha256','limits']};receipt.update(members=len(members),destination=str(DEST),status='LOCAL_ORIGINALS_AND_D_METADATA_ZIP_ALL_MEMBERS_SHA_VERIFIED')
with (out/'逐样本效用完成资料本地保存核验.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(receipt))
