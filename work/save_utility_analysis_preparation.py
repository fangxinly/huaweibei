from pathlib import Path
import datetime,hashlib,json,shutil
r=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');out=r/'outputs'
p=out/'研究接续状态.md';s=p.read_text(encoding='utf-8')
text='本轮新增冻结诊断计划 outputs/逐样本效用冻结诊断计划.md（SHA47e41c08671d989d9ab3427f819c8673693834b45f5e34ae2d682ee8853a2bec），完成后全六方向教师误差/符号及排序/固定分箱，默认、全部关闭、固定.5、预测、逐通道关闭及供体置换（128/101批内循环1，分别保留权重和完整重算路径），零标签前向、零优化更新、完整state/checkpoint前后SHA及默认/off重放。已准备work/analyze_utility_completed_v1.py SHA53040891bbdc5e615c9b16cb81e7438c81be187a67ff24afe7e4a2f4c3c7c576，只读complete audit的229保存数组并重核整CP/ZIP/memberSHA，再算教师校准与all-off风险恒等式；真实完整数据尚未分析。AST及合成公式/平局/零预测检查已实际通过，准备证明 outputs/逐样本效用分析器准备核验.json。初次合成Spearman期望严格-1浮点比较失败，改np.isclose后通过，未改任何训练或GPU任务。GPU逐通道/供体干预实现及执行仍待完成，不把CPU工具当GPU干预结果。此新增计划/源码/准备证明/最终状态另存D唯一utility_analysis_preparation_202610050428Z；早先本轮9文件D快照含当时状态，晚更新状态另存并单独SHA。\n\n'
assert text not in s
p.write_text(s.replace('\n\n','\n\n'+text,1),encoding='utf-8')
dest=Path('D:/CodexBackups/selective_flow_20261003_1105/utility_analysis_preparation_202610050428Z')
dest.mkdir(exist_ok=False);rows=[]
for src in [p,out/'逐样本效用冻结诊断计划.md',out/'逐样本效用分析器准备核验.json',r/'work/analyze_utility_completed_v1.py',r/'work/check_utility_analysis_formulas.py',Path(__file__)]:
    dst=dest/src.name;assert not dst.exists();shutil.copyfile(src,dst)
    a=hashlib.sha256(src.read_bytes()).hexdigest();assert a==hashlib.sha256(dst.read_bytes()).hexdigest()
    rows.append({'name':src.name,'bytes':src.stat().st_size,'sha256':a})
receipt={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ANALYSIS_PREPARATION_SOURCE_PLAN_AND_STATE_D_SHA_VERIFIED','files':rows,'actual_model_calibration_analyzed':False,'gpu_intervention_run':False}
for dst in [out/'逐样本效用分析准备保存核验.json',dest/'local_verification.json']:
    assert not dst.exists();dst.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'files':len(rows)}))
