"""Publish the fixed diagnostic and verified CSVs; seal only small new artifacts."""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def makezip(root,zp):
    members={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p!=zp}
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
        for name in members:z.write(root/name,name)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for name,h in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    return dict(SHA=sha(zp),CRC_passed=True,unique_members=True,all_member_SHA_passed=True,member_sha256=members)

def worker(a):
    assert sha(a.plan)==a.plan_sha;plan=read(a.plan);assert sha(__file__)==plan['worker_SHA']
    p=read('work/scale_shortcut_current.json');ip=read('work/scale_shortcut_independent_current.json');parent=Path(p['D']);ind=Path(ip['D'])
    assert sha(parent/'actual/actual_result.json')==p['result_SHA'];assert sha(ind/'actual_independent_result.json')==ip['result_SHA']
    r=read(parent/'actual/actual_result.json');ir=read(ind/'actual_independent_result.json');protocol=read(parent/'source/protocol.json')
    training=Path('work/weak_retention_train40_20261007T145201Z/bundle/training_execution_protocol.json');t=read(training);fr=read('work/weak_retention_train40_20261007T145201Z/freeze_record.json');assert sha(training)==fr['protocol_SHA']
    for sealed,name in [('pinned_anchored_flow.py','anchored_flow.py'),('pinned_fixed_flow_components.py','fixed_flow_components_candidate.py')]:assert sha(parent/'source'/sealed)==t['source_sha256'][name]
    assert sha(parent/'source/pinned_author_model_reflow_new.py')==t['asset_sha256']['assets/CaReFlow/model_reflow_new.py']
    stem='标量捷径核验与best20_best36逐样本导出实际结果_20261008';outputs=Path('outputs').resolve();bundle=outputs/('best20_best36_FIT_INNER_review_'+plan['stamp']);assert not bundle.exists();bundle.mkdir()
    for model in ('best20','best36'):
        for role in ('fit','inner'):
            name=f'{model}_{role}_y_b_p_video.csv';shutil.copy2(parent/'actual'/name,bundle/name);assert sha(bundle/name)==r['models'][model]['roles'][role]['csv_SHA']
    oldmeta=read('D:/CodexBackups/selective_flow_20261003_1105/inner_lovo_output_gate_20261007T160409Z/source/protocol.json')
    newmeta=read('D:/CodexBackups/selective_flow_20261003_1105/fixed_weak_donor_mask_20261007T161226Z/A_original_capsule/original/actual_stage_receipt.json')
    readme=f'''# best20 / best36 逐样本复核数据

这四份 CSV 来自已保存预测。每个模型 FIT1494条/64视频，INNER264条/11视频；FIT 与 INNER 按视频隔离。新旧行ID、顺序与真标签完全一致，每一浮点数和ID已逐行回读核验。

- y：真实情绪标签，原连续尺度。
- b：原始特征池化后经共享 decoder 的预测；它本身也使用多模态特征，**不是消息全关 p0**。
- f：两步流轨迹共享 decoder 读出，再加 role head。
- p：b + tanh(gain) × (f − b)。best20 gain≈−0.795142945；best36≈−0.805566777。它不是凸组合。
- p0 / p1：仅 best36 提供。同一个固定模型、同一流轨迹/reader/role head/decoder/gain，六 donor feedback 输出全关/全开；p1逐值等于p。p0仍有其它跨模态路径。
- row_id：原片段ID；video：ID中第一个 `[` 前的视频ID。
- 弱/强分组：|y|≤1 / |y|>1，只用于描述与FIT训练；推理不使用真标签。

这些是全 TRAIN/DEV/TEST 合并后的第0折 FIT/INNER。官方 TEST 的部分行已进入训练，**不能把这些分数称为官方保留TEST、全新独立确认、完整五折或正式超越CaReFlow**。两检查点均由这份 INNER 选出，INNER 是探索性评价。

selected state SHA256：
- best20：{protocol['selected_state_SHA']['best20']}
- best36：{protocol['selected_state_SHA']['best36']}

校准仅用FIT标签拟合 α、β；INNER只评价。主分析片段等权；视频等权只是固定敏感性分析，均报告，不据分数挑读出。bootstrap按11个完整视频成对重采样10000次，seed20261008，条件于固定检查点与拟合参数，不重拟合或重选epoch。

包内 JSON 保存所有未舍入五项、MSE、弱/强分组、参数、结构相关性和配对区间。source_refs保留原件路径与SHA；合成测试只验证计算，不代表真实任务效果。报告见REPORT.md。
'''
    (bundle/'README.md').write_text(readme,encoding='utf8');shutil.copy2(parent/'actual/actual_result.json',bundle/'actual_result.json');shutil.copy2(ind/'actual_independent_result.json',bundle/'independent_result.json')
    shutil.copy2(parent/'source/protocol.json',bundle/'diagnostic_protocol.json');shutil.copy2(ind/'source/protocol.json',bundle/'independent_protocol.json')
    s=lambda model,role,key,region='all':r['models'][model]['roles'][role]['scores'][region][key]
    st=lambda model,role:r['models'][model]['roles'][role]['structure']
    def five(m,k):
        z=s(m,'inner',k);return '|'+m+' '+k+'|'+ '|'.join([f"{100*z['Acc7']:.6f}",f"{100*z['Acc2']:.6f}",f"{100*z['F1']:.6f}",f"{z['MAE']:.9f}",f"{z['Corr']:.9f}",f"{z['MSE']:.9f}"])+'|'
    def region(m,k):return '|'+m+' '+k+'|'+'|'.join(f"{s(m,'inner',k,z)['MAE']:.9f}" for z in ('all','weak','strong'))+'|'
    def struct(m,role):
        x=st(m,role);return '|'+m+' '+role.upper()+'|'+'|'.join([f"{x['target_on_base']['slope']:.9f}",f"{x['target_on_base']['R2']:.9f}",f"{100*x['delta_on_base']['R2']:.6f}%",f"{x['partial_corr_delta_rho_controlling_linear_base']:.9f}"])+'|'
    def ci(x):return f"{x['point']:+.9f}，95% [{x['percentile95'][0]:+.9f}, {x['percentile95'][1]:+.9f}]"
    old=st('best20','inner');new=st('best36','inner');donor=r['models']['best36']['roles']['inner']['same_flow_donor_off_on_structure']
    report=f'''# 标量捷径核验、FIT校准对照及逐样本交付

本地实际计算：{r['actual_utc']}。交付整理实际时间：{utc()}。保存数组CPU运算，无新模型前向或训练；本轮未打开官方VAL/TEST、OUTER预测或标签文件。

**结论：当前检查点的输出修正高度接近标量校准；现有证据不足以支撑“任务残差驱动的样本特异 donor 消息收益”。但缩放占比不能等同于信息占比，新模型的全部改善也不能归因于缩放。** Claude提出的三个检查已完成；best36及FIT逐样本CSV已导出，数值和ID回读完全一致。

## 1. 不训练，核验best20与best36的缩放

固定公式 Δ=p−b，ρ=y−b。以下回归均含截距；偏相关先把Δ和ρ都对[1,b]回归，再相关两个残差。标准差使用总体ddof=0。

|检查点/角色|p~b斜率|p~b R²|Δ被线性b解释的方差|corr(Δ,ρ\|b)|
|---|---:|---:|---:|---:|
'''+ '\n'.join(struct(m,role) for m in ('best20','best36') for role in ('fit','inner'))+f'''

best20 INNER 的 p/b（|b|>.3）为 **{old['ratio_abs_base_gt_point3']['mean']:.9f} ± {old['ratio_abs_base_gt_point3']['std_population']:.9f}**，213条，与Claude的1.368±.042一致。p~b R²={old['target_on_base']['R2']:.9f}也复现。按照本次明确的含截距定义，Δ解释率是 **{100*old['delta_on_base']['R2']:.6f}%**，偏相关 **{old['partial_corr_delta_rho_controlling_linear_base']:.9f}**；Claude的99.6%与.088没有复现，不能写成全数一致。若只相关“Δ去掉b后的剩余”与未残差化的ρ，则为{old['corr_delta_remainder_raw_rho']:.9f}，也不是.088。

从保存的(b,f,p)恒等式独立反推，best20 gain≈**{old['gain_from_exact_b_f_p_identity']:.9f}**，best36≈**{new['gain_from_exact_b_f_p_identity']:.9f}**；−.8057属于best36，不能拿来反推best20。best20 f~b≈{old['flow_readout_on_base']['slope']:.9f}b+{old['flow_readout_on_base']['intercept']:.9f}，best36≈{new['flow_readout_on_base']['slope']:.9f}b+{new['flow_readout_on_base']['intercept']:.9f}。恒等式数值误差低于5e−7。

best36的缩放斜率比best20更接近1，符合弱情绪放大减轻的解释。但原始b的INNER MAE也从{s('best20','inner','b')['MAE']:.9f}降至{s('best36','inner','b')['MAE']:.9f}；校准b的MAE从{s('best20','inner','b_calibrated_rows')['MAE']:.9f}降至{s('best36','inner','b_calibrated_rows')['MAE']:.9f}。表示/读出变化也参与了改善，不能把.679→.640全部归因于调缩放。

## 2. FIT拟合两参数校准，INNER一次固定对照

主分析对每条FIT片段等权，只在FIT拟合 q=αb+β。best20 α={r['models']['best20']['calibration']['rows']['b_to_y_FIT']['slope']:.9f}、β={r['models']['best20']['calibration']['rows']['b_to_y_FIT']['intercept']:.9f}；best36 α={r['models']['best36']['calibration']['rows']['b_to_y_FIT']['slope']:.9f}、β={r['models']['best36']['calibration']['rows']['b_to_y_FIT']['intercept']:.9f}。同时固定报告p的FIT校准与视频等权敏感性分析，未选择部署读出。

INNER264条/11视频，弱111、强153。五项沿用CaReFlow评价语义：Acc7全行clip-round；Acc2/support-weighted F1排除真值0、预测≥0为正；MAE/Pearson使用原连续值。以下百分比显示Acc7/Acc2/F1。

|固定预测|Acc7 %|Acc2 %|F1 %|MAE|Corr|MSE|
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(five(m,k) for m in ('best20','best36') for k in ('b','p','b_calibrated_rows','p_calibrated_rows'))+'''

|固定预测|整体MAE|弱MAE|强MAE|
|---|---:|---:|---:|
'''+ '\n'.join(region(m,k) for m in ('best20','best36') for k in ('b','p','b_calibrated_rows','p_calibrated_rows'))+f'''

best20的校准b比p点值稍差，并不能说原流完全等于校准。best36校准b整体与弱MAE更低，强MAE几乎不变；但Acc2/F1/Corr更差、Acc7打平，**没有五项全面胜出**。两个模型的b均来自多模态特征，不能据此断言没有使用跨模态信息。

按11个完整视频配对bootstrap10000次：
- best20 校准b−p，整体MAE：{ci(r['models']['best20']['roles']['inner']['paired_video_bootstrap']['b_calibrated_rows']['all']['MAE'])}。
- best36 校准b−p，整体MAE：{ci(r['models']['best36']['roles']['inner']['paired_video_bootstrap']['b_calibrated_rows']['all']['MAE'])}。
- best36 校准b−p，弱MAE：{ci(r['models']['best36']['roles']['inner']['paired_video_bootstrap']['b_calibrated_rows']['weak']['MAE'])}。
- best36 校准b−p，强MAE：{ci(r['models']['best36']['roles']['inner']['paired_video_bootstrap']['b_calibrated_rows']['strong']['MAE'])}。

整体MAE区间均跨0。两边都做FIT校准后，best36校准p−校准b的MAE差是{ci(ir['fixed_calibrated_p_minus_calibrated_b']['best36']['rows_calibrated_p_minus_calibrated_b']['all']['MAE'])}；MSE差是{ci(ir['fixed_calibrated_p_minus_calibrated_b']['best36']['rows_calibrated_p_minus_calibrated_b']['all']['MSE'])}。这提示仍有小量可能有用的非纯缩放变化；它并不能单独归因于donor反馈，也不支持“流无任何贡献”。所有区间条件于固定模型与FIT参数，不重选epoch或重拟合校准。

视频等权FIT校准的INNER MAE：best20 b/p={s('best20','inner','b_calibrated_videos')['MAE']:.9f}/{s('best20','inner','p_calibrated_videos')['MAE']:.9f}；best36 b/p={s('best36','inner','b_calibrated_videos')['MAE']:.9f}/{s('best36','inner','p_calibrated_videos')['MAE']:.9f}，方向相近；不是据结果更换主分析。

## 3. 真正同流donor开关，比b/p更直接

复用best36既有六方向mask预测：p0为六donor反馈输出全关、p1全开，p1逐值等于p；reader、role head与其它路径仍保留。INNER p1~p0斜率 **{donor['target_on_base']['slope']:.9f}**、R² **{donor['target_on_base']['R2']:.9f}**；donor变化的方差被p0解释 **{100*donor['delta_on_base']['R2']:.6f}%**；残差化后的corr(Δdonor, y−p0\|p0)=**{donor['partial_corr_delta_rho_controlling_linear_base']:.9f}**。同流反馈主要形成约3.8%的进一步幅度放大。

先前原保存结果：p0整体/弱/强MAE=.632093/.579883/.669972，p1=.640391/.601410/.668671。这是这些donor模块在此检查点的功能效应；关模块也会偏离训练时分布，不能概括所有跨模态机制。R²高与剩余方差小是诊断信号，**不是剩余“信息量”比例或必须拒绝所有消息的风险界**。

## 4. 解码器有没有硬饱和

实际训练协议与封存源码字节SHA相符。fusion与predictor都是Linear→ReLU→Linear，最终输出没有tanh、sigmoid、clamp或硬范围约束。b和f已经调用同一个decoder；f额外加role head，因此“仅让两者共享读出”不能自动堵住缩放捷径。gain的tanh只约束混合系数；Velocity的tanh只约束速度，不给最终b/p设固定上下限。

best20 INNER b范围[{old['base_range'][0]:.9f},{old['base_range'][1]:.9f}]；best36扩大到[{new['base_range'][0]:.9f},{new['base_range'][1]:.9f}]，p扩大到[{new['target_range'][0]:.9f},{new['target_range'][1]:.9f}]。原窄范围不是最终激活强制卡在±1.2。未读取内部激活，不能据源码排除有效饱和、ReLU失活或表示坍缩；这些仍需独立激活诊断。

## 5. 下一步决定

停止在现有b、Δ的幅度/符号上扫门或重挑全局gain。后续机制实验先固定、校准消息关闭基线，并保留标量校准对照；应把预算投向有条件新信息的消息，而非重复重标定。方案是冻结公共编码特征，按视频隔离拟合接收模态能预测的供体部分，再以新息和折外任务残差监督控制接受量。条件均值残差不等于一般流传输残差，更不等于信息论PID或统计独立。

必要对照为同容量条件回归/流分解、全供体消息/新息消息、普通任务损失学习门/折外残差效用门，并加校准基线。检验输出变化对p0的仿射依赖和剩余任务相关性，不能只检查MAE或单一R²阈值。输出残差化用于诊断，不能直接删掉有效校准部分就宣称更好。

这套升级尚未实现或训练；当前CSV只有标量预测，不包含完成新息分解所需的各模态状态和真正折外模型预测。当前编码器随任务微调，不能把本轮成本当成冻结特征K折的成本。完整视频五折与正式五项全面超过仍未完成。本次提供的是把后续优化方向收窄的实测依据。

## 6. 数据角色与可复现交付

本次FIT1494/64视频与INNER264/11视频按视频隔离，但来自官方TRAIN/DEV/TEST合并，第0折两个checkpoint都在INNER选模。不能将0.63017或0.62747叫官方TEST。原正式保留TEST F MAE=.643699787、CaReFlow=.619535294，与这些INNER结果不是同范围；本轮没有再次评分官方TEST。旧约0.59是开发集表现，其身份保持原记录。

四CSV在 `{bundle}`，包含best20/best36各FIT与INNER；best36额外含真正同流p0/p1。逐样本ID、标签、预测回读逐值相同；独立协方差OLS、Python混淆矩阵与指标、不同聚合方式的bootstrap核验最大误差{ir['max_covariance_and_metric_error']:.3g}。两个本地计算子进程均自然0。完整未舍入结果见同名JSON和包内actual_result.json/independent_result.json。

主计算D：{parent}。独立计算D：{ind}。交付报告/CSV/源码/自然退出记录另做小件D封存；不会复制整模型权重或删除其它原件。本轮没有把本地保存称为新remote capture或异节点CPU模型前向。
'''
    report_path=outputs/(stem+'.md');assert not report_path.exists();report_path.write_text(report,encoding='utf8');(bundle/'REPORT.md').write_text(report,encoding='utf8')
    manifest={p.name:sha(p) for p in bundle.iterdir()};write(bundle/'member_SHA.json',manifest);zp=bundle.with_suffix('.zip');audit=makezip(bundle,zp)
    result=dict(status='SCALE_SHORTCUT_AUDIT_AND_FOUR_CSV_DELIVERY_WORKER_COMPLETE',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,
        frozen_delivery_plan_SHA=a.plan_sha,diagnostic=p,independent=ip,report=str(report_path),CSV_directory=str(bundle),share_ZIP=str(zp),share_ZIP_audit=audit,
        diagnostics=r,independent_results=ir,training_source_provenance=dict(protocol=str(training.resolve()),SHA=sha(training),author_decoder_SHA=t['asset_sha256']['assets/CaReFlow/model_reflow_new.py'],new_anchored_flow_SHA=t['source_sha256']['anchored_flow.py']),
        parent_old_selected_state_SHA=protocol['selected_state_SHA']['best20'],parent_new_selected_state_SHA=protocol['selected_state_SHA']['best36'],new_complete_checkpoint_SHA=newmeta['checkpoint_sha256'],
        no_new_training_or_model_forward=True,no_official_TEST_OUTER_opened=True,all_five_formal_goal_complete=False,full_video_fivefold_complete=False,new_innovation_implemented=False)
    resultpath=outputs/(stem+'.json');assert not resultpath.exists();write(resultpath,result)
    write(a.plan.parent.parent/'worker_result.json',dict(report=str(report_path),result=str(resultpath),share_ZIP=str(zp),bundle=str(bundle)))
    print(json.dumps(dict(report=str(report_path),share_ZIP=str(zp),ZIP_SHA=audit['SHA']),ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--clock');ap.add_argument('--worker',action='store_true');ap.add_argument('--plan',type=Path);ap.add_argument('--plan-sha');a=ap.parse_args()
    if a.worker:return worker(a)
    stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z';base=Path('D:/CodexBackups/selective_flow_20261003_1105');root=base/('scale_shortcut_report_delivery_'+stamp);assert not root.exists();root.mkdir();source=root/'source';source.mkdir();script=source/Path(__file__).name;shutil.copy2(__file__,script)
    plan=dict(actualclock_freeze_UTC=a.clock,stamp=stamp,worker_SHA=sha(script),scope='Deliver completed fixed selected20/36 local FIT/INNER diagnostic and all four exact CSV exports; no fit, selection, forward or TEST access.')
    write(source/'protocol.json',plan);psha=sha(source/'protocol.json');cmd=[sys.executable,'-X','utf8',str(script.resolve()),'--worker','--plan',str((source/'protocol.json').resolve()),'--plan-sha',psha]
    write(root/'dispatch.json',dict(actual_UTC=utc(),fullargv=cmd))
    with (root/'stdout.log').open('wb') as out,(root/'stderr.log').open('wb') as err:
        child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));code=child.wait()
    write(root/'natural_exit.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd,natural_exit=code,plan_SHA=psha))
    if code:print((root/'stderr.log').read_text(encoding='utf8'));raise SystemExit(code)
    wr=read(root/'worker_result.json');delivery=root/'delivery';delivery.mkdir()
    for name in ('report','result','share_ZIP'):shutil.copy2(wr[name],delivery/Path(wr[name]).name);assert sha(wr[name])==sha(delivery/Path(wr[name]).name)
    continuation_path=Path('outputs/残差消息最新诊断接续.json');previous=read(continuation_path);shutil.copy2(continuation_path,root/'previous_continuation.json')
    zp=root/'complete_local_delivery_originals.zip';audit=makezip(root,zp);write(root/'actual_local_archive_audit.json',dict(actual_UTC=utc(),**audit))
    updated=dict(previous);updated.update(status='ACTUAL_SCALE_SHORTCUT_FIT_CALIBRATION_SATURATION_AUDIT_FOUR_CSV_D_COMPLETE',actual_UTC=utc(),report=wr['report'],result=wr['result'],D_seal=str(root),seal_SHA=audit['SHA'],share_ZIP=wr['share_ZIP'],
        primary_diagnostic=read('work/scale_shortcut_current.json'),independent_diagnostic=read('work/scale_shortcut_independent_current.json'),previous_continuation_record=str(root/'previous_continuation.json'),overall_complete=False)
    write(continuation_path,updated)
    state=Path('outputs/研究接续状态.md');prefix=f"最新实际 {a.clock}：先读《标量捷径核验与best20_best36逐样本导出实际结果_20261008.md/json》。旧20/新36 FIT1494/64与INNER264/11保存数组缩放/FIT-only校准/五项/视频bootstrap/真实p0p1/源码输出范围核验自然0；独立协方差/混淆/CSV逐值复核自然0，误差1.55e-15，两计算与四CSV/报告全SHA ZIPCRC唯一D保存。best20 INNER p~b R².99935555/Δ解释99.1694%/偏相关.14956，gain−.79514；新36 R².99895758/gain−.80557。新校准b INNER MAE.630167 vs p.640391但整体CI跨0/非五项同时胜；同流donor p1~p0 slope1.038383/R².999974/偏相关.02578。解码器末层Linear无硬clamp，不排除有效激活饱和。两者INNER选模/合并角色非官方TEST/完整五折；无新模型前向/训练/TESTOUTER读取/删原件/新remote capture。升级仍未实现，整体目标未完成；原闭合会话ID继续禁复用。以下历史。\n\n"
    state.write_text(prefix+state.read_text(encoding='utf8'),encoding='utf8')
    post=root/'actual_continuation_update';post.mkdir();shutil.copy2(continuation_path,post/continuation_path.name)
    # Preserve the short updated prefix; old full history remains in its original file.
    (post/'研究接续短状态.md').write_text(prefix,encoding='utf8');pa=makezip(post,post/'continuation_update.zip');write(root/'actual_continuation_update_archive_audit.json',dict(actual_UTC=utc(),**pa))
    pointer=dict(status='REPORT_CSV_D_DELIVERY_AND_CONTINUATION_COMPLETE',D=str(root),ZIP_SHA=audit['SHA'],report=wr['report'],result=wr['result'],share_ZIP=wr['share_ZIP'],overall_complete=False)
    write('work/scale_shortcut_delivery_current.json',pointer);print(json.dumps(pointer,ensure_ascii=False));print((root/'stdout.log').read_text(encoding='utf8'))
if __name__=='__main__':main()
