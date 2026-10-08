"""Save new CPU-only analyses and update short continuation without remote access."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


work = Path(__file__).resolve().parent
outputs = work.parent / 'outputs'
a = json.loads((outputs / '固定新路径全局幅度收缩受限最优CPU分析.json').read_text(encoding='utf-8'))
b = json.loads((outputs / '现有两路径交叉接受器CPU诊断.json').read_text(encoding='utf-8'))
assert a['calibration_parameters_fitted'] is b['calibration_parameters_fitted'] is False
assert sha(outputs / '现有两路径交叉接受器CPU冻结数组.npz') == b['frozen_selection_array_sha256']
assert a['conditions']['new_path_step_1']['cannot_beat_old_learned_endpoint_with_one_global_scalar'] is True
utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
space = {'utc': utc, 'c_free_bytes': shutil.disk_usage('C:/').free,
         'd_free_bytes': shutil.disk_usage('D:/').free, 'deletion_required': False}
assert space['d_free_bytes'] >= 1024**3
note = '''# 方向、幅度和接受规则：已完成数组的CPU机制分析

本轮只读取已经冻结并永久保存的1281 TRAIN行预测数组。没有GPU调用、新生成路径、参数更新、正式CAL拟合或CAL/EVAL分组指标。原完成源/采集v2/角色计划不改；旧全TRAIN拟合且DEV选模的参考边界保留。

## 固定方向的全局幅度能补救多少

对固定输出方向d_i，考察自由输出插值p_i(alpha)=p_Fi+alpha*d_i，一个全局alpha在[0,1]内。已观察TRAIN风险差精确为alpha*A+alpha²*B，其中A=mean(2*(p_F-y)*d)，B=mean(d²)。此受限类的最小值用alpha*=clip(-A/(2B),0,1)求出，并用直接预测平方误差以及101点逐一检验恒等式/边界。

| 固定方向来源 | 全TRAIN事后alpha* | 此受限类最小MSE | 相对原F下降 |
|---|---:|---:|---:|
| 新OOF生成第1步 | 0.0686993 | 0.0158471 | 1.74% |
| 新OOF生成第2步 | 0.0607570 | 0.0158834 | 1.51% |
| 新OOF生成第3步 | 0.0581437 | 0.0158956 | 1.44% |
| 新OOF按mu选择的输出 | 0.0579653 | 0.0158891 | 1.48% |
| 教师mu与F的输出差 | 0.0217140 | 0.0158254 | 1.87% |

原F MSE0.0161277，原学习控制末步0.0152679，下降5.33%。因此，仅保留这些固定输出方向并乘一个全局系数，即使使用全TRAIN真实标签事后取最优，也不能达到旧控制末步。本次第3步的0.05814与此前教师差方向的0.02171对应不同方向，不能混作一个参数。

这里的“最小值”只属于固定方向、全局系数、这一组已观察数组。它不是消息空间可达最优、逐样本门控/非线性再求解上界，也不是独立校准或泛化结果。输出自由插值没有被部署成消息控制，事后系数均不得移入正式实验。小幅度仍可能有助于降低损害，但不足以支持只优化步长的路线。

## 相同候选池、相同接受形式的2×2交叉检查

两生成路径分别复用旧学习残差路径和已完成OOF路径，各4个候选包含原F。两个接受器分别使用旧估计残差所隐含的标量目标p0-rho_old和OOF mu；全条件采用float64的(pred-target)²最早最小。选择数组先写盘并冻结SHA，之后才读既有真实TRAIN y统计，没有调参。

| 既有生成路径 | 接受目标 | MSE | 相对各自原F下降 | 回退原F |
|---|---|---:|---:|---:|
| 旧学习 | 旧学习 | 0.0152546 | 5.41% | 0/1281 |
| 旧学习 | OOF老师 | 0.0154823 | 4.00% | 623/1281 |
| 新OOF | 旧学习 | 0.0160738 | 0.33% | 1110/1281 |
| 新OOF | OOF老师 | 0.0789019 | -389.23% | 0/1281 |

5.41%是此次固定规则选路径的CPU条件，5.33%是原训练控制取末步，两者不是同一选择器。交叉检查补齐了新路径配旧接受目标的缺项：回退大幅降低损害，但这条既有新路径可提取的净收益仍很小。生成与接受之间存在重要交互，不能把旧路径配OOF接受器4.00%当新OOF生成有效的证据。旧学习目标来自全TRAIN拟合，不能把这些差异解释为无泄漏条件均值精度或未见视频上的因果效果。

## 下一步优化顺序

1. 保留同T_k FIT/INNER零标签采集和CAL/EVAL隔离的先决顺序。远端合法恢复、源/空间/预算/GPU原回执/独立审核全通过后采集，再只CAL拟合；不跳依赖、不重跑已完成训练。
2. 幅度短实验主要检验能否减少有害修改和合理回退，同时明确纠错方向的效用。正式100轮不能仅因新目标更接近教师或代理损失下降而启动；需要公平的原F对照和方向能力证据。
3. 校准目标应以冻结且detach的原F预测为锚：mu_lambda=p_F+lambda*(mu-p_F)。若终端可微，在原F位置且接近度梯度为零时，任务梯度恰为未收缩目标梯度的lambda倍；lambda=0应精确回退。但二阶曲率和后续非线性路径并不整体乘lambda，因此上述输出插值最优值不等于新消息求解最优。实际GPU预检应核lambda=0回放、锚/目标无梯度、首步梯度缩放和单token二阶有限性。
4. 泛化阶段需按折隔离参考编码器/流/供体/读出，与同T_k匹配训练/选模，再在未参与校准的EVAL视频比较。当前全TRAINfit参考的极小TRAIN误差与OOF教师的误差处于不同拟合状态；本负结果不能单独证明教师纠错路线在未见视频无效。新增完整训练要另行冻结初始化、角色、100订单、容量、预算及永久空间，不能用全TRAINfit A/C2初始化冒隔离。

本轮只有原数组上的探索性机制证据，未校准lambda/epsilon/输出上限、未部署新控制器、未作新的GPU或异节点CPU执行。研究和租期保存未完成。源/数组/报告永久D direction_amplitude_factorial_analysis_20261006T0444Z，全SHA/ZIPCRC/member另见本地保存证明。
'''
(outputs / '方向幅度与接受规则交叉机制分析.md').write_text(note, encoding='utf-8')
state_path = outputs / '研究接续状态.md'
old_state = state_path.read_text(encoding='utf-8')
archive = outputs / '研究接续状态_方向幅度CPU分析前_20261006T0444Z.md'
assert not archive.exists()
archive.write_text(old_state, encoding='utf-8')
prefix = f'''最新本地研究UTC{utc}；本轮只CPU原冻结数组分析，无远端重试/新GPU/正式CAL拟合。见方向幅度与接受规则交叉机制分析.md及两新CPU JSON。

新OOF既有输出方向乘一个全局alpha，即使全TRAIN事后最优，第1/2/3步收益仅1.74%/1.51%/1.44%，均未及原旧控制末步5.33%；不是非线性消息/逐样本门控或泛化上界，不把事后alpha0.05814与老师差方向0.02171混为正式参数。2x2同形式最早最小接受器交叉：旧生成/旧目标5.41%，旧生成/OOF4.00%，新OOF生成/旧目标回退1110/1281且净降0.33%，新生成/OOF MSE.0789019。选择数组先SHA冻结才读既有y，未使用CAL/EVAL分组。新结果提示生成方向与接受规则需分开优化；幅度短实验仍有意义，但正式训练还需方向证据与参考按折隔离。所有结果仍在全TRAINfit/DEVselected参考上，仅探索机制；无新控制收益或全流程crossfit。新源/数组/报告永久D direction_amplitude_factorial_analysis_20261006T0444Z，本地SHA/CRC/member过。

'''
state_path.write_text(prefix + old_state, encoding='utf-8')
backup = Path('D:/CodexBackups/selective_flow_20261003_1105/direction_amplitude_factorial_analysis_20261006T0444Z')
assert not backup.exists()
backup.mkdir()
sources = [(work / name, 'sources/' + name) for name in
           ('analyze_same_direction_amplitude_v1.py', 'analyze_existing_path_cross_selector_v1.py',
            'finalize_direction_amplitude_analysis_v1.py')]
sources += [(outputs / name, 'evidence/' + name) for name in
            ('固定新路径全局幅度收缩受限最优CPU分析.json', '现有两路径交叉接受器CPU诊断.json',
             '现有两路径交叉接受器CPU冻结数组.npz', '方向幅度与接受规则交叉机制分析.md', '研究接续状态.md')]
records = {}
for source, relative in sources:
    target = backup / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert sha(source) == sha(target)
    records[relative] = {'sha256': sha(target), 'bytes': target.stat().st_size}
zip_path = backup / 'local_cpu_research.zip'
with zipfile.ZipFile(zip_path, 'x', zipfile.ZIP_DEFLATED) as z:
    for relative in records:
        z.write(backup / relative, relative)
with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(records)
    assert set(z.namelist()) == set(records)
    for name, record in records.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == record['sha256']
saved = {'status': 'LOCAL_CPU_DIRECTION_AMPLITUDE_FACTORIAL_RESEARCH_D_SHA_CRC_MEMBER_VERIFIED',
         'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'fresh_space': space,
         'directory': str(backup), 'files': records, 'zip_sha256': sha(zip_path),
         'new_gpu_execution': False, 'actual_calibration': False, 'new_remote_capture': False,
         'new_independent_remote_cpu_execution': False,
         'local_read_lookup_issue': {'exit_code': 1, 'reason': 'rg included absent analyze_oof_utility_same_pool_v1.py; other immutable source matches returned; no scientific execution failed'}}
(backup / 'manifest.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')
(outputs / '方向幅度交叉CPU研究本地永久保存.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')
print(json.dumps({'status': saved['status'], 'files': len(records), 'zip_sha256': saved['zip_sha256'], 'space': space}))
