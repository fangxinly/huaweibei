"""Preserve this local-only revision; never connect to a remote node."""
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
stage = work / 'correction_calibration_collection_v2_20261006T0414Z'
proof = json.loads((outputs / '同折教师采集v2原回执数组审核本地准备核验.json').read_text(encoding='utf-8'))
assert proof['status'] == 'LOCAL_COLLECTION_V2_AST_AND_NEGATIVE_GATES_PASSED_NOT_GPU_VALIDATED'
assert not (stage / 'precheck').exists() and not (stage / 'execute').exists()
utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
space = {'utc': utc, 'c_free_bytes': shutil.disk_usage('C:/').free,
         'd_free_bytes': shutil.disk_usage('D:/').free, 'deletion_required': False}
assert space['d_free_bytes'] >= 1024**3
note = f'''# 同折教师采集v2本地接续

UTC{utc}。仅本地准备，未部署、未GPU预检、未实际FIT/INNER采集、未校准拟合或新训练。

本地审查发现v1在模型构造后重置CUDA峰值，会漏算构造/加载峰值；v1从未执行且原源、原计划保留。新v2只改资源计量及原回执门控，重置提前到模型构造前，构造后/加载后/每批和最终核4GiB；实际总时长20min、子根新资料1GiB上限仍需GPU实测。实际capture19源SHA、原始预检独立审核SHA、生成数组SHA均有门控。新plan SHA57b740fed4ee9ac2d9736956f2e6cbc3b9dcd30b102610abe49f563a493738f7，collector SHA7e25fd0f09da4b61418931108efc49a506a6926f47da7e7f34e5574f36844e47。

新增run_same_teacher_collection_v2.py等待子进程自然退出，保留precheck_exit.json/execute_exit.json原退出码、PID/argv和原回执SHA，不用超时终止健康进程。独立NumPy审核audit_same_teacher_collection_v2.py区分预检元数据与正式FIT/INNER数组，核原源/计划/角色/同T_k行序、显存时长、参数SHA不变、原退出0、INNER原预测重放和数组SHA；不是CPU整模型前向。脚本及审核bundle只冻结准备，未执行真实审核。

本地AST核验峰值重置在构造前/无step或backward；旧100轮完成回执冒新采集预检、缺新预检及缺正式原回执/退出均被拒绝。见同折教师采集v2原回执数组审核本地准备核验.json；这些负门控不能冒GPU通过。原v1角色SHA785c27b7bfb53796e3ee5d6cc9737009c85a29c1124a6633538f0e526722f0cc及三fold CAL/EVAL136/297、148/276、134/290保持不变。

下一必须有正式远端发送恢复证据；03:48自动审批拒绝后当前远端状态未知，本轮未重试、不绕过。恢复后先fresh UUID/完整argv/compute/完成标记/源/空间，核实际新子根capture及原始过程记录覆盖，再同T_k严格磁盘加载、完整INNER重放≤1e-6、fit-only统计exact、换标签0、零优化器步骤/参数SHA不变和真实预算预检，取得原exit0且独立审核后才采集。没有真实采集就不拟合λ/ε/输出上限或启动新学生。CAL只拟合、EVAL预测冻结后统计；已探索全TRAIN和全TRAINfit/DEVselected参考边界保留。

永久D collection_v2_local_preparation_20261006T0414Z为本地源/协议/审核准备，不是远端capture或异节点CPU保存；全部研究及租期保存未完成。
'''
(outputs / '同折教师采集v2本地接续.md').write_text(note, encoding='utf-8')
state_path = outputs / '研究接续状态.md'
old_state = state_path.read_text(encoding='utf-8')
archive = outputs / '研究接续状态_采集v2前_20261006T0414Z.md'
assert not archive.exists()
archive.write_text(old_state, encoding='utf-8')
body = old_state[old_state.index('# 多模态流研究短接续'):]
prefix = f'''最新本地状态UTC{utc}；最后成功远端C/B实际03:38:53，本轮未连接或发送远端命令。

最新同折教师采集v2仅本地冻结，见同折教师采集v2本地接续.md和同折教师采集v2原回执数组审核本地准备核验.json。修正v1峰值重置在模型构造之后的计量遗漏；v1未执行且保留。v2全程峰值/实际总时长/资料上限及原预检独立回执SHA门控，新自然退出wrapper/NumPy元数据与数组审核分开预检和执行，原exit0/PID/argv/行序/参数/INNER重放/数组SHA必核。AST和三旧证据/缺原回执负门控通过，不冒GPU通过；源/协议见work/correction_calibration_collection_v2_20261006T0414Z，plan SHA57b740fed4ee9ac2d9736956f2e6cbc3b9dcd30b102610abe49f563a493738f7。角色和完成教师源不改。新材料永久D collection_v2_local_preparation_20261006T0414Z，仅本地保存。

CAL限定最小二乘数学草稿合成核验已保留D calibration_amplitude_math_preparation_20261006T0404Z，解析lambda=clip(-a/b,0,1)及非CAL扰动不变/边界/老师角色拒绝；没有真实标签读取、lambda/epsilon/输出上限拟合、EVAL指标、GPU采集或求解。03:48审批限制没有正式恢复证据，不重复试探或绕过。全部实际步骤和租期保存仍未完成。

'''
state_path.write_text(prefix + body, encoding='utf-8')
backup = Path('D:/CodexBackups/selective_flow_20261003_1105/collection_v2_local_preparation_20261006T0414Z')
assert not backup.exists()
backup.mkdir()
sources = [(p, 'prepared/' + p.name) for p in stage.iterdir() if p.is_file()]
sources += [(outputs / name, 'evidence/' + name) for name in
            ('同折教师采集v2资源与原回执门控修订.json', '同折教师采集v2原回执数组审核本地准备核验.json',
             '同折教师采集v2本地接续.md', '研究接续状态.md')]
sources += [(work / name, 'builders/' + name) for name in
            ('prepare_same_teacher_collection_v2.py', 'check_same_teacher_collection_v2_preparation.py',
             'finalize_collection_v2_local_preparation.py')]
records = {}
for source, relative in sources:
    target = backup / relative
    target.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy2(source, target)
    assert sha(source) == sha(target)
    records[relative] = {'bytes': target.stat().st_size, 'sha256': sha(target)}
zip_path = backup / 'local_preparation.zip'
with zipfile.ZipFile(zip_path, 'x', zipfile.ZIP_DEFLATED) as z:
    for relative in records:
        z.write(backup / relative, relative)
with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(records)
    assert set(z.namelist()) == set(records)
    assert all(hashlib.sha256(z.read(name)).hexdigest() == records[name]['sha256'] for name in records)
saved = {'status': 'LOCAL_V2_COLLECTION_PREPARATION_PERMANENT_D_SHA_CRC_VERIFIED_NOT_REMOTE_CAPTURE',
         'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'fresh_space': space,
         'directory': str(backup), 'files': records, 'zip_sha256': sha(zip_path),
         'actual_gpu_precheck': False, 'actual_collection': False, 'actual_calibration': False,
         'remote_capture': False, 'independent_remote_cpu_preservation': False}
(backup / 'manifest.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')
(outputs / '同折教师采集v2本地永久保存.json').write_text(json.dumps(saved, indent=2), encoding='utf-8')
print(json.dumps({'status': saved['status'], 'files': len(records), 'zip_sha256': saved['zip_sha256'], 'space': space}))
