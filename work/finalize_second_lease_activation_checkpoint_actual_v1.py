"""Join original GPU and second-node whole-state CPU evidence, preserve records."""
import datetime, hashlib, json, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
BASE = Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_activation_checkpoint_actual_20261006T144615Z')
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

now = datetime.datetime.now(datetime.timezone.utc)
assert shutil.disk_usage('D:/').free > 6*1024**3
gpu = read(BASE/'local_joint_audit.json')
cpu = read(BASE/'B_full_state_CPU_original_receipt.json')
ex = read(BASE/'B_full_state_CPU_original_exit.json')
raw = BASE/'a/original_small_files/run'
receipt = read(raw/'out/actual_precheck_receipt.json')
assert gpu['GPU_precheck_complete'] and gpu['natural_exit_code'] == 0
assert ex['exit_code'] == 0 and ex['natural_wait_verified']
assert ex['child_pid'] == cpu['pid'] and ex['child_full_argv'] == cpu['argv']
assert cpu['source_sha256'] == '80d06a49146b7ebfb821760f25b6b5ff32284fa3df00fb376ec2920c7e12fadd'
assert cpu['original_receipt_sha256'] == sha(raw/'out/actual_precheck_receipt.json')
assert cpu['natural_exit_sha256'] == sha(raw/'natural_exit.json')
old = Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_precheck_failure_actual_20261006T141445Z/a/clean_initial_full.pt')
for name, file, key in [('clean_initial_full.pt', old, 'initial_full'),
                        ('after_two_steps_full.pt', BASE/'a/after_two_steps_full.pt', 'after_two_steps_full')]:
    c = cpu['checkpoints'][name]; r = receipt[key]
    assert c['file_sha256'] == sha(file) == r['file_sha256']
    assert file.stat().st_size == r['bytes'] == 741731078
    assert c['state_sha256'] == r['metadata']['state_sha256']
    assert c['tensors'] == r['tensors'] == 374
assert len(cpu['mechanism']) == 2 and not cpu['GPU_used'] and not cpu['new_model_forward_or_scores']
for c, r in zip(cpu['mechanism'], receipt['donor_mechanism']):
    assert c['stage'] == r['stage'] and c['terminal_scalar_change_max'] == r['terminal_scalar_change_max'] == 0
    assert c['context_change_max'] == r['context_change_max']
    assert c['sha256'] == sha(raw/'out'/c['file'])
archive = BASE/('research_records_'+now.strftime('%Y%m%dT%H%M%SZ'))
archive.mkdir(exist_ok=False)
for n in ('研究接续状态.md','研究建议交流接续.json','第二租期显存门控与优化实际接续.md'):
    target=archive/'preceding_state'/n;target.parent.mkdir(exist_ok=True);shutil.copy2(OUT/n,target)
joint = {'actual_utc': now.isoformat(), 'status':'GPU_FULL_PRECHECK_D_COMPLETE_WEIGHTS_AND_ORIGINAL_OTHER_NODE_CPU_JOINT_PASSED',
         'original_GPU_audit': gpu, 'original_CPU_receipt_sha256':sha(BASE/'B_full_state_CPU_original_receipt.json'),
         'original_CPU_exit_sha256':sha(BASE/'B_full_state_CPU_original_exit.json'),
         'original_CPU_receipt':cpu,'original_CPU_exit':ex,
         'CPU_model_forward':False,'formal100_started':False,'new_performance_scores':False,
         'initial_full_is_previously_preserved_original':True,'new_after_two_steps_full_original_saved_D_and_B':True}
(BASE/'complete_GPU_D_B_CPU_joint_audit.json').write_text(json.dumps(joint,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'第二租期完整流GPU预检与D_B全状态审核最新.json').write_text(json.dumps(joint,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc=f'''第二租期完整流真实GPU预检与完整权重保存

实际联合审核UTC {now.isoformat()}。v6 child960在UTC14:44:02.230802自然exit0，wrapper等待exit0。完整GPU预检76.503秒，构造/加载/两步/尾23/INNER重放全程累计峰值 allocated3745930752bytes(3.49GiB)、reserved4324327424bytes(4.03GiB)，均低于原6GiB门槛，无构造后峰值重置。

修正只启用public文本DebertaV2非重入激活检查点，preserve_rng_state=True，不改batch32/目标/seed91819/init/角色/100订单/学习率或预算。原v4保留峰值6.293GiB和v5释放缓存6.611GiB自然失败保留，v5方案停止。v6两步目标2.843935728/1.458111644及364参数张量梯度L1范数与v4差0；没有比较完整梯度张量逐字节。新保留参数185402807，全链可训练，区别于旧仅520506供体头和旧老师184749003参数。

FIT695/30-only统计，正常32与真实singleton混合32两optimizer步，tail23仅前后向没有第三步；所有364参数张量finite非None，主辅梯度实际路径通过，零值如实记录。INNER153/4原输入换dummy标签0与7误差0，参数/FIT统计不变；fresh实例加载自己的完整新pt严格磁盘重放误差0。无INNER真标签、OUTER/CAL/EVAL/DEV/TEST指标，HVP未实现，无formal100或新性能分数。

初始完整741731078bytes SHA38d589de501ddaf7caa9f5328db421f3406b017b1753bfef19ebb62d71adca57沿用已有D/B原件，非重传；两步后完整同字节数SHAc387a1a25d62978c9e004188522f808547a6f9c5f1979acaee030b3483de20b2本轮真实D与B保存。B原独立TorchCPU自然exit0，对两整pt各374状态张量/完整元素/SHA/FIT统计/元数据及INNER和供体NPZ核验，非CPU模型前向。A capture23 UTC14:46:26.508410实际CAPTURE_COMPLETE/exit0后receipt，98成员全SHA/ZIPCRC/unique/新源/完整argv和预算联合核过。原件目录 {BASE.as_posix()}。

同checkpoint供体置零机制：初始feedback/context/state/scalar变化0。两步后feedback约2.78e-7、context6.40e-8、第二Euler状态1.19e-7，但终端标量变化仍0，恢复重放0。只能说明微小内部路径变化，不能称供体改善预测或得分，不加额外step强求正结果。

B转存最初校验摘要抄写多一字符造成AssertionError而阻止展开，随后上传因缺目录未开始；按D原capture_receipt纠正后才完成实际转存与CPU审核。原A科学结果未受影响，不回填首次转存成功。

下一是另冻结正式100源、clean初始全state/RNG、完整FIT订单和尾23/2200步、同容量init与共享前10订单比较协议、未来head合法信息和232/201视频角色、保存及时间预算，然后fresh UUID/fullargv/compute/asset/source/空间门控；不能把两步预检或本记录冒正式100完成、整体crossfit或性能收益。公共源码依据 https://github.com/pytorch/pytorch/blob/v2.1.0/torch/utils/checkpoint.py 和 https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/deberta_v2/modeling_deberta_v2.py 。
'''
(OUT/'第二租期完整流GPU预检与全状态保存实际结果.md').write_text(doc,encoding='utf-8')
ledger=read(OUT/'研究建议交流接续.json'); assets=ledger['second_lease_new_assets']
v=assets['activation_checkpoint_v6'];v.update({'status':'ACTUAL_COMPLETE_GPU_D_B_FULL_CPU_JOINT_PASSED',
    'actual_GPU_complete':True,'original_other_node_complete_two_state_CPU_audit_pending':False,
    'complete_joint_audit':str(BASE/'complete_GPU_D_B_CPU_joint_audit.json'),
    'complete_joint_audit_sha256':sha(BASE/'complete_GPU_D_B_CPU_joint_audit.json'),
    'GPU_final_budget':gpu['final_budget'],'donor_mechanism':gpu['donor_mechanism'],
    'formal100_started':False,'new_performance_scores':False})
for key in ('deployment_nodes','deployment_evidence'):
    entries=assets.get(key,{})
    if isinstance(entries,dict):
        for entry in entries.values():
            if isinstance(entry,dict):entry['dependency_install_complete_verified']=True
ledger['scientific_state']='新P4完整流v6真实GPU完整预检与D两完整state/B原TorchCPU核验完成；76.5s/allocated3.49/reserved4.03GiB/INNER换标签和strict重放0，无新分数/100；v4/v5原失败保留。'
ledger['updated_at_utc']=ledger['updated_utc']=now.isoformat()
(OUT/'研究建议交流接续.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state=(OUT/'研究接续状态.md').read_text(encoding='utf-8')
lines=state.splitlines()
lines[0]=f'更新UTC {now.isoformat()}。继续自主研究/实际优化及新旧租期证据保存，整体未完成。'
lines=[(f'2) 最新实质GPU结果v6 child960 UTC14:44:02自然exit0，完整预检76.5s，全程allocated3.49GiB/reserved4.03GiB<原6GiB，正常32/mixedsingleton32两step+尾23无第三步/全364finite非None/主辅路径/INNER153原输入dummy换标签0/fresh整新pt strict重放0/FIT统计不变均真实通过。新全链185402807参数；HVP未实现，无真INNER标签或OUTER/CAL/EVAL/DEV/TEST/新成绩/100。详见第二租期完整流GPU预检与全状态保存实际结果.md及最新JSON。' if x.startswith('2) ') else x) for x in lines]
lines=[(f'3) v6 A capture23 UTC14:46:26实际COMPLETE/exit0→receipt、98成员ZIP全SHA/CRC/唯一/源/完整argv过，D {BASE.as_posix()}。完整新两步pt741731078bytes SHAc387a1a...真实D/B；初始SHA38d589...联结既有原D/B非重传。B原TorchCPU两完整374state/元数据/FITstats/数组exit0通过，不是CPU模型前向。原v3峰值缺失不补，v4reserved6.293/v5cacheclear6.611GiB失败D/B原件保留，cacheclear停止。' if x.startswith('3) ') else x) for x in lines]
lines=[('4) v6只启文本非重入checkpoint/RNG保留，不改原模型目标batch/init/角色/订单/LR/6GiB、不重置构造后峰值；两步目标及全部梯度L1与v4差0，不冒全梯度张量同一。供体零钩子：初始全0，两步后微小context/state变化但终端scalar0/恢复0，不称预测收益或追加step救结论。formal100源/clean初始全RNG/完整订单/共享10/匹配容量与未来头信息角色/预算保存仍需另冻，不填卡跳依赖。' if x.startswith('4) ') else x) for x in lines]
lines=[('9) SSH79535/62639/17537和SFTP6854/47542/93212当前活动，GPU无健康训练；B当前CPU审核已完成。结束exit/bye核真实0后禁复用。password只实际提示后人类新凭据，禁文件/命令/自动/猜测/发送审视；新审批拒绝不换路，当前可用不称工具代码修复。' if x.startswith('9) ') else x) for x in lines]
(OUT/'研究接续状态.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
files=[OUT/'研究接续状态.md', OUT/'研究建议交流接续.json', OUT/'第二租期完整流GPU预检与D_B全状态审核最新.json',
       OUT/'第二租期完整流GPU预检与全状态保存实际结果.md',BASE/'complete_GPU_D_B_CPU_joint_audit.json',
       BASE/'B_full_state_CPU_original_receipt.json',BASE/'B_full_state_CPU_original_exit.json',Path(__file__)]
members={}
for p in files:
    name=('outputs/'+p.name if p.parent==OUT else 'evidence/'+p.name)
    target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    members[name]={'sha256':sha(target),'bytes':target.stat().st_size}
for p in (archive/'preceding_state').iterdir():members[p.relative_to(archive).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
(archive/'member_manifest.json').write_text(json.dumps(members,indent=2)+'\n')
with zipfile.ZipFile(archive/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
    for n in list(members)+['member_manifest.json']:z.write(archive/n,n)
with zipfile.ZipFile(archive/'records.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
proof={'actual_utc':now.isoformat(),'permanent_D':str(archive),'records_sha256':sha(archive/'records.zip'),
       'members':len(members),'SHA_CRC_unique_passed':True,'complete_GPU_D_B_full_CPU_passed':True,
       'formal100_started':False,'new_performance_scores':False}
(archive/'preservation_receipt.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
