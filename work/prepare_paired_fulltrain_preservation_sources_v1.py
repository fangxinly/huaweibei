"""Build a new source-only candidate; preserve old frozen candidates unchanged."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

BASE = Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
BACKUP = Path('D:/CodexBackups/selective_flow_20261003_1105')
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()
def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--clock-utc', required=True); a = parser.parse_args()
    stamp = a.clock_utc[:19].replace('-', '').replace(':', '').replace(' ', 'T') + 'Z'
    require_space = 6 * 1024**3
    if shutil.disk_usage(BASE).free <= require_space or shutil.disk_usage(BACKUP).free <= require_space:
        raise RuntimeError('Fresh local preservation space floor')
    parent = BASE / 'work/paired_fulltrain_runtime_preparation_20261007T003225Z'
    root = BASE / 'work' / ('paired_fulltrain_preservation_sources_' + stamp)
    D = BACKUP / root.name
    if root.exists() or D.exists(): raise RuntimeError('Fresh candidate only')
    root.mkdir(); D.mkdir()
    for f in parent.iterdir():
        if f.is_file() and f.suffix in ('.py', '.npy'):
            shutil.copy2(f, root / f.name)
    added = ['paired_fulltrain_evidence_candidate_v1.py', 'paired_fulltrain_CPU_audit_candidate_v1.py',
             'paired_fulltrain_CPU_natural_wrapper_candidate_v1.py', 'paired_fulltrain_capture_candidate_v1.py',
             'paired_fulltrain_capture_natural_wrapper_candidate_v1.py', 'paired_fulltrain_saved_joint_candidate_v1.py',
             'paired_fulltrain_identity_input_only_candidate_v1.py','paired_fulltrain_fresh_selected_replay_candidate_v1.py',
             'paired_fulltrain_auxiliary_natural_wrapper_candidate_v1.py']
    for n in added: shutil.copy2(BASE / 'work' / n, root / n)
    runtime_path = root / 'paired_fulltrain_runtime_candidate_v1.py'
    source = runtime_path.read_text(encoding='utf-8')
    source = source.replace('from pathlib import Path', 'from pathlib import Path\nfrom paired_fulltrain_evidence_candidate_v1 import precheck_training_gate')
    source = source.replace('    return plan\n', "    if args.stage=='train':args.verified_original_precheck=precheck_training_gate(args,plan)\n    return plan\n", 1)
    start = source.index('    if not args.precheck_root or not args.precheck_joint or not args.precheck_joint_sha:')
    end = source.index("    if parent['clean_initial_state_sha256']", start)
    source = source[:start] + '    parent=args.verified_original_precheck\n' + source[end:]
    source = source.replace("    for row in order:train_batches([int(x) for x in row])", "    for row in order:train_batches([int(x) for x in row])\n    shutil.copyfile(args.bundle/plan['orders_file'],out/'original_shared_TRAIN_orders.npy')\n    if sha(out/'original_shared_TRAIN_orders.npy')!=plan['orders_sha256']:raise ValueError('Physical original orders copy mismatch')")
    source = source.replace("updates.append(step(rows));budget('precheck_optimizer_state_allocation')", "updates.append(dict(step(rows),rows=rows));budget('precheck_optimizer_state_allocation')")
    source = source.replace("optimizer_index_to_name=optimizer_index_to_name,missing_gradient_names=updates[-1]['gradient_missing_tensors']),", "optimizer_index_to_name=optimizer_index_to_name,missing_gradient_names=updates[-1]['gradient_missing_tensors'],\n                rng_sha256=rng_hash(rng())),")
    source = source.replace("    metadata={'method':args.method", "    final_rng=rng()\n    metadata={'method':args.method")
    source = source.replace("'missing_gradient_names':result['gradient_missing_tensors'],", "'missing_gradient_names':result['gradient_missing_tensors'],'rng_sha256':rng_hash(final_rng),")
    source = source.replace("'rng':rng(),'history':history", "'rng':final_rng,'history':history")
    source = source.replace("        frozen_sha=sha(path);y=session.guard.dev_labels_after_frozen_prediction(path,frozen_sha,state)", "        frozen_sha=sha(path)\n        with (out/'actual_DEV_prediction_frozen_before_labels.jsonl').open('a') as frozen_log:\n            frozen_log.write(json.dumps({'actual_utc':utc().isoformat(),'epoch':epoch+1,'path':str(path),'sha256':frozen_sha,'state_sha256':state,'true_labels_not_yet_accessed_for_this_epoch':True})+'\\n')\n        y=session.guard.dev_labels_after_frozen_prediction(path,frozen_sha,state)")
    runtime_path.write_text(source, encoding='utf-8')
    # Bind official identity before model construction and before train_supervision.
    session_path = root / 'paired_fulltrain_session_candidate.py'
    session = session_path.read_text(encoding='utf-8').replace('import importlib.util,json,pickle,sys', 'import importlib.util,json,pickle,sys,hashlib')
    session = session.replace('def construct(method,bundle,assets,plan):','def construct(method,bundle,assets,plan,supervision=True):')
    session = session.replace("    train=author.get_appropriate_dataset(guard.train_supervision())", "    train=author.get_appropriate_dataset(guard.train_supervision()) if supervision else train_inputs")
    session = session.replace('    guard=load_approved_official_roles(assets)', "    guard=load_approved_official_roles(assets)\n    identity=hashlib.sha256(json.dumps(guard.ids,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()\n    if identity!=plan['official_train_dev_ID_identity_sha256']:raise PermissionError('Official ID binding failed before model/TRAIN label access')")
    session_path.write_text(session, encoding='utf-8')
    # CPU Adam counts follow actual nonmissing-gradient records for baseline too.
    cpu_path = root / 'paired_fulltrain_CPU_audit_candidate_v1.py'
    cpu = cpu_path.read_text(encoding='utf-8')
    cpu = cpu.replace('def verify_parameters(torch, checkpoint, model, steps):', 'def verify_parameters(torch, checkpoint, model, steps, gradient_records=None):')
    cpu = cpu.replace("        s = optimizer['state'].get(index)\n        if s is None:", "        s = optimizer['state'].get(index)\n        expected_parameter_steps=steps if gradient_records is None else sum(n not in row['gradient_missing_tensors'] for row in gradient_records)\n        if s is None:\n            require(expected_parameter_steps==0, 'Adam missing despite recorded actual gradient: '+n)")
    cpu = cpu.replace("require(float(s['step']) == steps,", "require(float(s['step']) == expected_parameter_steps,")
    # Empty clean Adam is appropriate even though all parameter names appear in initial state.
    cpu = cpu.replace("            require(expected_parameter_steps==0,", "            require(steps==0 or expected_parameter_steps==0,")
    cpu = cpu.replace("checks['after2'] = verify_parameters(torch, after, after['model'], 2)", "checks['after2'] = verify_parameters(torch, after, after['model'], 2,receipt['updates'])")
    cpu = cpu.replace("checks['resume'] = verify_parameters(torch, resume, resume['model'], 4000)", "original_steps=[json.loads(x) for x in (a.original_root/'out/actual_TRAIN_steps.jsonl').read_text().splitlines()]\n        require(len(original_steps)==4000,'Original 4000 gradient-step records absent')\n        checks['resume'] = verify_parameters(torch, resume, resume['model'], 4000,original_steps)")
    # State absence is tested against all actual updates, rather than only the last gradient.
    cpu = cpu.replace("require(actual_missing <= missing,", "require(actual_missing <= missing,")
    cpu = cpu.replace("        with np.load(a.original_root / 'out/precheck_DEV_dummy_full_replay.npz', allow_pickle=False) as z:", "        identity=read(a.original_root/'out/actual_official_row_identity.json')\n        require(identity['sha256']==receipt['official_row_ID_identity_sha256']==plan['official_train_dev_ID_identity_sha256'],'Precheck official identity binding')\n        require(hashlib.sha256(json.dumps(identity['ids'],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==identity['sha256'],'Precheck physical full ID SHA')\n        with np.load(a.original_root / 'out/precheck_DEV_dummy_full_replay.npz', allow_pickle=False) as z:")
    cpu = cpu.replace("require(z['prediction_dummy0'].shape == (229,)", "require(z['row_ids'].tolist()==identity['ids']['dev'] and z['prediction_dummy0'].dtype==np.float32 and z['prediction_dummy0'].shape == (229,)")
    cpu_path.write_text(cpu, encoding='utf-8')
    sources = {}
    for f in root.glob('*.py'):
        tree = ast.parse(f.read_text(encoding='utf-8')); compile(tree, str(f), 'exec')
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('kill','terminate','reset_peak_memory_stats') for n in ast.walk(tree)):
            raise RuntimeError('Forbidden task interruption/peak reset source')
        sources[f.name] = sha(f)
    # This actual local subprocess rejection occurs before Torch, no real assets/model/labels.
    fixture = root / 'synthetic_negative_fixture'; fixture.mkdir()
    write(fixture / 'paired_fulltrain_execution_plan.json', {'status':'LOCAL_SOURCE_PREPARATION_NOT_EXECUTION_FREEZE'})
    local_target = Path('/data/coding/paired_fulltrain_negative_preservation_fixture_' + stamp)
    argv = [sys.executable, str(runtime_path), '--root', str(local_target), '--bundle', str(fixture), '--assets', str(fixture),
            '--method','minimal_fixed_F','--stage','train','--plan-sha',sha(fixture/'paired_fulltrain_execution_plan.json')]
    result = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8')
    if result.returncode == 0 or 'Local preparation cannot launch' not in result.stderr or local_target.exists():
        raise RuntimeError('Local pre-Torch negative fixture failed')
    (fixture/'stdout.txt').write_text(result.stdout,encoding='utf-8');(fixture/'stderr.txt').write_text(result.stderr,encoding='utf-8')
    write(fixture/'actual_negative_receipt.json',{'clock_start_utc':a.clock_utc,'argv':argv,'exit_code':result.returncode,'expected_source_gate_rejection_before_Torch':True,'no_official_data_or_root_created':True})
    sys.path.insert(0,str(root))
    import paired_fulltrain_evidence_candidate_v1 as evidence
    import paired_fulltrain_CPU_audit_candidate_v1 as cpu_module
    import numpy as np
    # Meaningful pure-file mutation tests: argv/receipt mismatch, traversal/duplicate ZIP,
    # earliest strict tie and author batch weighting; no Torch deserialization or real data.
    for bad in ('../outside', '/absolute', 'x\\y', 'C:escape', 'x//y'):
        try: evidence.safe_member(bad)
        except ValueError: pass
        else: raise RuntimeError('Unsafe ZIP name accepted')
    from official_fulltrain_dev_guard_candidate import FullTrainDevGuard
    class SyntheticInputOnlyRecord:
        def __init__(self,identity):self.identity=identity
        def __getitem__(self,index):
            if index==1:raise RuntimeError('Synthetic true label access is forbidden')
            if index==0:return (['synthetic'],np.zeros((2,2)),np.zeros((2,3)))
            if index==2:return self.identity
            raise IndexError(index)
    synthetic_guard=FullTrainDevGuard([SyntheticInputOnlyRecord('TRAIN_'+str(i)) for i in range(1281)],
                                    [SyntheticInputOnlyRecord('DEV_'+str(i)) for i in range(229)])
    synthetic_guard.inputs_only('train');synthetic_guard.inputs_only('dev')
    identity_tree=ast.parse((root/'paired_fulltrain_identity_input_only_candidate_v1.py').read_text())
    if any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value in (1,'test') for n in ast.walk(identity_tree)):
        raise RuntimeError('Input-only collector contains label or TEST entry indexing')
    duplicate = fixture/'duplicate.zip'
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with zipfile.ZipFile(duplicate,'w') as z:z.writestr('same','a');z.writestr('same','b')
    try: evidence.zip_audit(duplicate)
    except ValueError: pass
    else: raise RuntimeError('Duplicate ZIP accepted')
    predictions=np.concatenate([np.zeros(128,dtype=np.float32),np.ones(101,dtype=np.float32)])
    if cpu_module.dev_batch_metric(np,predictions,np.zeros(229))!=.5:raise RuntimeError('Author common batch weighting altered')
    fake=fixture/'association';(fake/'out').mkdir(parents=True)
    command=['python','runner.py','--root','/data/coding/paired_fulltrain_fake','--bundle','/bundle','--plan-sha','a'*64]
    receipt={'argv':command[1:],'root':command[command.index('--root')+1],'source_bundle':'/bundle','plan_sha256':'a'*64,'method':'careflow'}
    write(fake/'out/actual_stage_receipt.json',receipt)
    write(fake/'actual_child_launch.json',{'pid':123,'full_argv':command})
    write(fake/'wrapper_actual_start.json',{'child_full_argv':command})
    write(fake/'natural_exit.json',{'child_pid':123,'exit_code':0,'natural_exit':True,'receipt_sha256':sha(fake/'out/actual_stage_receipt.json'),'full_argv':command})
    evidence.stage_association(fake,'a'*64,'careflow')
    broken=evidence.read(fake/'natural_exit.json');broken['receipt_sha256']='b'*64;write(fake/'natural_exit.json',broken)
    try:evidence.stage_association(fake,'a'*64,'careflow')
    except ValueError:pass
    else:raise RuntimeError('Unlinked natural exit accepted')
    # Call graph/order audit: validation gate cannot be reached after construction.
    runtime_tree=ast.parse(source);run_fn=next(n for n in runtime_tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    calls={n.func.id:n.lineno for n in ast.walk(run_fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('validate','construct')}
    if not calls['validate']<calls['construct']:raise RuntimeError('Early precheck gate misplaced')
    receipt={'status':'LOCAL_PAIRED_CPU_CAPTURE_JOINT_FRESH_REPLAY_IDENTITY_SOURCE_PREPARATION_NOT_EXECUTION_FREEZE','clock_start_utc':a.clock_utc,
             'local':str(root),'D':str(D),'source_sha256':sources,'AST_compilation_passed':True,
             'local_pre_Torch_negative_rejection_passed':True,'synthetic_ZIP_argv_receipt_and_batch_metric_mutations_passed':True,
             'synthetic_input_only_label_sentinel_passed':True,'identity_collector_has_no_label_or_TEST_entry_indexing_AST':True,
             'fresh_replay_constructs_with_supervision_False_source_only':True,
             'original_precheck_joint_gate_before_model_and_TRAIN_label_access':True,'official_ID_gate_before_model_and_TRAIN_label_access':True,
             'Torch_or_model_imported':False,'original_checkpoints_deserialized':False,'official_data_or_labels_read':False,
             'new_GPU_or_original_CPU_or_remote_capture_executed':False,'new_results':False,
             'remaining':['Actual fresh public-instance selected whole-model replay and D capture joint','Actual official TRAIN/DEV identity/scale/asset binding',
                          'End-to-end original checkpoint audit on actual TorchCPU, capture and D joint are source only until run',
                          'Complete protocol freeze plus actual fresh physical/resource/execution+2h gates before new fitting']}
    write(root/'local_preservation_source_receipt.json',receipt)
    doc='''# 正式双方原CPU、捕获与D关联：源码准备

新增独立TorchCPU原整文件审核、CPU自然退出wrapper、新root/source/fullargv捕获、COMPLETE后自然退出wrapper及D总关联源，并补零标签官方ID采集、fresh公共实例selected整文件重放和对应自然wrapper。当前仅AST编译、本地非冻结协议拒绝和合成ZIP/原argv-receipt/批MSE/标签sentinel保护核验；没有Torch导入、真实模型读取、数据标签、GPU/异节点CPU任务或remote capture。合成拒绝是预期源门，不是审批或GPU故障。

新的训练源在模型构造与真实TRAIN标签前核原预检natural0、receipt/exit、D-异节点CPU-capture总门与订单/身份。构造源也把官方TRAIN/DEV ID核验前移到模型和标签前。原候选与完成实验未改。预检仍从干净初态执行2步，正式100轮仍新构造相同初态/allRNG、空Adam/step0，不继承预检2步。

原CPU源将核整文件SHA/ZIP CRC/唯一成员再Torch载入CPU，参数身份/完整Adam两矩/真实各参数步数/scheduler/allRNG；100轮所有订单、4000更新、DEV预测先冻后标签、共同float64两批MSE及earliest strictmin，完整latest/best-state与selected文件关联。baseline实际缺梯度的参数如实保留，不称其每个参数均4000步。CPU仅张量/数组审核，非模型前向。

捕获只完成根及冻结源，保存真实UUID/compute/完整进程argv/空间；小文件全ZIP，整模型稳定inode/SHA大引用明确不等于下载。D总门要求实际完整模型文件另已下载D、fresh整SHA/CRC，与双原COMPLETE/natural0捕获和原异节点CPU loaded-file关联。完成源不等于完成实际门。

零标签身份采集仅索引原TRAIN/DEV input/ID，无真标签或TEST条目；whole可信pickle确实物化其它角色字节，不能冒从未进内存。ID/shape本身不能证实标签尺度。fresh重放源新公共实例、FIT统计重建、strict整selected载入229 DEV dummy0/7，参数/统计/allRNG不变、不监督或更新；这些全是待实际执行的源。实际fresh重放后D捕获总门仍待补齐。

仍需实际官方TRAIN/DEV ID/尺度/公共资产绑定、全部实际端到端门、完整执行协议及fresh物理资源/预算/执行加2h保存余量。不得直接启动；无新五项成绩或最终TEST。暂停旧加权仿射扩容与201救分，整体目标和租期保存仍未完成。
'''
    (root/'preparation.md').write_text(doc,encoding='utf-8')
    shutil.copytree(root,D/'source_candidate');shutil.copy2(__file__,D/Path(__file__).name)
    outputs=BASE/'outputs'
    for name, file in [('正式双方原CPU捕获与D关联源准备接续.md','preparation.md'),('正式双方原CPU捕获与D关联源准备接续.json','local_preservation_source_receipt.json')]:
        shutil.copy2(root/file,outputs/name)
    pro=outputs/'正式CaReFlow比较来源与预算本地准备接续.json';value=json.loads(pro.read_text(encoding='utf-8'))
    value['latest_paired_original_CPU_capture_joint_source_candidate']={'status':receipt['status'],'local':str(root),'D':str(D),'clock_start_utc':a.clock_utc};write(pro,value)
    short=outputs/'研究接续状态.md';previous=short.read_text(encoding='utf-8')
    short.write_text('本地准备actualclock '+a.clock_utc+'：原TorchCPU/完整capture/natural0/D joint、零标签官方ID及fresh公共实例整selected replay候选AST和合成门过；预检总门与官方ID已前移到模型/真实TRAIN标签前。新源仅本地准备，未Torch/数据/新模型或remote任务。先补读正式双方原CPU捕获与D关联源准备接续.md/json及来源预算接续最新指针；实际官方身份/尺度、fresh重放后D关联与完整协议/资源/保存门未过，不启动旧完成实验，整体研究与租期保存未完成。\n\n'+previous,encoding='utf-8')
    control=D/'control';control.mkdir()
    for name in ['研究接续状态.md','正式CaReFlow比较来源与预算本地准备接续.json','正式双方原CPU捕获与D关联源准备接续.md','正式双方原CPU捕获与D关联源准备接续.json']:
        shutil.copy2(outputs/name,control/name)
    members={f.relative_to(D).as_posix():sha(f) for f in D.rglob('*') if f.is_file()};write(D/'member_SHA.json',members)
    with zipfile.ZipFile(D/'snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in [*members,'member_SHA.json']:z.write(D/n,n)
    with zipfile.ZipFile(D/'snapshot.zip') as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())) or len(z.namelist())!=len(members)+1:raise RuntimeError('Local saved ZIP invalid')
        for n,h in members.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('Local complete ZIP/member SHA')
    write(D/'preservation_receipt.json',{'clock_start_utc':a.clock_utc,'ZIP_SHA':sha(D/'snapshot.zip'),'members':len(members)+1,'CRC_unique_all_SHA':True,
         'scope':'Only local candidate sources and synthetic negative fixtures; no original remote research audit/capture'})
    print(json.dumps({'D':str(D),'local':str(root),'members':len(members)+1,'status':receipt['status']},ensure_ascii=False))

if __name__=='__main__':main()
