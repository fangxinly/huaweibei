"""Freeze a distinct prospective paired fullTRAIN pipeline, before its new TRAIN labels.

No real dataset, checkpoint, Torch or GPU is opened here. Physical input-only ID evidence is bound.
"""
import argparse,ast,hashlib,importlib.util,json,shutil,sys,zipfile
from pathlib import Path
BASE=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
BACKUP=Path('D:/CodexBackups/selective_flow_20261003_1105')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--clock-utc',required=True);a=parser.parse_args()
    stamp=a.clock_utc[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
    root=BASE/'work'/('paired_official_fulltrain_v1_'+stamp);D=BACKUP/root.name
    if root.exists() or D.exists():raise RuntimeError('Fresh immutable full protocol required')
    if min(shutil.disk_usage(BASE).free,shutil.disk_usage(BACKUP).free)<6*1024**3:raise RuntimeError('Actual fresh local space floors')
    root.mkdir();D.mkdir()
    parent=BASE/'work/paired_fulltrain_preservation_sources_20261007T005109Z'
    for p in parent.iterdir():
        if p.is_file() and p.suffix in ('.py','.npy'):shutil.copy2(p,root/p.name)
    shutil.copy2(BASE/'work/paired_fulltrain_fresh_saved_joint_candidate_v1.py',root/'paired_fulltrain_fresh_saved_joint_candidate_v1.py')
    # New pipeline canonicalizes value-unused dtype marker bytes for BOTH methods.
    session=root/'paired_fulltrain_session_candidate.py';s=session.read_text(encoding='utf-8')
    s=s.replace("    matched=verify_public_encoder(model,assets/'assets/deberta-v3-base')", "    matched=verify_public_encoder(model,assets/'assets/deberta-v3-base')\n    for encoder in (model.dberta.transa,model.dberta.transv):encoder.embed_positions._float_tensor.zero_()")
    s=s.replace("        for encoder in (model.dberta.transa,model.dberta.transv):encoder.embed_positions._float_tensor.zero_()\n",'')
    s=s.replace("    clean_initial={'model':", "    if supervision:\n        labels=train.tensors[3].reshape(-1)\n        if labels.numel()!=1281 or not torch.isfinite(labels).all() or labels.min() < -3 or labels.max() > 3:\n            raise ValueError('Official TRAIN scalar finite original-scale -3..3 gate')\n    clean_initial={'model':")
    s=s.replace("'same_seed_does_not_mean_identical_task_state_across_architectures':True,", "'same_seed_does_not_mean_identical_task_state_across_architectures':True,\n             'dtype_only_positional_marker_values_zeroed_for_both_methods':True,\n             'label_numeric_representation':'Author float32 passthrough, no centering/scaling/clipping of targets',\n             'supervision_enabled':supervision,")
    session.write_text(s,encoding='utf-8')
    guard=root/'official_fulltrain_dev_guard_candidate.py';s=guard.read_text(encoding='utf-8')
    s=s.replace("dtype=np.float64)","dtype=np.float32).astype(np.float64)")
    s=s.replace("if not np.isfinite(values).all():raise ValueError('Nonfinite DEV target')", "if not np.isfinite(values).all() or (np.abs(values)>3).any():raise ValueError('Original-scale finite DEV target gate')")
    guard.write_text(s,encoding='utf-8')
    # Whole original TRAIN targets are retained for CPU role/precision association, never TEST.
    runtime=root/'paired_fulltrain_runtime_candidate_v1.py';s=runtime.read_text(encoding='utf-8')
    s=s.replace("    write(out/'actual_construction_receipt.json',session.construction_receipt)", "    np.save(out/'original_TRAIN_supervision.npy',session.train.tensors[3].detach().cpu().numpy().reshape(-1).astype(np.float32),allow_pickle=False)\n    write(out/'actual_construction_receipt.json',session.construction_receipt)")
    runtime.write_text(s,encoding='utf-8')
    cpu=root/'paired_fulltrain_CPU_audit_candidate_v1.py';s=cpu.read_text(encoding='utf-8')
    s=s.replace("    expected_lr = 1e-5", "    require(scheduler['_step_count']==steps+1 and len(scheduler['base_lrs'])==len(optimizer['param_groups']) and\n            all(float(x)==1e-5 for x in scheduler['base_lrs']),'Complete original scheduler base/count mismatch')\n    expected_lr = 1e-5")
    s=s.replace("    rng_digest = rng_sha", "    require(isinstance(checkpoint['rng']['python'],tuple) and len(checkpoint['rng']['python'])==3 and\n            checkpoint['rng']['python'][0]==3 and len(checkpoint['rng']['python'][1])==625,'Complete Python RNG schema')\n    require(isinstance(checkpoint['rng']['numpy'],tuple) and len(checkpoint['rng']['numpy'])==5 and\n            checkpoint['rng']['numpy'][0]=='MT19937' and checkpoint['rng']['numpy'][1].shape==(624,), 'Complete NumPy RNG schema')\n    rng_digest = rng_sha")
    s=s.replace("    keys = ('clean_initial_full'", "    targets=np.load(a.original_root/'out/original_TRAIN_supervision.npy',allow_pickle=False) if False else None\n    keys = ('clean_initial_full'")
    # Above intentionally does not execute NumPy before delayed import; drop unnecessary placeholder.
    s=s.replace("    targets=np.load(a.original_root/'out/original_TRAIN_supervision.npy',allow_pickle=False) if False else None\n",'')
    s=s.replace("    checkpoints = {key:", "    targets=np.load(a.original_root/'out/original_TRAIN_supervision.npy',allow_pickle=False)\n    require(targets.shape==(1281,) and targets.dtype==np.float32 and np.isfinite(targets).all() and (np.abs(targets)<=3).all(),'Original TRAIN scalar scale/representation')\n    checkpoints = {key:")
    cpu.write_text(s,encoding='utf-8')
    # Source-only scale lineage: originals unchanged, float32 author targets are common.
    old=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen/outputs/careflow_reproduction/official')
    author=old/'train_reflow_new.py';author_sha=sha(author)
    if author_sha!='d0266a55931fbae4329c2123b3b0f0d475799f31b8e202fa393ecef406ce841e':raise RuntimeError('Pinned original author training source mismatch')
    original_tree=ast.parse(author.read_text(encoding='utf-8-sig'));derived_tree=ast.parse((root/'careflow_train_dev_author_components_candidate.py').read_text())
    by_name=lambda t:{x.name:ast.dump(x,include_attributes=False) for x in t.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
    original_functions,derived_functions=by_name(original_tree),by_name(derived_tree)
    for name in ['InputFeatures','convert_to_features','get_appropriate_dataset','prep_for_training','batch_minmax','train_epoch','_forward_eval','multiclass_acc']:
        if original_functions[name]!=derived_functions[name]:raise RuntimeError('Author numerical/data source adaptation changed: '+name)
    position=old/'modules/position_embedding.py';position_tree=ast.parse(position.read_text());position_source=position.read_text()
    marker_reads=[n for n in ast.walk(position_tree) if isinstance(n,ast.Attribute) and n.attr=='_float_tensor']
    if len(marker_reads)!=1 or 'type_as(self._float_tensor)' not in position_source:raise RuntimeError('Positional marker has other numerical use')
    sources={}
    for p in root.glob('*.py'):
        tree=ast.parse(p.read_text(encoding='utf-8'));compile(tree,str(p),'exec')
        if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('kill','terminate','reset_peak_memory_stats') for n in ast.walk(tree)):raise RuntimeError('Forbidden stop/reset')
        sources[p.name]=sha(p)
    identity_D=BACKUP/'official_TRAIN_DEV_input_identity_actual_20261007T005631Z'
    identity_joint=identity_D/'complete_input_identity_A_D_capture_association.json';identity=read(identity_joint)
    if sha(identity_joint)!='dcc555595bcffe294038da4f7f0ecc5b299379237abf85f8662d5c514d1cbc4f':raise RuntimeError('Exact original physical input identity joint mismatch')
    shutil.copy2(identity_joint,root/'original_input_identity_A_D_joint.json')
    shutil.copy2(identity_D/'a/run/out/actual_stage_receipt.json',root/'original_official_INPUT_ONLY_identity_receipt.json')
    asset_plan=read(BASE/'work/minimal_fixed_staged_reference_v1_20261006T151117Z/runtime_candidate_plan.json')
    orders=root/'prospective_common_TRAIN_position_orders_seed128_100.npy'
    plan={'status':'PAIRED_FULLTRAIN_RUNTIME_EXECUTION_PROTOCOL_FROZEN','clock_source_freeze_utc':a.clock_utc,
          'experiment_scope':'A new paired official MOSI fullTRAIN, distinct from completed fold0 and232/201 experiments; no task weight reuse',
          'methods':['minimal_fixed_F','careflow'],'method_readout_locked':'Direct terminal prediction for both; no201 chosen head or residual readout',
          'task_seed':128,'orders_seed':128,'epochs':100,'updates':4000,'train_rows':1281,'dev_rows':229,'batch':32,'drop_last':True,'dev_batch':128,
          'DEV_selection':'author_batch_MSE_strict_earliest','DEV_numeric_precision':'Original author float32 target values cast float64 and FP32 predictions cast float64 for batch MSE; same both; old cached score was float32',
          'final_TEST_enabled':False,'residual_head_enabled':False,'OUTER_INNER_CAL_EVAL201_roles_enabled':False,
          'source_sha256':sources,'asset_sha256':asset_plan['asset_sha256'],'orders_file':orders.name,'orders_sha256':sha(orders),
          'official_train_dev_ID_identity_sha256':identity['official_train_dev_ID_identity_sha256'],'original_input_identity_joint_sha256':sha(identity_joint),
          'original_input_identity_source_sha256':sha(root/'original_official_INPUT_ONLY_identity_receipt.json'),
          'original_CPU_and_root_specific_capture_sources_frozen':True,'this_flag_proves_source_presence_not_actual_CPU_or_capture_execution':True,
          'assigned_gpu_UUID':{'minimal_fixed_F':'GPU-53696803-875e-eec8-2231-29db63579891','careflow':'GPU-417d3577-0525-788b-7296-0808a0f52012'},
          'original_CPU_gpu_UUID':'GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f',
          'conservative_lease_end_UTC':'2026-10-07T13:30:00+00:00','execution_budget_seconds':10800,'saving_reserve_seconds':7200,
          'human_lease_provenance_reference':'Human original second24h P4 lease in this chat approximately Oct7UTC13:35; conservative13:30 is not platform confirmation',
          'max_gpu_peak_bytes':6*1024**3,'remote_free_floor_bytes':4*1024**3,'local_D_free_floor_bytes':6*1024**3,
          'precheck_mixed_TRAIN_rows':[454,505,620,*range(29)],'precheck_two_steps_never_inherited_by_formal_training':True,
          'precheck_original_clean_and_after2_D_other_CPU_capture_joint_required_before_training':True,
          'task_weight_or_optimizer_reuse_from_old_experiments':False,'old_cached_baseline_NOT_reused_as_shared_order_model':True,
          'same_seed_not_identical_task_state_across_architectures':True,'baseline_parameter_count_not_claimed_equal_to_method':True,
          'author_source_sha256':author_sha,'archived_metric_source_SHA_separate':'d88739d1d1a224a2f159f29fa723cc24c347eadcf921e93bf57bd0e2cdaa3cb7',
          'author_fixed_configuration':'100/batch32/lr1e-5/ratio4/dropout.5/inter150/share100/3layers/Euler2/lossf.2/lossb.1/eps1e-3/seed128',
          'baseline_not_CLI_all_defaults_or_paper_recipe_confirmed':True,
          'target_source_lineage':'Original pinned public mosi.pkl; author raw float32 scalar passthrough. New TRAIN and post-prediction DEV finite range[-3,3] checked at actual authorized access; no labels inspected to freeze',
          'dtype_only_positional_marker_value_canonicalized_zero_both_methods':True,'marker_source_sha256':sha(position),
          'cumulative_peak_no_reset':True,'completed_preservation_required':['Natural0 original receipt','Entire resume and selected genuinely D downloaded','Other original TorchCPU state/Adam/RNG/history/order/arrays','Two COMPLETE natural0 root/source/argv captures','Fresh public-instance whole selected229 dummy/allRNG replay and D association'],
          'final_TEST_future_gate':'Only a separate frozen once-only source and both complete selected pipelines; no TEST-based structure/seed/threshold/readout changes',
          'historical_TEST_access_disclosed':'Old baseline runner has historical once TEST after DEV selection; this experiment does not claim previously-unseen or independent confirmation',
          'MOSI_only_MOSEI_not_implemented':True,'new_actual_model_or_gradient_or_checkpoint_or_score_observed':False}
    plan_path=root/'paired_fulltrain_execution_plan.json';write(plan_path,plan);plan_sha=sha(plan_path)
    remote='/data/coding/'+root.name
    for method in plan['methods']:
        args=['--root','/data/coding/paired_fulltrain_precheck_'+method+'_'+stamp,'--bundle',remote,
              '--assets','/data/coding/multimodal_flow_public_20261006T1341Z','--method',method,'--stage','precheck','--plan-sha',plan_sha]
        write(root/('reserved_precheck_'+method+'_arguments.json'),args)
    # Local pure-file/core mathematical contract validation. No Torch or official data.
    sys.path.insert(0,str(root))
    import numpy as np
    import paired_fulltrain_CPU_audit_candidate_v1 as cpu_module
    from common_budget_selection_candidate import train_batches,improves_earliest
    order=np.load(orders,allow_pickle=False)
    if order.shape!=(100,1281):raise RuntimeError('Common physical order shape')
    for row in order:
        batches,dropped=train_batches([int(x) for x in row])
        if len(batches)!=40 or len(dropped)!=1:raise RuntimeError('Shared40 updates/tail1 rule')
    if improves_earliest(.5,.5) or not improves_earliest(.5,.49):raise RuntimeError('Earliest strict rule')
    if cpu_module.dev_batch_metric(np,np.r_[np.zeros(128),np.ones(101)],np.zeros(229))!=.5:raise RuntimeError('Batch average weight changed')
    if not all(plan['source_sha256'][n]==sha(root/n) for n in sources):raise RuntimeError('Source bytes changed during freeze')
    with zipfile.ZipFile(root/'upload_bundle.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in root.iterdir():
            if f.is_file() and f.name!='upload_bundle.zip':z.write(f,f.name)
    with zipfile.ZipFile(root/'upload_bundle.zip') as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())):raise RuntimeError('Full upload source CRC/unique')
        for n in z.namelist():
            if hashlib.sha256(z.read(n)).hexdigest()!=sha(root/n):raise RuntimeError('Full upload source member SHA')
    result={'status':'PAIRED_OFFICIAL_MOSI_FULLTRAIN_SOURCE_PROTOCOL_FROZEN_ACTUAL_EXECUTION_GATES_PENDING','clock_source_freeze_utc':a.clock_utc,
            'local':str(root),'D':str(D),'reserved_remote_bundle_not_uploaded':remote,'plan_sha256':plan_sha,'upload_bundle_sha256':sha(root/'upload_bundle.zip'),
            'source_sha256':sources,'original_physical_OFFICIAL_input_identity_joint_SHA':sha(identity_joint),'local_source_and_pure_order_metric_contract_passed':True,
            'author_component_AST_identity_and_dtype_marker_only_type_use_passed':True,
            'actual_TRAIN_or_DEV_labels_read_for_this_new_pipeline':False,'actual_GPU_precheck_or_formal_training_executed':False,
            'next':'Fresh actual C/D and assigned A/C/B identity/fullargv/empty compute/source/assets/space/lease gates; then two clean prechecks each original D+otherCPU+capture joint before distinct fullTRAIN100'}
    write(root/'local_protocol_freeze_receipt.json',result)
    doc='''# 新双方官方MOSI fullTRAIN：源协议冻结，实际门待执行

唯一方法锁定为原完整fixed F的直接terminal读出，baseline锁定原CaReFlow配置seed128及确切训练源。新实验从公共预训练骨干与随机任务模块开始，TRAIN1281/DEV229、共同100×40更新、batch32丢尾1、同物理订单与DEV128/101批MSE strict最小且平局最早；不继承旧fold100/232/201模型、头、Adam或预检2步。此协议只MOSI，MOSEI与最终TEST未执行。

实际零标签官方行ID及公共资产已D绑定；双方标签原值经作者float32表示后用于损失/float64批MSE，不中心化/缩放/clip标签。源链证明双方同口径，实际TRAIN及预测先冻后的DEV还须原标量/finite/[-3,3]检查。共同float64选模不同于旧缓存float32，原缓存订单也不等于新共同订单，因此不冒旧缓存全流程重现或模型资格复用。

双方的positional `_float_tensor`原源只通过type_as使用dtype，未初始化值统一置0用于干净初态整字节/RNG复现；原作者源码不改，也不声明已实测对数值影响。任务参数容量如实分别报，不称完全同容量。原配置不是CLI全默认或已核论文同配方；旧TEST历史访问保留，未来一次正式TEST不冒全新独立确认。

预检每方新构造公共初态、正常/mixed原TRAIN32行两真实更新，全参数/实际梯度/Adam/scheduler/RNG和DEV229 dummy0/7/自己的整pt重放；全原件先D+异节点TorchCPU+真实COMPLETE双capture总门，才能正式另根从新相同clean初态/空Adam/step0训练。正式完成原latest/resume与整selected、4000更新/100轮历史/预测先冻后标签、独立CPU与fresh公共实例整selected重放及D捕获关联全必须过。

源flag只证明源存在，不能当实际预检/CPU/GPU/capture或预算通过。执行预算上限3h加至少2h保存、累计6GiB不reset、remote-D过程中4GiB与结束D6GiB门；真实≤5min assignedUUID/空compute/完整argv/源/资产/空间/人类租期来源门须执行。A/C为两独立GPU，B原TorchCPU；保守Oct7UTC13:30不冒平台确认。自然wait不杀健康任务，旧完成实验不重跑/传/改源。

当前只有源冻结和本地AST/纯合成订单/批MSE检查；没有新的真标签、模型、梯度或成绩。最终一个锁定方法与一个完整baseline管线及共同一次五项TEST仍另门；无201择头、调lambda/seed/feature救分，正式超过与整体研究/租期保存未完成。
'''
    (root/'protocol.md').write_text(doc,encoding='utf-8')
    shutil.copytree(root,D/'source_protocol');shutil.copy2(__file__,D/Path(__file__).name)
    out=BASE/'outputs'
    for ext,name in [('md','protocol.md'),('json','local_protocol_freeze_receipt.json')]:shutil.copy2(root/name,out/('正式双方官方MOSI_fullTRAIN源冻结接续.'+ext))
    pro=out/'正式CaReFlow比较来源与预算本地准备接续.json';v=read(pro)
    v['latest_paired_OFFICIAL_fullTRAIN_execution_protocol']={'status':result['status'],'local':str(root),'D':str(D),'plan_sha256':plan_sha,
          'upload_bundle_sha256':result['upload_bundle_sha256'],'clock_source_freeze_utc':a.clock_utc};write(pro,v)
    state=out/'研究接续状态.md';previous=state.read_text(encoding='utf-8')
    state.write_text('actualclock '+a.clock_utc+'：新双方官方MOSI fullTRAIN源协议已D冻；只原完整fixedF直接读出vs指定CaReFlow128，共同原官方ID/资产、TRAIN1281/DEV229/100×40drop1/订单/DEV批MSE earliest strictmin。新公共初态不继承旧fold100或预检2步。实际labels/GPU/CPU/capture未执行；先补读正式双方官方MOSI_fullTRAIN源冻结接续.md/json和来源预算最新指针；freshA/C/B实际物理/源/空间/执行3h+保存2h门待核，预检完整D/异CPU/双capture后才另根新fullTRAIN。dtype-only buffer双方统一0及共同float32targets→float64选模偏离旧缓存已明示。最终TEST仍禁，整体目标/保存未完成。\n\n'+previous,encoding='utf-8')
    control=D/'control';control.mkdir()
    for n in ['研究接续状态.md','正式CaReFlow比较来源与预算本地准备接续.json','正式双方官方MOSI_fullTRAIN源冻结接续.md','正式双方官方MOSI_fullTRAIN源冻结接续.json']:shutil.copy2(out/n,control/n)
    members={f.relative_to(D).as_posix():sha(f) for f in D.rglob('*') if f.is_file()};write(D/'member_SHA.json',members)
    with zipfile.ZipFile(D/'snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in [*members,'member_SHA.json']:z.write(D/n,n)
    with zipfile.ZipFile(D/'snapshot.zip') as z:
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())) or len(z.namelist())!=len(members)+1:raise RuntimeError('Full D freeze CRC/unique')
        for n,h in members.items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('Full D frozen member SHA')
    write(D/'preservation_receipt.json',{'clock_source_freeze_utc':a.clock_utc,'whole_ZIP_sha256':sha(D/'snapshot.zip'),'members':len(members)+1,
          'all_SHA_CRC_unique_passed':True,'scope':'New immutable paired source freeze, not new original GPU/CPU execution or scores'})
    print(json.dumps({'local':str(root),'D':str(D),'plan_sha256':plan_sha,'members':len(members)+1,'status':result['status']},ensure_ascii=False))

if __name__=='__main__':main()
