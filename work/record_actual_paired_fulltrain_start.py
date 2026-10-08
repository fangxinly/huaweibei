"""Link genuine downloaded startup originals to actual saved precheck joints.

No model/labels/arrays; not a completed-task capture or superiority claim.
"""
from pathlib import Path
import sys,hashlib,zipfile,json,shutil
sys.path.insert(0,str(Path(__file__).parent/'paired_official_fulltrain_v2_20261007T011543Z'))
from paired_fulltrain_evidence_candidate_v1 import read,sha,write,require,zip_audit,plan_gate
stamp=sys.argv[1]
base=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_formal_actual_dispatch_v3_20261007T015004Z')
pre=base.parent/'paired_official_prechecks_v2_actual_20261007T012116Z'
bundle=Path(__file__).parent/'paired_official_fulltrain_v2_20261007T011543Z'
plan_sha='8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb'
plan=plan_gate(bundle,plan_sha)
dispatch=read(base/'exact_formal_dispatch_arguments_receipt.json')
failure=base.parent/'paired_formal_actual_dispatch_v2_20261007T014523Z/actual_failed_operator_D_complete_original_file_association.json'
require(read(failure)['status']=='ACTUAL_BOTH_ROOT_PREFIX_FAILED_WRAPPER_ORIGINAL_D_FILES_PASSED_NO_TRAIN_CHILD','Original wrong-root failure preserved')
result={'status':'ACTUAL_BOTH_OFFICIAL_FULLTRAIN100_RUNNING_FROM_CLEAN_INITIALIZATION_NOT_COMPLETE',
        'clock_checked_utc':stamp,'D_startup_evidence_root':str(base),'frozen_source_bundle':str(bundle),
        'remote_source_bundle':'/data/coding/'+bundle.name,'plan_sha256':plan_sha,'methods':{},
        'both_actual_precheck_complete_association_SHA':sha(pre/'actual_both_v2_prechecks_complete_association.json'),
        'original_operator_root_prefix_failure_D_association_SHA':sha(failure),
        'first_wrong_root_wrappers_did_not_launch_train_children':True,'first_wrong_root_wrapper_exit_codes_unobserved':True,
        'frozen_model_runtime_optimizer_selection_protocol_not_changed_to_fix_root':True,
        'formal100_complete':False,'final_TEST_executed':False,'final_selected_whole_D_original_CPU_fresh_replay_pending':True,
        'overall_five_metric_superiority_or_lease_preservation_complete':False,
        'strong_dynamic_saving_windows_UTC':['2026-10-07T09:30:00+00:00','2026-10-07T11:30:00+00:00','2026-10-07T13:00:00+00:00'],
        'conservative_lease_end_UTC':'2026-10-07T13:30:00+00:00','this_is_not_platform_lease_confirmation':True}
for method in plan['methods']:
    r=base/method/'run'; launch=read(r/'actual_child_launch.json'); w=read(r/'wrapper_actual_start.json'); start=read(r/'out/actual_training_start.json')
    original=read(pre/method/'a/run/out/actual_stage_receipt.json')
    observed=read(r/'actual_startup_observation_20261007T015458Z.json')
    args=read(base/(method+'_formal_child_arguments.json'))
    own=dispatch['methods'][method]
    require(launch['full_argv']==w['child_full_argv'] and launch['full_argv'][2:]==args,'Original complete child argv/D frozen argument equality')
    require(w['child_args_sha256']==sha(base/(method+'_formal_child_arguments.json'))==own['child_arguments_sha256'] and
            w['child_source_sha256']==plan['source_sha256']['paired_fulltrain_runtime_candidate_v1.py'],'Original child source/argv exact SHA')
    require(start['initial_state_sha256']==original['clean_initial_state_sha256'] and start['initial_rng_sha256']==original['clean_initial_rng_sha256'] and
            start['optimizer_steps']==0 and start['clean_initial_matches_original_precheck'] and start['precheck_two_steps_not_inherited'],'Actual fresh clean model/empty Adam/scheduler/all RNG gate')
    require(observed['actual_training_start']==start and observed['child_pid']==launch['pid'] and observed['observed_completed_updates']>0 and
            observed['not_COMPLETE_capture_or_completed_training'],'Actual active completed-update prefix observation')
    require(observed['root']==own['root'] and observed['full_argv']==launch['full_argv'],'Actual root/observation association')
    result['methods'][method]={'root':own['root'],'actual_child_pid':launch['pid'],'actual_wrapper_pid':read(r/'actual_formal_dispatch.json')['wrapper_pid'],
        'actual_child_launch_utc':launch['actual_utc'],'actual_cold_training_start':start,
        'actual_observation_utc':observed['actual_utc'],'observed_completed_updates':observed['observed_completed_updates'],
        'actual_selection_progress_only':observed['actual_progress'],'actual_cumulative_budget':observed['last_actual_cumulative_budget'],
        'precheck_joint_sha256':own['precheck_joint_sha256'],'original_startup_observation_sha256':sha(r/'actual_startup_observation_20261007T015458Z.json'),
        'nvidia_compute_PID_not_assumed_container_child_PID':True,'whole_checkpoint_final_saving_pending':True}
write(base/'actual_both_formal_start_D_original_association.json',result)
result['actual_start_D_association_sha256']=sha(base/'actual_both_formal_start_D_original_association.json')
out=Path('outputs/正式双方官方fullTRAIN100实际训练接续.json');write(out,result)
text=f'''actualclock {stamp}：新双方官方MOSI fullTRAIN100已真实启动并健康更新，禁止重复启动或停止。整体五项超过目标、100完成/最终保存/最终TEST仍未完成。\n\n冻结源 /data/coding/{bundle.name}，plan SHA {plan_sha}。与完成的fold100、FIT232/201及v1失败不同的新实验：原fixed F直接terminal vs指定作者固定CaReFlow配置seed128；双方官方TRAIN1281/DEV229、同100×40×32/drop尾1=4000更新、同物理订单、同DEV128/101批MSE float64算术均值strictmin/平局最早。无201读出选择或残差头、无TEST。baseline非CLI全默认/已核论文同配方；双方text nonreentrant gradient checkpoint/RNG保留和dtype-only marker0适配已预声明，未声称旧baseline数值等价或参数容量相等。\n\nF root {result['methods']['minimal_fixed_F']['root']}，wrapper2784/child2785；actual cold start01:52:56.248981。CaReFlow root {result['methods']['careflow']['root']}，wrapper806/child807；actual cold start01:52:58.656119。双方各自完整model state/all RNG与原clean预检精确一致，Adam空/scheduler0/step0，新构造绝不继承预检2步。\n\n原观察UTC01:55:02.781718/01:55:02.777681：F125与CaReFlow127次更新，前三轮已完成；实际累计reserved峰值4525654016/4481613824B低于原6GiB，无peakreset、无Traceback。此时的DEV MSE仅选模过程，不是最终五项成绩或TEST。NVML compute PID与容器child PID不同命名空间，不冒二者相同。不是此报告写入时刻的实时三机健康。\n\n预检全部保存门已过：D {pre}，四整件5931787667B真实D/B，全SHA/ZIPCRC/唯一/完整参数Adam两矩step2/scheduler/RNG/数组/source/fullargv/两原capture通过，原B TorchCPU自然0非CPU模型前向。原capture F46/61、CaReFlow46/61，actual receipts与source全部关联。先读《正式双方v2完整预检与保存实际接续》。\n\n第一次formal派发误用paired_fulltrain100_根前缀，冻结wrapper在TRAIN子进程/模型/输出目录前拒绝，完整原9文件/源/fullargv/fresh检查/Traceback与原manifest真实D保存，退出码没有原wait记录所以保持未知；不冒已训练或新审批拒绝。corrected fresh root采用paired_fulltrain_train100_，模型/runtime/训练源与协议不改，原失败目录不改不删，所有三机fresh物理/全source-assets/空间/人类保守13:30与执行3h+至少2h保存余量重新通过后启动。\n\n本轮启动原件 D {base}；actual startup关联SHA {result['actual_start_D_association_sha256']}。只启动证据，不冒最终自然COMPLETE捕获或已保存final weights。自然完成后立即完整resume+selected真D下载、异节点原CPU、双实际COMPLETE/natural0捕获、fresh公共实例整selected229 dummy/state/allRNG重放及D关联，再另冻最终一次TEST协议。原10分钟观察，健康不停止/不重启，结束/新失败及时处理。09:30/11:30/13:00 UTC强化动态保存待做；保守13:30不是平台确认。整体研究与租期保存未完成。\n'''
out.with_suffix('.md').write_text(text,encoding='utf-8');(base/'actual_formal_start_report.md').write_text(text,encoding='utf-8')
state=Path('outputs/研究接续状态.md')
lead=f'最新actualclock {stamp}：双方新官方fullTRAIN100真实从clean开始，F root {result["methods"]["minimal_fixed_F"]["root"]} child2785；CaReFlow root {result["methods"]["careflow"]["root"]} child807。UTC01:55实际125/127更新、前三轮/峰值<6GiB；严禁重复启动或停健康。先读《正式双方官方fullTRAIN100实际训练接续.md/json》及《正式双方v2完整预检与保存实际接续.md/json》。预检完整D/B原CPU/双capture已过，第一次仅bad-root wrapper拒绝保留无训练child；100完成、final完整保存/原CPU/fresh重放与最终TEST/五项正式超过/租期强化保存仍待做。\n\n'
state.write_text(lead+state.read_text(encoding='utf-8'),encoding='utf-8')
prov=Path('outputs/正式CaReFlow比较来源与预算本地准备接续.json');prior=read(prov)
prior['latest_actual_fullTRAIN100_running_pointer']={'actual_clock':stamp,'continuation_file':str(out.resolve()),'actual_startup_D':str(base),'plan_sha256':plan_sha,'formal100_complete':False,'final_TEST_executed':False,'do_not_relaunch_or_stop_healthy':True}
write(prov,prior)
md=prov.with_suffix('.md');md.write_text('最新实际：双方新官方fullTRAIN100已真实开始；先读《正式双方官方fullTRAIN100实际训练接续.md/json》与v2预检完整保存接续。仅actual启动/运行，未完成或最终TEST/五项超过；旧本地准备段保留为历史。\n\n'+md.read_text(encoding='utf-8'),encoding='utf-8')
print(result['status'],result['actual_start_D_association_sha256'],flush=True)
