from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'paired_official_fulltrain_v2_20261007T011543Z'))
from paired_fulltrain_evidence_candidate_v1 import read,write,sha,require,PRECHECK_JOINT
stamp=sys.argv[1]
base=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_official_prechecks_v2_actual_20261007T012116Z')
result={'status':'ACTUAL_BOTH_V2_PRECHECK_GPU_D_ORIGINAL_OTHER_CPU_TWO_CAPTURES_COMPLETE',
  'clock_checked_utc':stamp,'D_root':str(base),'plan_sha256':'8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb',
  'methods':{},'CPU_model_forward':False,'final_TEST_access':False,'formal100_completion':False,
  'overall_five_metric_superiority_or_lease_preservation_complete':False,
  'supplementary_actual_physical_order_audit_SHA':sha(base/'actual_precheck_supplementary_physical_order_audit.json')}
for method in ('minimal_fixed_F','careflow'):
    folder=base/method; file=folder/'actual_original_precheck_GPU_D_B_CPU_capture_joint.json'; j=read(file)
    require(j['status']==PRECHECK_JOINT and j['method']==method,'Original saved joint actual status')
    gpu=read(folder/'a/run/out/actual_stage_receipt.json'); cpu=read(folder/'b/run/out/actual_stage_receipt.json')
    result['methods'][method]={'actual_saved_joint':j,'saved_joint_sha256':sha(file),
       'GPU_natural_exit':read(folder/'a/run/natural_exit.json'),'CPU_natural_exit':read(folder/'b/run/natural_exit.json'),
       'CPU_original_checks':cpu['checks'],'GPU_cumulative_peaks':gpu['final_budget'],
       'GPU_dummy_and_own_complete_replay_error':gpu['DEV_dummy0vs7_and_same_instance_full_replay_error'],
       'DEV_true_labels_read':gpu['DEV_true_labels_read'],'actual_original_cpu_capture_members':j['captures']['b']['members'],
       'actual_original_gpu_capture_members':j['captures']['a']['members'],
       'failed_first_capture_directory_prefix_error_preserved':True}
dest=base/'actual_both_v2_prechecks_complete_association.json'; require(not dest.exists(),'Preserve original complete association');write(dest,result)
result['association_sha256']=sha(dest)
out=Path('outputs/正式双方v2完整预检与保存实际接续.json');write(out,result)
text=f'''actualclock {stamp}：双方官方MOSI v2两步预检与完整保存联合通过。正式100另根启动情况另见《正式双方官方fullTRAIN100实际训练接续》，此报告只证明预检及原件保存。\n\nF GPU child2557 UTC01:22:44.235936自然0；B原TorchCPU child31358 UTC01:43:07.158716自然0。CaReFlow GPU child603 UTC01:22:49.924840自然0；B原CPU child31431 UTC01:44:01.753275自然0。四份完整clean/after2共5931787667bytes真正下载D并转存B，全SHA/ZIPCRC/唯一成员、完整模型/Adam两矩step2/scheduler/四种RNG/参数身份/数组和源全部通过。CPU是原整件反序列化及张量审核，没有CPU模型构造或前向。\n\nF GPU/CPU原capture 46/61成员，实际01:23:35.644055/01:43:49.494129；CaReFlow 46/61，实际01:23:38.166781/01:44:34.938691，均先COMPLETE和自然0 receipt，再完整ZIP成员SHA/CRC/唯一与原件关联。目录后缀仅命名，实际时刻以receipt为准。首次错误capture前缀导致自然1已保存，不冒成功；正确前缀另次通过。大稳定inode SHA引用不冒新下载，四份整件物理下载另有独立D审核。\n\nF累计allocated/reserved 3924499968/4427087872B；CaReFlow 3918803968/4387241984B，均低原6GiB，含构造、两更新、全量保存与自己的完整pt重放，无peakreset。F364参数/374state/364Adam；CaReFlow325参数/333state/323Adam，原两个未前向pooler梯度None已声明保留，不冒全325都有Adam。F预检24个供体参数零梯度仅原记录，不冒全部路径非退化或收益。\n\n两方法同一物理100×1281订单全SHA与原首批/预声明混合批行号补充审核过；只TRAIN1281标签用于预检拟合，DEV229输入dummy0/7与整件重放误差0、无DEV真标签或五项成绩、未Test/CAL/EVAL201。\n\nD {base}\nF joint SHA {result['methods']['minimal_fixed_F']['saved_joint_sha256']}\nCaReFlow joint SHA {result['methods']['careflow']['saved_joint_sha256']}\n双方汇总实际关联SHA {result['association_sha256']}\n\nv1累计峰值失败原件及第16usage失败保持。全五项正式超过/最终TEST/MOSEI/租期动态保存仍未完成。\n'''
out.with_suffix('.md').write_text(text,encoding='utf-8')
(base/'actual_both_v2_prechecks_complete_report.md').write_text(text,encoding='utf-8')
print(result['status'],result['association_sha256'],flush=True)
