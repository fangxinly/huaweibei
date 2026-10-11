from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'paired_official_fulltrain_v2_20261007T011543Z'))
from paired_fulltrain_evidence_candidate_v1 import read,write,sha,stage_association,capture_association

stamp=sys.argv[1]
base=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_official_prechecks_v2_actual_20261007T012116Z')
plan='8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb'
result={'clock_checked_utc':stamp,'status':'ACTUAL_BOTH_V2_PRECHECK_NATURAL0_AND_WHOLE_D_PASSED_ORIGINAL_OTHER_CPU_PENDING',
        'plan_sha256':plan,'D_root':str(base),'methods':{},'formal100_started':False,'final_TEST_executed':False,
        'overall_five_metric_superiority_or_lease_preservation_complete':False,'CPU_model_forward':False}
for method in ('minimal_fixed_F','careflow'):
    folder=base/method
    original,ex=stage_association(folder/'a/run',plan,method)
    cap,manifest=capture_association(folder/'a')
    whole={key:read(folder/f'actual_{key}_D_SHA_CRC_receipt.json') for key in ('clean','after2')}
    for key,record in whole.items():
        name='clean_initial_full' if key=='clean' else 'precheck_after2_full'
        assert record['sha256']==original[name]['sha256'] and record['bytes']==original[name]['bytes'] and record['CRC']
    result['methods'][method]={'root':original['root'],'natural_exit':ex,'stage_receipt_sha256':sha(folder/'a/run/out/actual_stage_receipt.json'),
      'capture_receipt_sha256':sha(folder/'a/capture_receipt.json'),'capture_actual_utc':cap['actual_utc'],
      'snapshot_sha256':cap['snapshot_sha256'],'capture_members':cap['members'],'whole_actual_D':whole,
      'optimizer_steps':original['optimizer_steps'],'dummy_and_same_instance_full_replay_error':original['DEV_dummy0vs7_and_same_instance_full_replay_error'],
      'DEV_true_labels_read':original['DEV_true_labels_read'],'cumulative_peak':original['final_budget'],
      'original_other_CPU_pending':True,'failed_first_capture_preserved':True}
out=Path('outputs/正式双方v2完整预检与保存实际接续.json')
write(out,result)
text=f'''actualclock {stamp}：双方官方MOSI v2两步预检均自然exit0，完整clean与after2实际D下载、全部SHA/ZIPCRC/唯一成员及原GPU capture/source/fullargv关联通过。\n\nF累计allocated/reserved 3924499968/4427087872B；CaReFlow 3918803968/4387241984B，均含全量保存与整件重放且无peakreset。229 DEV仅输入dummy，换dummy及整件重放误差0，未读DEV真标签。TRAIN1281用于预检拟合；非正式100完成、非五项收益。\n\nD {base}；四整件约5.93GB，capture大SHA引用与实际下载分明。B整件正在转存，完整原TorchCPU与B原capture/joint待过，未CPU模型前向。首次capture目录前缀错误导致自然1，失败记录保留；正确前缀另次COMPLETE/natural0已保存。\n\n正式训练只能在两方法全部原D/异节点CPU/两原capture联合门过后，另新根从clean公共初态、Adam空、scheduler0和全RNG新构造；绝不继承预检2步。双方共同100轮/4000更新/原共同订单/DEV批MSE strictmin earliest、不201择读出、不读TEST。v1峰值门失败原证据保持。\n'''
out.with_suffix('.md').write_text(text,encoding='utf-8')
status=Path('outputs/研究接续状态.md')
status.write_text(f'最新actualclock {stamp}：双方v2两步预检自然0/完整四整件D实际SHA-CRC通过，B整件转存与原CPU/capture/joint待完成；正式100尚未启动。先读《正式双方v2完整预检与保存实际接续.md/json》，v1失败原件保留，五项目标/最终TEST/租期保存未完成。\n\n'+status.read_text(encoding='utf-8'),encoding='utf-8')
write(base/'actual_v2_D_progress_20261007T0131Z.json',result)
print(result['status'],flush=True)
