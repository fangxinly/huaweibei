from pathlib import Path
import json,sys,hashlib
d=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_fulltrain100_complete_actual_20261007T030311Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
record={'status':'ACTUAL_BOTH_FULLTRAIN100_ORIGINAL_COST_AND_SELECTION_ONLY_CPU_FRESH_PENDING','clock_checked_utc':sys.argv[1],'methods':{}}
for m in ('minimal_fixed_F','careflow'):
 p=d/m/'a/run/out/actual_stage_receipt.json';r=json.loads(p.read_text(encoding='utf-8'));meta=r['metadata']
 assert r['status']=='ACTUAL_PAIRED_METHOD_FULLTRAIN100_COMPLETE_PRESERVATION_PENDING_NO_FINAL_TEST'
 record['methods'][m]={'original_receipt_sha256':sha(p),'selected_epoch':meta['best_epoch'],'fixed_DEV_batch_MSE_selection_only':meta['best_metric'],'parameters':sum(v['numel'] for v in meta['parameter_info'].values()),'parameter_tensors':len(meta['parameter_info']),'epochs':100,'optimizer_steps':4000,'elapsed_seconds_including_complete_save_own_selected_replay':r['elapsed_seconds'],'allocated_peak':r['final_budget']['allocated_peak'],'reserved_peak':r['final_budget']['reserved_peak'],'whole_D_SHA_CRC_receipt_sha256':sha(d/m/'actual_D_GPU_whole_SHA_CRC_source_capture_receipt.json')}
f,c=[record['methods'][x] for x in ('minimal_fixed_F','careflow')]
record.update({'F_minus_C_DEV_selection_MSE':f['fixed_DEV_batch_MSE_selection_only']-c['fixed_DEV_batch_MSE_selection_only'],'F_vs_C_elapsed_ratio':f['elapsed_seconds_including_complete_save_own_selected_replay']/c['elapsed_seconds_including_complete_save_own_selected_replay'],'same_order_4000_updates_and_selection_not_same_parameter_capacity_or_wall_time':True,'new_five_metric_or_TEST_results':False,'model_or_readout_changed_in_response_to_DEV':False,'whole_files_D_saved_bytes':7415841507,'B_other_CPU_and_fresh_whole_selected_replay_pending':True,'overall_five_metric_superiority_and_lease_preservation_complete':False})
p=d/'actual_original_training_cost_and_fixed_selection_only.json';assert not p.exists();p.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
md='双方新官方fullTRAIN100实际完成，完整保存链尚待B原CPU和fresh公共实例重放。\n\n| 固定模型 | 参数 | 更新 | 固定best | DEV批MSE，仅选模 | 含整保存/自身重放耗时 | 累计reserved |\n|---|---:|---:|---:|---:|---:|---:|\n'
for m in ('minimal_fixed_F','careflow'):
 x=record['methods'][m];md+=f"| {m} | {x['parameters']} | 4000 | {x['selected_epoch']} | {x['fixed_DEV_batch_MSE_selection_only']:.12f} | {x['elapsed_seconds_including_complete_save_own_selected_replay']:.3f}s | {x['reserved_peak']}B |\n"
md+='\n共同顺序、4000更新、选模规则；容量和实际耗时不同。F在这项DEV选模MSE上较差，未换模型/seed/读出。不把该MSE当五指标或TEST；目标尚未达成。双方7415841507字节整pt已真实D SHA/CRC/唯一通过，B上传未完成，未冒CPU。\n\n原D证据：'+str(p)+'，SHA '+sha(p)+'。\n'
Path('outputs/正式双方fullTRAIN100完成与完整保存实际结果.md').write_text(md,encoding='utf-8')
Path('outputs/正式双方fullTRAIN100完成与完整保存实际结果.json').write_text(json.dumps({'latest_original_record':str(p),'sha256':sha(p),'result':record},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'record':str(p),'sha256':sha(p),'delta_selection_MSE':record['F_minus_C_DEV_selection_MSE'],'elapsed_ratio':record['F_vs_C_elapsed_ratio']}),flush=True)
