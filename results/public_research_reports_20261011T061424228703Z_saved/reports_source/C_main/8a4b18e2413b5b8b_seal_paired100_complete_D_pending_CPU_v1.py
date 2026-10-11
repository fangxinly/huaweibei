from pathlib import Path
import json,hashlib,zipfile,sys,shutil
d=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_fulltrain100_complete_actual_20261007T030311Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert shutil.disk_usage(d).free>=6*1024**3
files=[d/x for x in ('actual_both_training_completion_progress.json','actual_original_training_cost_and_fixed_selection_only.json','actual_B_transfer_reset_operator_record.json','actual_B_operator_transfer_pending_original.json','operator_CPU_transfer_gate_source_original.py','actual_interactive_closure_and_original_upload_pending.json')]
for m in ('minimal_fixed_F','careflow'):
 files.extend([d/m/'actual_D_GPU_whole_SHA_CRC_source_capture_receipt.json',d/m/'actual_CPU_child_arguments.json',d/m/'a/capture_receipt.json',d/m/'a/capture_actual_exit.json',d/m/'a/snapshot.zip'])
manifest={str(p.relative_to(d)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
manifest_path=d/'actual_local_pending_CPU_seal_members.json';assert not manifest_path.exists();manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
zpath=d/'actual_local_pending_CPU_full_source_receipt_seal.zip'
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(p.relative_to(d)).replace('\\','/'))
 z.write(manifest_path,manifest_path.name)
with zipfile.ZipFile(zpath) as z:
 assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
 for n,r in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==r['sha256']
pending=read(d/'actual_B_operator_transfer_pending_original.json')
result={'status':'ACTUAL_BOTH_TRAIN100_GPU_D_COMPLETE_WHOLE_FILES_AND_LOCAL_OPERATOR_PENDING_SEAL_PASSED_B_CPU_FRESH_PENDING',
 'clock_checked_utc':sys.argv[1],'D':str(d),'local_operator_seal_SHA':sha(zpath),'local_operator_seal_unique_members':len(manifest)+1,'local_operator_seal_CRC_and_all_member_SHA_passed':True,
 'whole_models_D_bytes':7415841507,'whole_models_are_physical_original_downloads_not_SHA_refs':True,'original_B_pending_observation_utc':pending['actual_utc'],
 'original_B_pending_observation_SHA':sha(d/'actual_B_operator_transfer_pending_original.json'),'CPU_roots':{m:pending['methods'][m]['root'] for m in pending['methods']},
 'CPU_transfer_gate_PIDs':{m:pending['methods'][m]['transfer_gate_launch']['transfer_wait_pid'] for m in pending['methods']},
 'original_B_transfer_guard_not_CPU_audit_launch':True,'B_CPU_and_fresh_replay_joint_complete':False,'not_remote_COMPLETE_CPU_capture':True,'new_five_metrics_or_final_TEST':False,'overall_goal_or_lease_preservation_complete':False,
 'session_closure_record_SHA':sha(d/'actual_interactive_closure_and_original_upload_pending.json')}
out=d/'actual_local_pending_CPU_seal_receipt.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
live=Path('outputs/正式双方官方fullTRAIN100实际训练接续.json');v=read(live);v['latest_complete_D_CPU_pending']={'file':str(out),'sha256':sha(out),'result':result};v['status']=result['status'];live.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
continuation={'status':result['status'],'D':str(d),'actual_record':str(out),'sha256':sha(out),'CPU_roots':result['CPU_roots'],'transfer_guard_PIDs':result['CPU_transfer_gate_PIDs'],'closed_ids_no_reuse':[72143,34370,11741,47872,36738,6701,67854,62809],'only_wait_original_upload_and_queued_bye_pending_ids':[83561,29114],'next':'Fresh B actual SSH authentication; read these two original natural_exit/actual_stage_receipt and guard stderr/launch; do not dispatch another CPU child or training. Once natural0, original CPU COMPLETE capture then receipt/D ZIP/joint. Then fresh original A/C whole-selected replay, complete capture/D joint. After both qualified, freeze source and both-parent SHA for fixed selected DEV five report, no201 reread or TEST yet.','source_bundle':'/data/coding/paired_official_fulltrain_v2_20261007T011543Z','plan_SHA':'8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb','CPU_model_forward':False,'all_five_goal_complete':False,'strong_actual_save_windows_UTC':['09:30','11:30','13:00'],'conservative_lease_end_UTC':'2026-10-07T13:30:00+00:00','not_platform_lease_confirmation':True}
cp=Path('outputs/正式双方fullTRAIN100整保存与CPU待接续.json');cp.write_text(json.dumps(continuation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
md='最新接续：先读《正式双方fullTRAIN100整保存与CPU待接续.json》。双方新官方100已原natural0，各4000更新，7415841507字节整selected/resume已D全SHA/CRC/唯一通过。B原03:24:00观察两个等待守卫PID31627/31628 fullargv一致、stderr空，尚无CPU child/自然退出；不是新CPU COMPLETE或fresh模型重放。原文件到齐SHA/CRC并fresh资产/源/UUID/fullargv/compute/空间/至少2h余量门过后，才唯一启动冻结CPU wrapper，不手动重派。\nA/C空闲SSH最后观察reset实际1，B SSH明确exit0，A/C SFTP返回0但带reset警告，不冒全干净bye。关闭ID均禁复用；83561/29114只等待原上传及已排队bye结果，不发新命令，退出后禁复用。\nD新增17成员本地operator seal实际SHA/CRC/唯一/全memberSHA通过；不是remote CPU COMPLETE capture，不回填静态素材成新capture。B原CPU、fresh整模型重放与最终联合保存待完成；五项正式超过、最终TEST、租期09:30/11:30/13:00实际强化保存未完成。\n\n'
for name in ('outputs/研究接续状态.md','outputs/正式双方fullTRAIN100完成与完整保存实际结果.md'):
 p=Path(name);p.write_text(md+p.read_text(encoding='utf-8'),encoding='utf-8')
print(json.dumps({'D_seal':str(out),'SHA':sha(out),'members':len(manifest)+1,'next_file':str(cp)}),flush=True)
