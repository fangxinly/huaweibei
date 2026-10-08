"""Join completed GPU, actual D originals, other-node CPU and atomic captures."""
from pathlib import Path
import datetime,hashlib,json
root=Path('D:/CodexBackups/selective_flow_20261003_1105/scalar_geometry_actual_20261006T060719Z')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
gpu=read(root/'execute/receipt.json');ex=read(root/'execute_exit.json');plan=read(root/'plan.json')
assert gpu['passed'] and ex['exit_code']==0 and gpu['optimizer_steps']==0 and gpu['no_parameter_gradients']
assert gpu['model_state_before']==gpu['model_state_after']
assert gpu['source_sha256']==plan['source_sha256']==sha(root/'diagnose_scalar_direction_geometry_v1.py')
assert gpu['geometry_sha256']==sha(root/'execute/geometry_frozen.npz')
audits=[read(root/'execute/independent_audit.json'),read(root/'local_execute_audit.json'),read(root/'independent_cpu/cpu_array_audit.json')]
assert audits[0]==audits[1]==audits[2]
cpu=read(root/'independent_cpu/cpu_preservation_receipt.json');ce=read(root/'independent_cpu/cpu_exit.json')
assert cpu['status']=='OTHER_NODE_SCALAR_GEOMETRY_ORIGINALS_AND_NUMPY_VERIFIED' and ce['exit_code']==0
assert ce['argv']==ce['actual_proc_argv']
assert cpu['cpu_array_audit_sha256']==sha(root/'independent_cpu/cpu_array_audit.json')
assert cpu['cpu_exit_sha256']==sha(root/'independent_cpu/cpu_exit.json')
assert cpu['original_gpu_receipt_sha256']==sha(root/'execute/receipt.json')
assert cpu['original_packet_sha256']==sha(root/'snapshot.zip')==read(root/'package_receipt.json')['sha256']
assert cpu['wrapper_source_sha256']==sha(Path(__file__).parent/'verify_scalar_geometry_cpu_v1.py')
ca=read(root/'completed_C_capture_audit_v3.json');ba=read(root/'completed_B_CPU_capture_audit_v2.json')
assert ca['status']=='ACTUAL_SCALAR_GEOMETRY_CAPTURE_ORIGINALS_AND_OLD_FROZEN_EVIDENCE_VERIFIED'
assert ba['status']=='ACTUAL_B_SCALAR_GEOMETRY_CPU_CAPTURE_AND_OLD_FROZEN_EVIDENCE_VERIFIED'
assert ca['new_large_array_references_joined_to_actual_D_originals'][0]['sha256']==gpu['geometry_sha256']
assert ba['new_large_array_references_joined_to_actual_originals'][0]['sha256']==gpu['geometry_sha256']
closures=read(root/'session_closures.json')
assert len(closures['sessions'])==4 and all(v['result']['status']=='fulfilled' and v['result']['value']['exit_code']==0 for v in closures['sessions'])
result=dict(status='SCALAR_GEOMETRY_GPU_D_OTHER_NODE_CPU_AND_ACTUAL_CAPTURES_JOINTLY_VERIFIED',actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_natural_exit_utc=gpu['utc'],child_pid=ex['child_pid'],gpu_receipt_sha256=sha(root/'execute/receipt.json'),arrays_sha256=gpu['geometry_sha256'],source_sha256=gpu['source_sha256'],plan_sha256=sha(root/'plan.json'),actual_seconds=gpu['seconds'],actual_peak_allocated_bytes=gpu['actual_peak_allocated_bytes'],science=audits[0],cpu_original_receipt_sha256=sha(root/'independent_cpu/cpu_preservation_receipt.json'),cpu_actual_utc=cpu['actual_utc'],cpu_node_uuid=cpu['gpu_uuid'],cpu_no_model_forward=True,actual_C_capture=ca,actual_B_CPU_capture=ba,all_four_sessions_explicit_exit0=True,scope='Same-point first-step label-free GPU geometry, not new control performance, message-path lambda=0 replay, student main/aux training gate, CAL fit or EVAL metric.',old_fullweights_reuploaded_or_redownloaded=False)
(root/'joint_preservation_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
outputs=Path(__file__).parent.parent/'outputs'
(outputs/'首步标量方向几何GPU与D_CPU快照联合核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
failure=dict(status='REJECTED_AUDIT_VERSION_PRESERVED_NOT_SCIENTIFIC_FAILURE',phase='execute_actual_capture',auditor='audit_scalar_geometry_capture_v2.py',actual_tool_exit_code=1,reason='KeyError: execute/geometry_frozen.npz absent from ZIP small members; actual capture19 recorded it as stable opened inode large SHA reference.',resolution='New C v3/B v2 auditors join exact fresh large NPZ path/byte count/SHA to actual separately downloaded D original and original other-node CPU packet. All other assertions remain. No frozen scientific source or capture19 changed.',array_bytes=32258674,array_sha256=gpu['geometry_sha256'],reference_is_full_payload_download=False)
(outputs/'首步几何完成捕获大数组引用原审核拒绝与修订.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','actual_utc','child_pid','actual_seconds','arrays_sha256','all_four_sessions_explicit_exit0']}))
