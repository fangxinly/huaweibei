import ast,json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'work/minimal_fixed_precheck_local_20261006T1240Z'
OUT=ROOT/'work/second_lease_fixed_precheck_v3_20261006T1345Z';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in PARENT.iterdir():
 if p.is_file() and (p.suffix in ('.py','.npy') or p.name in ('runtime_candidate_plan.json','train_row_video_mapping.json')):shutil.copyfile(p,OUT/p.name)
shutil.copyfile(ROOT/'work/donor_terminal_mechanism_precheck_v1.py',OUT/'donor_terminal_mechanism_precheck_v1.py')
gates=(PARENT/'precheck_gates_v1.py').read_text(encoding='utf-8')
gates=gates.replace('GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7','GPU-53696803-875e-eec8-2231-29db63579891').replace('GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f').replace('GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa','GPU-417d3577-0525-788b-7296-0808a0f52012')
old="    if not e.get('platform_lease_confirmed', False):\n        raise PermissionError('ESTIMATED_LEASE_IS_NOT_CONFIRMED')"
assert old in gates
gates=gates.replace(old,"    if e.get('lease_source') != 'DIRECT_HUMAN_NEW_P4_24H_20261006' or not e.get('human_new_lease_assertion_verified', False):\n        raise PermissionError('NEW_HUMAN_LEASE_SOURCE_NOT_VERIFIED')\n    # Human supplied 24h directly in this chat; estimated absolute end is not\n    # a platform confirmation. Source provenance must also be checked by caller.")
(OUT/'precheck_gates_v2.py').write_text(gates,encoding='utf-8')
runner=(PARENT/'precheck_minimal_fixed_v2.py').read_text(encoding='utf-8').replace('from precheck_gates_v1 import','from precheck_gates_v2 import')
old="    if actual_uuid != target_uuid.removeprefix('GPU-'):\n        raise ValueError('ACTUAL_TORCH_DEVICE_UUID_MUST_MATCH_FRESH_QUERY')"
assert old in runner
runner=runner.replace(old,"    import subprocess\n    queried_uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()\n    if len(queried_uuids)!=1 or queried_uuids[0].strip()!=target_uuid or torch.cuda.device_count()!=1:\n        raise ValueError('CURRENT_SINGLE_DRIVER_GPU_UUID_MISMATCH')\n    if actual_uuid and actual_uuid != target_uuid.removeprefix('GPU-'):\n        raise ValueError('ACTUAL_TORCH_DEVICE_UUID_MISMATCH')\n    uuid_method='torch_properties_and_single_driver' if actual_uuid else 'single_driver_query_Torch2_1_property_unavailable'")
runner=runner.replace('        loss=fit_objective(session.model,batch)',"        from minimal_fixed_flow_v2 import objective\n        prediction=forward_batch(session.model,batch)\n        loss=objective(prediction,batch[3],session.model.dberta.last_losses)",1)
runner=runner.replace("primary=(session.model.dberta.last_first_prediction.view(-1)-batch[3].view(-1)).square().mean()","primary=(prediction.view(-1)-batch[3].view(-1)).square().mean()",1)
runner=runner.replace('        del batch,loss,report,terms,names','        del batch,loss,report,terms,names,prediction',1)
anchor="    roles=plan['precheck_row_candidates']; gradients=[]; component_gradients=[]; times=[]"
assert anchor in runner
runner=runner.replace(anchor,anchor+"\n    from donor_terminal_mechanism_precheck_v1 import donor_terminal_check\n    mechanism=[]\n    small=row_batch(session,roles['normal'][:4])\n    result,arrays=donor_terminal_check(session,small,'clean_initial')\n    np.savez(output/'donor_mechanism_clean_initial.npz',**arrays);mechanism.append(result)\n    del small,result,arrays\n    budget('initial_donor_mechanism')")
anchor='    # Tail has all 23 FIT rows: forward/backward only; no third step.'
runner=runner.replace(anchor,"    small=row_batch(session,roles['normal'][:4])\n    result,arrays=donor_terminal_check(session,small,'after_two_steps')\n    np.savez(output/'donor_mechanism_after_two_steps.npz',**arrays);mechanism.append(result)\n    del small,result,arrays\n    if sum(x['seconds'] for x in mechanism)>120:raise RuntimeError('COMBINED_MECHANISM_TWO_MINUTE_BUDGET')\n    budget('post_step_donor_mechanism')\n"+anchor)
runner=runner.replace("'gpu_uuid':target_uuid,'permission_evidence':permission", "'gpu_uuid':target_uuid,'gpu_uuid_check_method':uuid_method,'donor_mechanism':mechanism,'permission_evidence':permission")
runner=runner.replace("'objective':float(loss.detach()),'optimizer_steps_after':index+1,", "'objective':float(loss.detach()),'optimizer_steps_after':index+1,'optimizer_lr_after':[g['lr'] for g in session.optimizer.param_groups],")
(OUT/'precheck_minimal_fixed_v3.py').write_text(runner,encoding='utf-8')
for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
runtime=json.loads((OUT/'runtime_candidate_plan.json').read_text(encoding='utf-8'))
for relative,digest in {**runtime['source_sha256'],**runtime['role_file_sha256']}.items():assert sha(OUT/relative)==digest
plan={'status':'SECOND_HUMAN_LEASE_NEW_PRECHECK_SOURCE_V3_NOT_ACTUAL_GPU','frozen_actual_utc':datetime.now(timezone.utc).isoformat(),
 'runtime_plan_sha256':sha(OUT/'runtime_candidate_plan.json'),'parent_precheck_plan_sha256':sha(PARENT/'precheck_candidate_plan_v2.json'),
 'runner':'precheck_minimal_fixed_v3.py','source_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},
 'changes':'New human endpoints/UUID/24h provenance, honest Torch2.1 absent uuid property uses fresh single driver plus CUDA device_count; primary probe uses actual objective prediction (one pass same as old first_prediction); label-free fixed donor zeroing only in second context four normal FIT inputs clean/post states, restored exact replay; no objective/update/init changes.',
 'scope':'fold0 public full fixed reference two FIT steps tail23 no step INNER153 dummy-label own full disk replay; no formal100/OUTER/performance metrics',
 'mechanism':{'rows':runtime['precheck_row_candidates']['normal'][:4],'eval':True,'dummy_labels':0,'same_checkpoint_first_state':True,'zero_only_donor_term':True,'calls_each_state':3,'combined_seconds_max':120,'no_threshold_for_small_nonzero_effect':True,'no_extra_training_to_force_nonzero':True},
 'budget':{'max_precheck_seconds':1200,'max_Torch_peak_bytes':6*1024**3,'remote_min_free_bytes':4*1024**3,'D_min_free_bytes':6*1024**3,'save_reserve_seconds':7200},
 'formal_training_allowed':False,'actual_GPU_executed':False,'natural_wrapper_and_independent_CPU_auditor_pending':True}
(OUT/'second_lease_precheck_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'directory':str(OUT),'plan_sha256':sha(OUT/'second_lease_precheck_plan.json'),'runner_sha256':sha(OUT/plan['runner']),'AST_and_parent_pins':True,'actual_GPU':False}))
