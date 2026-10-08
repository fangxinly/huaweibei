from pathlib import Path
w=Path(__file__).parent
s=(w/'audit_finite_snapshot_full_extras_v4.py').read_text(encoding='utf-8').replace('finite_c2_snapshots_20261005T1612Z','finite_c2_completed_snapshots_20261005T1627Z').replace('capture_soft_vector_v12.py','capture_soft_vector_v13.py').replace('有限任务风险C2启动前旧保存联合快照核验.json','有限任务风险C2完成后旧保存联合快照核验.json')
(w/'audit_finite_snapshot_full_extras_v5.py').write_text(s,encoding='utf-8')
s=(w/'audit_finite_c2_diagnostics_snapshot_v1.py').read_text(encoding='utf-8').replace('finite_c2_snapshots_20261005T1612Z','finite_c2_completed_snapshots_20261005T1627Z').replace('capture_soft_vector_v12.py','capture_soft_vector_v13.py')
s=s.replace("if node in ['a','b']:","if node in ['a','b','c']:")
s=s.replace("pre='finite_diagnostics_v3/';", "pre='finite_c2/diagnostics_v4/' if node=='c' else 'finite_diagnostics_v3/';")
s=s.replace("assert rr['source_sha256']==sha(z.read('finite_formal_source/diagnose_finite_task_risk_v3.py'))==sha((w/'diagnose_finite_task_risk_v3.py').read_bytes())", "formal='finite_c2/' if node=='c' else 'finite_formal_source/';version='v4' if node=='c' else 'v3';run='finite_c2/run/' if node=='c' else 'finite_run/'\n   assert rr['source_sha256']==sha(z.read(formal+'diagnose_finite_task_risk_'+version+'.py'))==sha((w/('diagnose_finite_task_risk_'+version+'.py')).read_bytes())")
s=s.replace("assert rr['plan_sha256']==sha(z.read('finite_formal_source/finite_diagnostics_plan_v3.json'))==sha((w/'finite_diagnostics_plan_v3.json').read_bytes())", "assert rr['plan_sha256']==sha(z.read(formal+'finite_diagnostics_plan_'+version+'.json'))==sha((w/('finite_diagnostics_plan_'+version+'.json')).read_bytes())")
s=s.replace("get('finite_run/selection.json')","get(run+'selection.json')").replace("m['finite_run/best_addon.pt']","m[run+'best_addon.pt']").replace("z.read('finite_run/predictions.npz')","z.read(run+'predictions.npz')")
s=s.replace("  else:\n   inv=get('finite_c2_inventory.json');", "  if node=='c':\n   inv=get('finite_c2_inventory.json');")
old="   assert inv['actual_process_argv']==inv['launch']['argv'] and inv['actual_process_argv'][1:]==pr['argv'] and 'State:\\tZ' not in inv['actual_process_status'] and pr['gpu_uuid'] in inv['compute'] and 'exit' not in inv\n   c2=dict(state='LIVE_C2_NOT_100_COMPLETE',epochs=len(h),last_epoch=h[-1],capture_inventory=inv,plan_sha256=pr['formal_plan_sha256'])"
new="""   sel=get('finite_c2/run/selection.json');assert len(h)==sel['epochs']==100 and sel['best_epoch']==np.argmin([x['dev_author_batch_mean_mse'] for x in h])+1
   assert inv['exit']['exit_code']==0 and inv['actual_process_argv'] in [None,[]]
   assert sel['effective_mode']==('fixed' if sel['best_epoch']<=10 else 'finite_vector') and sel['frozen_tensor_sha_before']==sel['frozen_tensor_sha_after'] and sel['selected_prediction_replay_max_error']==0
   assert m['finite_c2/run/best_addon.pt']['sha256']==sel['addon_sha256'] and m['finite_c2/run/predictions.npz']['sha256']==sel['prediction_sha256']
   c2=dict(state='C2_ACTUAL_100_EXIT0_COMPLETE',epochs=len(h),selection=sel,last_epoch=h[-1],capture_inventory=inv,plan_sha256=pr['formal_plan_sha256'])"""
assert old in s;s=s.replace(old,new)
s=s.replace('FINITE_AB_FROZEN20_ARRAYS_AND_C2_FRESH_LIVE_SNAPSHOT_INDEPENDENTLY_VERIFIED','FINITE_AB_AND_C2_FROZEN20_ARRAYS_AND_C2_ACTUAL100_SNAPSHOT_INDEPENDENTLY_VERIFIED').replace('有限任务风险两臂20条件诊断及C2启动快照独立核验.json','有限任务风险两对照及C2三组20条件诊断与100完成独立核验.json')
(w/'audit_finite_c2_diagnostics_completed_snapshot_v2.py').write_text(s,encoding='utf-8')
