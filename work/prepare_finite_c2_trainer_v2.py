from pathlib import Path
s=Path('work/train_finite_task_risk_v1.py').read_text(encoding='utf-8').replace('from finite_task_risk_runtime_v1 import','from finite_task_risk_runtime_v2 import').replace('finite_formal_plan_v1.json','finite_formal_plan_v2.json')
s=s.replace("assert plan['seed']==91817 and plan['epochs']==100 and plan['shared_fixed_epochs']==10","assert plan['seed']==91817 and plan['epochs']==100 and plan['shared_fixed_epochs']==10 and plan['amendment_id']=='C2_ANALYTIC_SINGLE_VALID_TOKEN_READER' and a.mode=='finite_vector'")
s=s.replace("    if epoch==10:save_addon(a.out/'shared_phase_addon.pt',model.addon())","    if epoch==10:\n        save_addon(a.out/'shared_phase_addon.pt',model.addon());assert sha(a.out/'shared_phase_addon.pt')==plan['expected_original_shared_phase_sha256']")
s=s.replace("assert protocol['trainable_parameters']==520506", "protocol.update(amendment_id=plan['amendment_id'],original_failed_root=plan['original_failed_root'],original_control_roots=plan['original_control_roots'],reader_source_sha256=plan['source_sha256']['finite_single_token_reader_v1.py'])\nassert protocol['trainable_parameters']==520506")
s=s.replace("FINITE_100_EPOCH_SELECTED_ADDON_NOT_FULL_CHECKPOINT","FINITE_C2_100_EPOCH_SELECTED_ADDON_NOT_FULL_CHECKPOINT")
Path('work/train_finite_task_risk_v2.py').write_text(s,encoding='utf-8')
s=Path('work/run_finite_training_wrapper_v1.py').read_text(encoding='utf-8').replace('train_finite_task_risk_v1.py','train_finite_task_risk_v2.py');Path('work/run_finite_training_wrapper_v2.py').write_text(s,encoding='utf-8')
