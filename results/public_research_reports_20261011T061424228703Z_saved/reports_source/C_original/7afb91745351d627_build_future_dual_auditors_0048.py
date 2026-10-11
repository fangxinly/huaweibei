from pathlib import Path
import ast,json
F=Path('outputs/parallel_gpu_20261003_1800');s=(F/'audit_source_train_shuffle_local_v1_20261004T0018Z.py').read_text(encoding='utf-8')
for case,name,loss in [('full_null','v8_matched_full_route_train_weak_source_permutation_control_v1','full source route only: (final MSE + 0.1 first-phase MSE)/1.1; seven routes still computed and evaluated'),('text_one','v8_matched_text_only_route_supervision_v1','text source route code 1 only: (final MSE + 0.1 second-Euler-step MSE)/1.1; seven routes still computed and evaluated; natural full-route dev selection')]:
 t=s.replace('v8_matched_seven_route_train_weak_source_permutation_control_v1',name).replace('source_train_shuffle_deployment_20261004T0016Z','dual_source_deployment_20261004T0044Z/'+case).replace('equal average over seven source availability routes of (final MSE + 0.1 first-phase MSE)/1.1',loss).replace('source_train_shuffle_seed128_4060Ti','dual_'+case+'_seed128_4060Ti')
 if case=='text_one':
  t=t.replace("assert p['training_source_permutation']['modalities']==['audio','visual']","assert 'training_source_permutation' not in p")
  t=t.replace("coverage=[x['training_mechanism']['training_source_permutation_moved_fraction'] for x in h];assert all(0<=c<=1 for c in coverage) and max(coverage)>0","assert all('training_source_permutation_moved_fraction' not in x['training_mechanism'] for x in h)")
  t=t.replace('Single seed train-only source relationship null control; compare original aligned seven-source pilot, no five-seed promotion.','Single seed code1 text-only-loss control with same seven-route compute; full-route dev selection unchanged. No five-seed promotion.')
 t=t.replace("assert r['test_evaluated'] is False", "assert p['parallel_authorization']['shared_gpu_pair_authorized'] is True and p['parallel_authorization']['maximum_training_processes']==2\nassert r['test_evaluated'] is False")
 ast.parse(t);(F/('audit_dual_'+case+'_local_v1_20261004T0048Z.py')).write_text(t,encoding='utf-8',newline='\n')
print('Two new-name completion auditors AST passed; not applied to future data.')

