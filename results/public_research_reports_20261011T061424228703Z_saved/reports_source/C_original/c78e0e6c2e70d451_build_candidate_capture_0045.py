import datetime,json,hashlib,ast
from pathlib import Path
F=Path('outputs/parallel_gpu_20261003_1800');d=Path('D:/CodexBackups/selective_flow_20261003_1105/completed_refresh_20261004T0037Z')
reports={p.name:json.loads((p/'local_checkpoint_verification.json').read_text()) for p in d.iterdir()}
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
decision={'created_at':now,'single_seed_dev_only':True,'test_evaluated':False,'five_seed_promoted':False,'observations':{}}
for n,r in reports.items():
 baseline=(.6599084734916687,.5856125354766846) if 'shuffle' in n else (.7274219393730164,.6033974289894104)
 decision['observations'][n]={'selection':r['selection'],'MAE':r['metrics_recomputed']['MAE'],'baseline_MSE':baseline[0],'baseline_MAE':baseline[1],'MSE_difference':r['selection']['valid_mse']-baseline[0],'MAE_difference':r['metrics_recomputed']['MAE']-baseline[1]}
decision['decision']='Both newly completed P4 coupled guarded runs worsen the same-node rho0 dev scores; wait matching unguarded. Seven-loss train AV permutation is worse than aligned seven-loss; this single-seed encoded-source observation does not prove raw-source causality. Run the missing full-loss permutation control and matched pure-text-loss control under explicit user-authorized concurrency; retain no five-seed promotion.'
with (F/'new_completed_candidate_decision_20261004T0045Z.json').open('x') as f:json.dump(decision,f,indent=2)
with (F/'completion_auditor_entry_key_failure_20261004T0045Z.json').open('x') as f:json.dump({'error':'V7 auditor initially requested completed guarded; actual supervisor keys guarded_run. Both first calls KeyError before any output, retry correct guarded_run independently passed. No training or source changes.','corrected':True,'original_source_retained':True},f,indent=2)
s=(F/'capture_live_supervision_snapshot_v7_20261004T0018Z.py').read_text(encoding='utf-8')
needle="B/'own_flow_v8_identifiability/source_train_shuffle_frozen_views_20261004T0018Z']"
assert needle in s
s=s.replace(needle,"B/'own_flow_v8_identifiability/source_train_shuffle_frozen_views_20261004T0018Z',"+",".join("B/'own_flow_v8_identifiability/dual_source_20261004T0044Z/"+c+"'" for c in ['full_null','text_one','full_null_diagnostic','text_one_diagnostic'])+"]")
ast.parse(s);(F/'capture_live_supervision_snapshot_v8_20261004T0045Z.py').write_text(s,encoding='utf-8',newline='\n')

