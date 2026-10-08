from pathlib import Path
import json
p=Path(__file__).resolve().parent.parent/'outputs/逐样本效用教师校准分析.json';d=json.loads(p.read_text(encoding='utf-8'))
for row in d['rows']:
    print(json.dumps({'mode':row['mode'],'own':row['own_teachers'],'main_retention':row['default_retention_benefit_vs_all_off'],'ordering':row['donor_ordering'],'directions':[{ 'direction':v['direction'],'pair_MSE':v['pair_teacher']['MSE'],**{k:v['calibration'][k] for k in ['q_mean','positive_fraction','sign_accuracy_nonzero_q','balanced_sign_recall','spearman_u_q','smooth_l1','zero_smooth_l1']}} for v in row['directions']]}))
