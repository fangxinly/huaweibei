"""Supplementary saved-original order/role audit; no labels/model or score."""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0,str(Path(__file__).parent/'paired_official_fulltrain_v2_20261007T011543Z'))
from paired_fulltrain_evidence_candidate_v1 import read,sha,write,require,now,plan_gate
base=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_official_prechecks_v2_actual_20261007T012116Z')
bundle=Path(__file__).parent/'paired_official_fulltrain_v2_20261007T011543Z'
plan_sha='8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb'
plan=plan_gate(bundle,plan_sha)
order=np.load(bundle/plan['orders_file'],allow_pickle=False)
require(sha(bundle/plan['orders_file'])==plan['orders_sha256'] and order.shape==(100,1281),'Source physical100x1281 common order')
require(all(np.array_equal(np.sort(row),np.arange(1281)) for row in order),'All physical epoch permutations')
result={'status':'ACTUAL_SAVED_V2_PRECHECK_PHYSICAL_COMMON_ORDERS_AND_ACTUAL_TWO_BATCH_ROWS_PASSED',
        'actual_utc':now(),'plan_sha256':plan_sha,'orders_sha256':plan['orders_sha256'],
        'method_records':{},'new_label_or_model_access':False,'new_metric_or_model_fit':False}
for method in plan['methods']:
    a=base/method/'a'; r=read(a/'run/out/actual_stage_receipt.json'); m=read(a/'member_manifest.json')
    physical=a/'run/out/original_shared_TRAIN_orders.npy'
    require(sha(physical)==plan['orders_sha256']==m['small_members']['run/out/original_shared_TRAIN_orders.npy']['sha256'],'Exact original GPU captured orders/source equality')
    require(np.array_equal(np.load(physical,allow_pickle=False),order),'Whole original common100 epoch orders equality')
    require(r['updates'][0]['rows']==order[0,:32].tolist() and r['updates'][1]['rows']==plan['precheck_mixed_TRAIN_rows'],'Actual two batch row proof')
    require(all(0<=i<1281 for u in r['updates'] for i in u['rows']),'Only authorized TRAIN indices')
    require(r['DEV_true_labels_read'] is False and not any(x['labels_read'] and x['role']!='train' for x in r['guard_journal']),'Original precheck role journal')
    result['method_records'][method]={'whole_physical_order_sha256':sha(physical),'original_stage_receipt_sha256':sha(a/'run/out/actual_stage_receipt.json'),
        'actual_gradient_rows_match_frozen_source':True,'DEV_true_labels_read':False}
dest=base/'actual_precheck_supplementary_physical_order_audit.json'
require(not dest.exists(),'Preserve prior supplementary order evidence')
write(dest,result)
print(result['status'],sha(dest),flush=True)
