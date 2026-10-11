from pathlib import Path
import datetime,hashlib,json,numpy as np
ROOT=Path('work/gated_checks');SOURCE=Path('work/inflow_gated_v3_20261005T0241Z')
rows=[];reference=None;ordersref=None
uuid={'a':'GPU-5902bbd4-2328-0822-777c-1311d53ee5a3','b':'GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
for node,mode in (('a','none'),('b','state'),('c','task')):
    folder=ROOT/node;p=json.loads((folder/'protocol.json').read_text());checks=json.loads((folder/'checks.json').read_text())
    assert p['name']=='inflow_conditions_v3' and p['mode']==mode and p['gpu_uuid']==uuid[node]
    assert p['seed']==91812 and p['epochs']==100 and p['total_updates']==4000 and p['updates_per_epoch']==40
    assert p['train_samples']==1281 and p['valid_samples']==229 and p['parent_pythonhashseed']=='91812'
    for name,digest in p['own_source_sha256'].items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest
    assert checks['pretrained_tensors_verified']==198 and checks['warmup_check_steps']==3
    assert all(checks[k] is True for k in ('optimizer_change_verified','initial_zero_gate_equality_verified','batch_composition_invariance_verified','padding_invariance_verified'))
    if mode!='none':assert checks['learned_feedback_gradient_sum']>0
    else:assert checks['feedback_gates_after_check']==[0,0,0]
    log=(folder/'check.log').read_text(encoding='utf-8')
    assert 'INFLOW_CONDITIONS_CHECK_COMPLETE' in log and 'Traceback' not in log
    orders=np.load(folder/'batch_orders.npy',allow_pickle=False)
    assert orders.shape==(100,1280) and orders.dtype==np.dtype('<i8')
    assert hashlib.sha256(orders.tobytes()).hexdigest()==p['batch_order_sha256']
    for i,order in enumerate(orders):
        assert len(np.unique(order))==1280 and order.min()>=0 and order.max()<1281
        assert hashlib.sha256(order.tobytes()).hexdigest()==p['batch_order_epoch_sha256'][i]
    shared={k:v for k,v in p.items() if k not in ('mode','gpu_uuid')}
    if reference is None:reference=shared;ordersref=orders
    else:assert shared==reference and np.array_equal(ordersref,orders)
    rows.append({'node':node,'mode':mode,'gpu_uuid':uuid[node],'checks':checks,
        'initial_model_sha256':p['initial_model_sha256'],'initial_flow_sha256':p['initial_flow_sha256'],
        'batch_order_sha256':p['batch_order_sha256']})
r={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'THREE_GATED_GPU_CHECKS_MATCHED_SOURCE_INITIAL_STATE_ALL_100_ORDERS_VERIFIED','rows':rows,
    'initial_model_sha256':reference['initial_model_sha256'],'initial_flow_sha256':reference['initial_flow_sha256'],'batch_order_sha256':reference['batch_order_sha256'],
    'source_sha256':reference['own_source_sha256'],'limits':'Three warmup checks are not 100-epoch completed performance. Gate gradient and later projection gradient are small; effectiveness remains to be measured.'}
(Path('outputs')/'门控条件流启动前核验.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r,indent=2))
