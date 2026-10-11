from pathlib import Path
import datetime,hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'work/inflow_utility_v4_20261005T0346Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assign={'a':('none','GPU-5902bbd4-2328-0822-777c-1311d53ee5a3'),'b':('fixed','GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d'),'c':('predicted','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa')}
ref=None;orderref=None;rows=[]
for node,(mode,uuid) in assign.items():
 folder=ROOT/'work/utility_checks'/node
 p=json.loads((folder/'protocol.json').read_text());c=json.loads((folder/'checks.json').read_text());launch=json.loads((folder/'check_launch.json').read_text())
 assert p['name']=='inflow_utility_v4' and p['seed']==91813 and p['epochs']==100 and p['total_updates']==4000 and p['updates_per_epoch']==40
 assert p['mode']==c['mode']==launch['mode']==mode and p['gpu_uuid']==launch['gpu_uuid']==uuid
 assert p['train_samples']==1281 and p['valid_samples']==229 and p['pretrained_tensors_verified']==198
 assert p['config']['pair_weight']==.025 and p['config']['utility_weight']==.01 and p['config']['utility_target_scale']==.5 and p['config']['utility_gate_sharpness']==4
 # Released train_reflow_new.py:217 sets this environment value in set_random_seed.
 # Launcher starts all interpreters with0; protocol records the later value91813.
 assert p['parent_pythonhashseed']==str(p['seed'])
 assert launch['launcher_sha256']==sha(ROOT/'work/launch_utility_v4.py') and launch['source_manifest_sha256']==sha(SOURCE/'source_manifest.json')
 assert all(sha(SOURCE/name)==digest for name,digest in p['own_source_sha256'].items())
 shared={k:v for k,v in p.items() if k not in ('mode','gpu_uuid')}
 if ref is None:ref=shared
 else:assert shared==ref
 assert all(c[k] is True for k in ('initial_zero_feedback_equality_verified','label_isolation_verified','utility_gradient_detach_verified','optimizer_change_verified','batch_composition_invariance_verified','padding_invariance_verified'))
 assert c['warmup_check_steps']==3
 assert {'text','audio','visual','pair_head','utility_head','queries','velocity','backward_velocity','role_head'}<=set(c['gradient_groups'])
 if mode!='none':assert c['feedback_output_gradient_first_backward_verified'] and c['learned_feedback_gradient_sum']>0 and 'feedback_output' in c['gradient_groups']
 else:assert c['feedback_output_gradient_first_backward_verified'] is False and c['learned_feedback_gradient_sum'] is None
 log=(folder/'check.log').read_text();assert 'INFLOW_UTILITY_CHECK_COMPLETE' in log and 'Traceback' not in log
 orders=np.load(folder/'batch_orders.npy',allow_pickle=False);assert orders.shape==(100,1280) and orders.dtype==np.dtype('<i8')
 assert hashlib.sha256(orders.tobytes()).hexdigest()==p['batch_order_sha256']
 for i,o in enumerate(orders):assert len(np.unique(o))==1280 and o.min()>=0 and o.max()<1281 and hashlib.sha256(o.tobytes()).hexdigest()==p['batch_order_epoch_sha256'][i]
 if orderref is None:orderref=orders
 else:assert np.array_equal(orders,orderref)
 rows.append({'node':node,'mode':mode,'gpu_uuid':uuid,'initial_model_sha256':p['initial_model_sha256'],'initial_flow_sha256':p['initial_flow_sha256'],'batch_order_sha256':p['batch_order_sha256'],'protocol_sha256':sha(folder/'protocol.json'),'checks_sha256':sha(folder/'checks.json'),'checks':c})
r={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'UTILITY_THREE_GPU_CHECKS_AND_EXACT_INITIAL_SOURCE_PROTOCOL_100ORDERS_VERIFIED','rows':rows,'limits':'Preflight only, three updates; no 100-epoch result or utility calibration claim. Raw log key last_train_batch_utility_weights is populated after eval and therefore refers to the last DEV batch.'}
out=ROOT/'outputs/逐样本效用启动前核验.json'
with out.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
print(json.dumps({'status':r['status'],'initial_model_sha256':ref['initial_model_sha256'],'initial_flow_sha256':ref['initial_flow_sha256'],'batch_order_sha256':ref['batch_order_sha256'],'rows':[{'mode':q['mode'],'feedback_hidden_gradient':q['checks']['learned_feedback_gradient_sum']} for q in rows]},indent=2))
