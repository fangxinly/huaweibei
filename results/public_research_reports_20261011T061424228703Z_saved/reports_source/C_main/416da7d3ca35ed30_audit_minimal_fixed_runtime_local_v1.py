"""No Torch, raw dataset, labels, scores, training, or remote execution."""
import ast, hashlib, importlib.util, json, tempfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
BUNDLE = ROOT/'work/minimal_fixed_runtime_v1_local_20261006T1228Z'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
plan = json.loads((BUNDLE/'runtime_candidate_plan.json').read_text(encoding='utf-8'))
for n,h in {**plan['source_sha256'],**plan['role_file_sha256']}.items(): assert sha(BUNDLE/n)==h
for p in BUNDLE.glob('*.py'):
    compile(p.read_text(encoding='utf-8'), str(p), 'exec')
tree = ast.parse((BUNDLE/'minimal_fixed_runtime_v1.py').read_text(encoding='utf-8'))
calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)]
assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ('step','load','load_state_dict') and isinstance(n.func.value,ast.Name) and n.func.value.id in ('optimizer','scheduler','torch') for n in calls)
assert not any(isinstance(n,ast.Attribute) and n.attr=='decoder_modules' for n in ast.walk(tree))
assert all(n.value!='test' and n.value!='dev' for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str))
orders=np.load(BUNDLE/'fit_orders_seed91819_100.npy',allow_pickle=False)
fit=np.load(BUNDLE/'fit_0.npy',allow_pickle=False)
assert orders.shape==(100,695) and orders.dtype==np.int64
assert all(np.array_equal(np.sort(row),np.sort(fit)) for row in orders)
assert len({row.tobytes() for row in orders})==100
assert len(orders[0,-23:])==23
assert all(len([row[i:i+32] for i in range(0,695,32)])==22 for row in orders)
ids={r:np.load(BUNDLE/(r+'_0.npy'),allow_pickle=False) for r in ('fit','inner','outer')}
mapping=json.loads((BUNDLE/'train_row_video_mapping.json').read_text(encoding='utf-8'))
assert [r['row'] for r in mapping]==list(range(1281))
spec=importlib.util.spec_from_file_location('local_guard_only',BUNDLE/'role_guard_v1.py')
guardmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(guardmod)
class PoisonableRecord:
    reads=[];allowed=set(ids['fit'].tolist())
    def __init__(self,i):self.i=i
    def __getitem__(self,k):
        if k==1:
            if self.i not in self.allowed:raise AssertionError('POISON_LABEL_READ_OUTSIDE_ALLOWED_ROLE')
            self.reads.append(self.i);return np.asarray([[float(self.i)]])
        return {'synthetic_row':self.i} if k==0 else mapping[self.i]['segment_id']
records=[PoisonableRecord(i) for i in range(1281)]
guard=guardmod.FoldZeroGuard(records,ids,[r['video_id'] for r in mapping])
for r in ('fit','inner'):
    inputs=guard.inputs_only(ids[r],r)
    assert len(inputs)==len(ids[r]) and all(float(x[1].sum())==0 for x in inputs)
assert not PoisonableRecord.reads
supervised=guard.fit_supervision(ids['fit'])
assert all(float(x[1].reshape(-1)[0])==float(i) for x,i in zip(supervised,ids['fit']))
negative=[]
def rejects(name,f):
    try:f()
    except (PermissionError,FileNotFoundError):negative.append(name)
    else:raise AssertionError('EXPECTED_GATE_DID_NOT_REJECT: '+name)
rejects('outer_inputs',lambda:guard.inputs_only(ids['outer'],'outer'))
rejects('inner_as_fit_supervision',lambda:guard.fit_supervision(ids['inner']))
rejects('float_row_coercion',lambda:guard.inputs_only(np.array([1.5]),'fit'))
rejects('duplicate_rows',lambda:guard.inputs_only(np.array([int(fit[0])]*2),'fit'))
rejects('negative_rows',lambda:guard.inputs_only(np.array([-1]),'fit'))
with tempfile.TemporaryDirectory(prefix='fixed_role_guard_only_') as tmp:
    path=Path(tmp)/'synthetic_predictions.npz'
    rejects('unfrozen_inner_predictions',lambda:guard.inner_labels_after_frozen_predictions(path,'0'*64))
    np.savez(path,row_ids=ids['inner'],prediction=np.zeros(153),model_state_sha256=np.asarray('0'*64))
    rejects('incorrect_prediction_sha',lambda:guard.inner_labels_after_frozen_predictions(path,'1'*64))
    PoisonableRecord.allowed=set(ids['inner'].tolist())
    before=len(PoisonableRecord.reads)
    values=guard.inner_labels_after_frozen_predictions(path,sha(path))
    assert np.array_equal(values,ids['inner'].astype(float))
    assert PoisonableRecord.reads[before:]==ids['inner'].tolist()
    np.savez(path,row_ids=ids['outer'][:153],prediction=np.zeros(153),model_state_sha256=np.asarray('0'*64))
    rejects('wrong_role_prediction_row_order',lambda:guard.inner_labels_after_frozen_predictions(path,sha(path)))
receipt={'status':'LOCAL_AST_SYNTHETIC_POISON_ROLE_GUARD_AND_LABEL_FREE_100_ORDERS_PASSED',
 'actual_utc':datetime.now(timezone.utc).isoformat(), 'plan_sha256':sha(BUNDLE/'runtime_candidate_plan.json'),
 'rejected_requests':negative, 'orders':[100,695], 'updates_per_epoch':22,'tail':23,
 'synthetic_label_scope_only':True,'actual_raw_data_or_real_labels_read':False,
 'actual_constructor_or_optimizer_imported':False,'Torch_autograd_or_GPU_verified':False,
 'strict_checkpoint_replay_verified':False,'new_scores':False,'formal100_allowed':False}
(BUNDLE/'local_role_orders_static_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
