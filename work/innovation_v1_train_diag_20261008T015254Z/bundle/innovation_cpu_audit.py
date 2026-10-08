"""Independent native CPU replay of saved small heads; no encoder forward/labels."""
import argparse,datetime,hashlib,io,os,sys,zipfile
from pathlib import Path
import numpy as np
import torch
from contract import read,write,sha,video
from pipeline import Mechanism,state_sha

def checkzip(path,expected):
    assert sha(path)==expected
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        manifest=__import__('json').loads(z.read('member_SHA.json'))
        assert set(z.namelist())==set(manifest)|{'member_SHA.json'}
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    return manifest

def model_hash(values):
    h=hashlib.sha256()
    for name,v in sorted(values.items()):
        x=v.detach().cpu().contiguous();h.update(name.encode());h.update(str((tuple(x.shape),str(x.dtype))).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()

def run(a):
    assert sha(a.plan)==a.plan_sha;p=read(a.plan);torch.set_num_threads(2)
    for name,h in p['source_sha256'].items():assert sha(a.plan.parent/name)==h
    checkzip(a.train,p['train_archive_SHA']);checkzip(a.cache,p['cache_archive_SHA'])
    with zipfile.ZipFile(a.train) as z:
        natural=__import__('json').loads(z.read('natural_exit.json'));assert natural['natural_exit']==0 and natural['plan_SHA']==p['parent_execution_plan_SHA']
        roles=__import__('json').loads(z.read('out/actual_role_identity.json'));receipt=__import__('json').loads(z.read('out/actual_train_receipt.json'))
        ck=z.read('out/complete_small_training_state.pt');assert hashlib.sha256(ck).hexdigest()==receipt['checkpoint_SHA']
        saved=torch.load(io.BytesIO(ck),map_location='cpu')
        q=np.load(io.BytesIO(z.read('out/frozen_INNER_predictions.npz')),allow_pickle=False);ids=q['row_ids'].astype(str);pred={k:q[k].copy() for k in p['methods']}
    with zipfile.ZipFile(a.cache) as z:
        f=np.load(io.BytesIO(z.read('out/public_frozen_features.npz')),allow_pickle=False);allids=f['row_ids'].astype(str);pos={s:i for i,s in enumerate(allids)};idx=np.array([pos[s] for s in ids]);features=[f[k][idx].copy() for k in ('text','audio','vision')]
    assert ids.tolist()==roles['inner_ids'] and saved['config']==p['config']
    calv={video(r) for r in roles['calibration_ids']};innerv={video(r) for r in ids};trv={video(r) for r in roles['task_train_ids']}
    assert not calv&trv and not innerv&(calv|trv)
    budgets={'baseline':p['config']['baseline_steps'],'flow':p['config']['flow_steps'],'regression':p['config']['flow_steps']}
    checked=[]
    for rec in saved['records']:
        s=rec['complete'];steps=budgets.get(rec['kind'],p['config']['proposal_steps'] if rec['kind'].startswith('proposal_') else p['config']['gate_steps'])
        assert s['steps']==steps and len(s['history'])==steps and np.isfinite(s['history']).all()
        assert model_hash(s['model'])==s['final_SHA']
        assert all(torch.isfinite(v).all() for v in s['model'].values())
        assert all(int(state['step'])==steps and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all() for state in s['optimizer']['state'].values())
        assert s['optimizer']['state'] and s['noise_rng'].numel()>0 and s['torch_rng'].numel()>0 and s['numpy_order_rng']
        tv={video(r) for r in rec['train_ids']};assert tv<=trv and not tv&(calv|innerv)
        if 'prediction_held_ids' in rec:assert not tv&{video(r) for r in rec['prediction_held_ids']}
        checked.append(dict(kind=rec['kind'],steps=steps,parameters=sum(v.numel() for v in s['model'].values()),Adam_states=len(s['optimizer']['state'])))
    for line in saved['lineage']:
        held={video(r) for r in line['held_ids']}
        for rec in saved['records'][line['record_start']:line['record_end']]:assert not held&{video(r) for r in rec['train_ids']}
    assert saved['calibration']['cal_ids']==roles['calibration_ids']
    model=Mechanism.restore(saved,'cpu');modules=[model.stack.baseline,*model.stack.decompositions.values(),*model.stack.proposals.values(),*model.gates.values(),model.ordinary]
    before=[state_sha(m) for m in modules];rng=torch.get_rng_state().clone();actual,extra=model.predict(features)
    assert [state_sha(m) for m in modules]==before and torch.equal(rng,torch.get_rng_state())
    error={k:float(np.max(abs(actual[k]-pred[k]))) for k in pred};assert max(error.values())<=p['CPU_prediction_tolerance']
    # Real all-off comparison shares baseline; proposal cannot bypass its message.
    z=torch.tensor(model.stack.reducer.transform(features));off={}
    with torch.no_grad():
        for name,m in model.stack.proposals.items():
            d,ch=m(z,torch.zeros(len(ids),6,p['config']['latent_dim']));assert torch.equal(d,torch.zeros_like(d));off[name]=float(d.abs().max())
    a.out.mkdir();np.savez(a.out/'original_CPU_INNER_predictions.npz',row_ids=ids,**actual)
    write(a.out/'actual_cpu_audit.json',dict(status='INNOVATION_PILOT_COMPLETE_NATIVE_CPU_STATE_AND_REAL_HEAD_REPLAY_PASSED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,checkpoint_SHA=receipt['checkpoint_SHA'],train_archive_SHA=p['train_archive_SHA'],cache_archive_SHA=p['cache_archive_SHA'],CPU_prediction_error=error,predeclared_CPU_tolerance=p['CPU_prediction_tolerance'],all_off_delta=off,records=checked,whole_pipeline_video_lineage_passed=True,CAL_excluded_from_training_passed=True,parameters_and_RNG_unchanged=True,actual_small_head_CPU_forward=True,encoder_CPU_forward=False,labels_read=False,official_TEST_forward=False))
    print(read(a.out/'actual_cpu_audit.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','train','cache','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())
