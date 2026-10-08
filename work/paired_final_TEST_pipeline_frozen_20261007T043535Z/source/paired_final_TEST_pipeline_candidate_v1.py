"""Fixed author-cache TEST pair: prospective prediction, original CPU audit and score.

No source in this module runs on import. Trusted pickle is not a sandbox.
Runtime plans must pin every source and completed parent before execution.
"""
import argparse
import datetime
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
from paired_final_TEST_guard_candidate_v1 import (
    PairedFinalTestGuard, METHODS, STATES, CHECKPOINTS, require, sha, read_pinned)
from paired_final_TEST_identity_candidate_v1 import inspect_inputs

PRED_STATUS = 'ACTUAL_FIXED_TEST_PAIR_PREDICTION_ONLY_COMPLETE'
CPU_STATUS = 'ACTUAL_FIXED_TEST_ORIGINAL_ARRAY_CPU_AUDIT_COMPLETE'
IDENTITY_STATUS = 'ACTUAL_FIXED_AUTHOR_MOSI_TEST685_INPUT_IDENTITY_A_D_OTHER_CPU_B_CAPTURE_JOINT_PASSED_NO_TEST_LABELS_OR_FORWARD'

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def file_spec(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def plan_gate(bundle, digest):
    bundle = Path(bundle)
    require(sha(bundle/'final_pair_execution_plan.json') == digest, 'Exact final execution plan required')
    plan = load(bundle/'final_pair_execution_plan.json')
    require(plan['status'] == 'FINAL_TEST_PAIR_EXECUTION_PROTOCOL_FROZEN' and
            plan['execution_enabled'] is True and
            plan['methods'] == list(METHODS) and plan['rows'] == 685 and
            plan['batch'] == 128 and plan['keep_tail'] is True and
            plan['readout'] == 'fixed_direct' and plan['old_TEST_access_disclosed'] is True,
            'Complete fixed pair execution scope required')
    for name, digest in plan['source_sha256'].items():
        require(sha(bundle/name) == digest, 'Final source/parent changed: '+name)
    require(load(bundle/plan['identity_joint_file'])['status'] == IDENTITY_STATUS,
            'Completed A/D/other CPU input identity joint required')
    a, b = (load(bundle/plan[n]) for n in ('identity_file', 'other_identity_file'))
    require(a['node'] == 'A' and b['node'] == 'B' and
            b['original_A_identity_sha256'] == sha(bundle/plan['identity_file']), 'Original identity provenance')
    for k in ('test_ids', 'train_ids', 'dev_ids', 'test_raw_input_sha256', 'test_original_input_layouts'):
        require(a[k] == b[k], 'Independent complete input identity differs')
    require(len(a['test_ids']) == 685 and a['all_role_labels_read'] is False and
            b['all_role_labels_read'] is False, 'Original complete label-free identity required')
    for method in METHODS:
        spec = plan['fixed_models'][method]
        require(spec['state_sha256'] == STATES[method] and
                spec['checkpoint_sha256'] == CHECKPOINTS[method], 'Fixed selected model changed')
        training = load(bundle/spec['training_joint_file'])
        fresh = load(bundle/spec['fresh_joint_file'])
        require(training['status'] == 'ACTUAL_PAIRED_METHOD_FULLTRAIN_GPU_D_OTHER_CPU_CAPTURE_JOINT_PASSED_NO_FINAL_TEST' and
                fresh['status'] == 'ACTUAL_PAIRED_FULLTRAIN_D_ORIGINAL_OTHER_CPU_AND_FRESH_PUBLIC_SELECTED_REPLAY_JOINT_PASSED_NO_FINAL_TEST' and
                training['method'] == fresh['method'] == method and
                fresh['whole_training_joint_sha256'] == sha(bundle/spec['training_joint_file']),
                'Original whole model CPU/fresh qualification required')
    oldplan = load(bundle/'training/paired_fulltrain_execution_plan.json')
    require(sha(bundle/'training/paired_fulltrain_execution_plan.json') == plan['training_plan_sha256'] and
            oldplan['asset_sha256'] == plan['asset_sha256'], 'Unchanged original construction/assets')
    return plan

def native_gate(plan, assets, node, root):
    commands = {'UUID': ['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
                'compute': ['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],
                'fullargv': ['ps','-eo','pid,ppid,args'], 'space': ['df','-B1',str(root)]}
    raw = {k: subprocess.run(v, check=True, capture_output=True, text=True).stdout for k,v in commands.items()}
    require(raw['UUID'].strip() == plan['allowed_node_UUID'][node] and not raw['compute'].strip(),
            'Fresh assigned UUID and empty compute required')
    free = shutil.disk_usage(root).free
    remaining = (datetime.datetime.fromisoformat(plan['conservative_lease_end_UTC'])-
                 datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    require(free >= plan['remote_free_floor_bytes'] and
            remaining >= plan['execution_budget_seconds']+7200, 'Execution and two-hour preservation reserve')
    checked = {n: sha(Path(assets)/n) for n in plan['asset_sha256']}
    require(checked == plan['asset_sha256'], 'Physical complete public assets changed')
    versions={name:importlib.metadata.version(name) for name in plan['runtime_exact_versions']}
    require(sys.platform=='linux' and versions==plan['runtime_exact_versions'], 'Fresh original Linux24 plus Torch dependency versions')
    return {'actual_utc': utc(), 'native_raw': raw, 'free_bytes': free,
            'remaining_conservative_lease_seconds': remaining,
            'human_lease_not_platform_confirmation': True,
            'human_asset_budget_provenance': plan['human_asset_budget_provenance'],
            'actual_asset_sha256': checked,'actual_runtime_versions':versions,
            'actual_python':sys.version,'actual_python_executable':sys.executable}

def container_gate(bundle, assets, plan):
    with (Path(assets)/'assets/mosi.pkl').open('rb') as f:
        data = pickle.load(f)
    observed = inspect_inputs(data)
    original = load(Path(bundle)/plan['identity_file'])
    for key, value in observed.items():
        require(original[key] == value, 'Actual complete TEST input/ID changed: '+key)
    return data

def fresh_input_gate(root):
    native=load(Path(root)/'out/actual_native_preflight.json')
    age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(native['actual_utc'])).total_seconds()
    require(0<=age<=300,'Input execution requires at most five-minute actual native evidence')

def prediction_array(path, ids, state):
    raw = Path(path).read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        require(set(z.files) == {'row_ids','prediction','model_state_sha256'}, 'Prediction-only schema')
        require(tuple(z['row_ids'].tolist()) == tuple(ids) and
                str(z['model_state_sha256'].item()) == state, 'Original complete row order/state')
        values = z['prediction']
        require(values.dtype == np.float32 and values.shape == (len(ids),) and np.isfinite(values).all(),
                'All original finite FP32 predictions required')
        return values.copy()

def original_prediction_gate(root, plan, digest, method, identity_sha):
    root = Path(root)
    launch, exitdata = load(root/'actual_child_launch.json'), load(root/'natural_exit.json')
    require(exitdata['natural_exit'] is True and exitdata['exit_code'] == 0 and
            exitdata['child_pid'] == launch['child_pid'] and exitdata['fullargv'] == launch['fullargv'],
            'Original natural0/fullargv association')
    require(sha(root/'out/actual_stage_receipt.json') == exitdata['stage_receipt_sha256'], 'Original receipt exact SHA')
    r = load(root/'out/actual_stage_receipt.json')
    spec = plan['fixed_models'][method]
    require(r['status'] == PRED_STATUS and r['method'] == method and r['labels_read'] is False and
            r['plan_sha256'] == digest and r['pid'] == launch['child_pid'] and
            r['fullargv'] == launch['fullargv'] and r['root'] == launch['root'] and
            r['identity_sha256'] == identity_sha and r['state_sha256'] == STATES[method] and
            r['checkpoint_sha256'] == CHECKPOINTS[method] and r['node'] == spec['node'] and
            r['state_before'] == r['state_after'] == STATES[method] and
            r['statistics_before'] == r['statistics_after'] == spec['statistics_sha256'] and
            r['rng_before'] == r['rng_after'] and r['dummy0vs7_exact_equal'] is True and
            r['optimizer_steps'] == 0 and r['batch_sizes'] == [128]*5+[45] and
            r['model_parameter_count'] == spec['parameters'] and r['model_state_count'] == spec['model_states'],
            'Fixed original prediction/checkpoint/statistics/RNG/budget qualification')
    for name,h in plan['source_sha256'].items(): require(sha(root/'source'/name) == h, 'Original source bytes changed')
    require(sha(root/'source/final_pair_execution_plan.json') == digest, 'Original plan bytes changed')
    require(sha(root/'out/actual_native_preflight.json') == r['native_sha256'], 'Original native receipt linkage')
    native = load(root/'out/actual_native_preflight.json')
    require(native['actual_asset_sha256'] == plan['asset_sha256'] and
            native['actual_runtime_versions'] == plan['runtime_exact_versions'] and
            native['native_raw']['UUID'].strip() == plan['allowed_node_UUID'][r['node']] and
            not native['native_raw']['compute'].strip(), 'Original native assets/UUID/compute')
    ids = load(root/'source'/plan['identity_file'])['test_ids']
    p0 = prediction_array(root/'out/prediction.npz', ids, STATES[method])
    p7 = prediction_array(root/'out/prediction_dummy7.npz', ids, STATES[method])
    require(np.array_equal(p0,p7) and sha(root/'out/prediction.npz') == r['prediction_sha256'] and
            sha(root/'out/prediction_dummy7.npz') == r['dummy_prediction_sha256'], 'Physical original prediction/dummy SHA/equality')
    return r

def predict(a, plan):
    require(a.node == plan['fixed_models'][a.method]['node'], 'Assigned original GPU node')
    original = Path(plan['fixed_models'][a.method]['original_root'])
    selected = original/'out/selected_best_full.pt'
    require(selected.stat().st_size == plan['fixed_models'][a.method]['checkpoint_bytes'] and
            sha(selected) == CHECKPOINTS[a.method], 'Original selected full checkpoint bytes')
    # Model construction reuses the original pinned train/dev-only components.
    # It reconstructs TRAIN input statistics, never refits on TEST or reads supervision.
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(a.bundle/'training'))
    import torch
    from paired_fulltrain_session_candidate import construct, predictions
    from fixed_flow_components_candidate import tensor_sha
    from paired_fulltrain_CPU_audit_candidate_v1 import rng_sha
    torch.set_num_threads(2)
    require(torch.__version__ == plan['torch_version'] and np.__version__ == plan['numpy_version'], 'Pinned numerical runtime')
    session = construct(a.method, a.bundle/'training', a.assets,
                        load(a.bundle/'training/paired_fulltrain_execution_plan.json'), supervision=False)
    require(tensor_sha(session.model.state_dict()) == plan['fixed_models'][a.method]['initial_state_sha256'],
            'Original qualified public/random initial model')
    value = torch.load(selected, map_location='cpu')
    require(tensor_sha(value['model']) == STATES[a.method], 'Whole selected state SHA')
    session.model.load_state_dict(value['model'], strict=True)
    session.clean_initial = None; del value
    fresh_input_gate(a.root)
    data = container_gate(a.bundle, a.assets, plan)
    identity = {'path': str(a.bundle/plan['identity_file']), 'sha256': sha(a.bundle/plan['identity_file'])}
    guard = PairedFinalTestGuard(data['test'], identity); del data
    dataset = session.author.get_appropriate_dataset(guard.inputs_only(0))
    require(len(dataset) == 685 and not any(x.get('labels_read') for x in session.guard.journal), 'No actual target access')
    statnames = [n for n in session.model.state_dict() if n.startswith('dberta.v6_')]
    def statistics(): return tensor_sha({n:session.model.state_dict()[n] for n in statnames})
    def rng(): return rng_sha({'python':random.getstate(),'numpy':np.random.get_state(),
                              'torch':torch.get_rng_state().clone(), 'cuda':[x.clone() for x in torch.cuda.get_rng_state_all()]})
    state0, stat0, rng0 = tensor_sha(session.model.state_dict()), statistics(), rng()
    require(stat0 == plan['fixed_models'][a.method]['statistics_sha256'], 'Original fixed TRAIN-only statistics')
    def run(dummy):
        pieces=[]; session.model.eval()
        with torch.no_grad():
            for start in range(0,685,128):
                batch=[x[start:start+128].to(session.author.DEVICE) for x in dataset.tensors]
                batch[3]=torch.full_like(batch[3],dummy)
                pieces.append(predictions(session,a.method,tuple(batch)).detach().cpu().numpy())
        values=np.concatenate(pieces).astype(np.float32,copy=False)
        require(values.shape==(685,) and np.isfinite(values).all(), 'Complete fixed TEST prediction')
        return values
    p0,p7=run(0),run(7)
    state1,stat1,rng1=tensor_sha(session.model.state_dict()),statistics(),rng()
    require(np.array_equal(p0,p7) and state0==state1==STATES[a.method] and stat0==stat1 and rng0==rng1,
            'Original fixed prediction dummy/all-state/all-RNG invariance')
    require(not session.optimizer.state and session.scheduler.last_epoch==0, 'No optimizer steps')
    torch.cuda.synchronize()
    peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
    require(max(peak.values()) <= plan['max_gpu_peak_bytes'], 'Cumulative peak without reset budget')
    for name,pred in (('prediction.npz',p0),('prediction_dummy7.npz',p7)):
        with (a.root/'out'/name).open('xb') as f:
            np.savez(f,row_ids=np.asarray(guard.ids),prediction=pred,model_state_sha256=np.asarray(state0))
            f.flush();os.fsync(f.fileno())
    return {'status':PRED_STATUS,'method':a.method,'node':a.node,'labels_read':False,
            'identity_sha256':identity['sha256'],'state_sha256':state0,'checkpoint_sha256':CHECKPOINTS[a.method],
            'state_before':state0,'state_after':state1,'statistics_before':stat0,'statistics_after':stat1,
            'rng_before':rng0,'rng_after':rng1,'dummy0vs7_exact_equal':True,
            'prediction_sha256':sha(a.root/'out/prediction.npz'),'dummy_prediction_sha256':sha(a.root/'out/prediction_dummy7.npz'),
            'model_parameter_count':sum(p.numel() for p in session.model.parameters()),
            'model_state_count':len(session.model.state_dict()),'optimizer_steps':0,'batch_sizes':[128]*5+[45],
            'guard_journal':guard.journal,'construction_guard_journal':session.guard.journal,
            'cumulative_peak_no_reset':peak,'CPU_model_forward':False,'trusted_pickle_all_role_bytes_materialized':True}

def cpu_audit(a, plan):
    require(a.node=='B' and a.original is not None, 'Independent B CPU original audit only')
    fresh_input_gate(a.root)
    data=container_gate(a.bundle,a.assets,plan);del data
    r=original_prediction_gate(a.original,plan,a.plan_sha,a.method,sha(a.bundle/plan['identity_file']))
    return {'status':CPU_STATUS,'method':a.method,'node':'B','labels_read':False,'CPU_model_forward':False,
            'original_root':r['root'],'original_receipt_sha256':sha(a.original/'out/actual_stage_receipt.json'),
            'original_natural_exit_sha256':sha(a.original/'natural_exit.json'),
            'prediction_sha256':r['prediction_sha256'],'identity_sha256':r['identity_sha256'],
            'state_sha256':r['state_sha256'],'checkpoint_sha256':r['checkpoint_sha256'],
            'original_complete_ID_input_source_natural_exit_arrays_invariance_passed':True}

def score(a, plan):
    require(a.score_protocol is not None and a.score_protocol_sha is not None and a.node=='B', 'Frozen joint score binding required')
    binding=read_pinned({'path':str(a.score_protocol),'sha256':a.score_protocol_sha})
    require(binding['execution_plan_sha256']==a.plan_sha and binding['score_source_sha256']==sha(__file__) and
            binding['once_token_path']==plan['fixed_score_once_token_path'] and
            binding['binding_source_sha256']==plan['source_sha256']['paired_final_TEST_score_binding_candidate_v1.py'],
            'Immutable source/once token path binding')
    for method in METHODS:
        j=read_pinned(binding['predictions'][method]['preservation_joint'])
        require(j['execution_plan_sha256']==a.plan_sha and
                j['joint_source_sha256']==plan['source_sha256']['paired_final_TEST_saved_joint_candidate_v1.py'],
                'Exact final original CPU/capture joint source and execution plan')
    fresh_input_gate(a.root)
    data=container_gate(a.bundle,a.assets,plan)
    identity={'path':str(a.bundle/plan['identity_file']),'sha256':sha(a.bundle/plan['identity_file'])}
    guard=PairedFinalTestGuard(data['test'],identity);del data
    try:
        y,p=guard.targets_once_after_joint_preservation({'path':str(a.score_protocol),'sha256':a.score_protocol_sha})
    finally:
        token=Path(plan['fixed_score_once_token_path'])
        if token.exists():shutil.copyfile(token,a.root/'out/final_score_attempt_intent.json')
    from sentiment_metrics_careflow_v1 import metrics, SEMANTICS
    results={m:metrics(p[m],y) for m in METHODS}
    for m in METHODS:
        q=p[m].astype(np.float64);yc=y-y.mean();qc=q-q.mean()
        require(abs(results[m]['MAE']-float(sum(abs(float(x)-float(t)) for x,t in zip(q,y))/len(y)))<=1e-14,
                'Independent original-scale MAE audit')
        independent_corr=float(np.corrcoef(q,y)[0,1])
        require(results[m]['Corr'] is not None and abs(results[m]['Corr']-independent_corr)<=1e-14,
                'Independent Pearson audit')
        cm=results[m]['nonzero_confusion_matrix'];n=sum(map(sum,cm));f1=0.
        for k in (0,1):
            support=sum(cm[k]);denom=2*cm[k][k]+cm[k][1-k]+cm[1-k][k]
            f1+=support*(2*cm[k][k]/denom if denom else 0.)/n
        require(abs(f1-results[m]['F1'])<=1e-14,'Independent support-weighted F1 audit')
    five=('Acc7','Acc2','F1','MAE','Corr')
    wins={k:(results[METHODS[0]][k]<results[METHODS[1]][k] if k=='MAE' else
             results[METHODS[0]][k]>results[METHODS[1]][k]) for k in five}
    with (a.root/'out/joint_targets_and_fixed_predictions.npz').open('xb') as f:
        np.savez(f,row_ids=np.asarray(guard.ids),labels=y,**{m:p[m] for m in METHODS});f.flush();os.fsync(f.fileno())
    return {'status':'ACTUAL_FIXED_AUTHOR_CACHE_TEST685_ONCE_JOINT_FIVE_SCORE_COMPLETE',
            'labels_read':True,'methods':list(METHODS),'metrics':results,'strict_F_wins':wins,
            'all_five_strict_F_wins':all(wins.values()),'semantics':SEMANTICS,'guard_journal':guard.journal,
            'score_protocol_sha256':a.score_protocol_sha,'no_model_forward':True,
            'old_TEST_access_disclosed':True,'not_paper_unrounded_or_raw_SDK686_confirmation':True}

def main():
    p=argparse.ArgumentParser()
    for n in ('root','bundle','assets'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=('A','B','C'),required=True)
    p.add_argument('--stage',choices=('predict','cpu','score'),required=True)
    p.add_argument('--method',choices=METHODS);p.add_argument('--original',type=Path)
    p.add_argument('--score-protocol',type=Path);p.add_argument('--score-protocol-sha')
    a=p.parse_args();started=time.perf_counter();plan=plan_gate(a.bundle,a.plan_sha)
    require(a.root.parent==Path('/data/coding') and a.root.name.startswith('paired_final_TEST_') and
            not (a.root/'out').exists(),'Unique new final-stage root')
    require(a.stage=='score' or a.method in METHODS,'Fixed method required')
    require(str(a.root)==plan['fixed_stage_roots'][a.stage][a.method if a.stage!='score' else 'pair'],
            'One original root per fixed method/stage; alternate execution roots forbidden')
    (a.root/'out').mkdir()
    native=native_gate(plan,a.assets,a.node,a.root);write(a.root/'out/actual_native_preflight.json',native)
    result={'predict':predict,'cpu':cpu_audit,'score':score}[a.stage](a,plan)
    elapsed=time.perf_counter()-started
    require(elapsed<=plan['execution_budget_seconds'],'Frozen stage wall time budget exceeded')
    result.update(actual_utc=utc(),pid=os.getpid(),fullargv=sys.argv,root=str(a.root),
                  plan_sha256=a.plan_sha,native_sha256=sha(a.root/'out/actual_native_preflight.json'),
                  elapsed_seconds=elapsed)
    write(a.root/'out/actual_stage_receipt.json',result)
    print(json.dumps({'status':result['status'],'receipt_sha256':sha(a.root/'out/actual_stage_receipt.json'),
                      'actual_utc':result['actual_utc'],'labels_read':result['labels_read']}),flush=True)

if __name__=='__main__':main()
