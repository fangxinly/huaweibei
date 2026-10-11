"""Prospective fresh public-instance whole selected replay; no supervision or optimizer updates."""
import argparse
import hashlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write, plan_gate, stage_association, zip_audit, stable_file, TRAIN_JOINT

def main():
    p = argparse.ArgumentParser()
    for n in ('root', 'bundle', 'assets', 'original-root', 'training-joint'):
        p.add_argument('--' + n, type=Path, required=True)
    for n in ('plan-sha', 'training-joint-sha'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--method', choices=('careflow', 'minimal_fixed_F'), required=True)
    a = p.parse_args()
    plan = plan_gate(a.bundle, a.plan_sha)
    require(sha(a.training_joint) == a.training_joint_sha, 'Original full training preservation joint SHA mismatch')
    joint = read(a.training_joint)
    parent, _ = stage_association(a.original_root, a.plan_sha, a.method)
    require(joint['status'] == TRAIN_JOINT and joint['method'] == a.method and
            joint['plan_sha256'] == a.plan_sha and joint['original_root'] == parent['root'] and
            joint['original_stage_receipt_sha256'] == sha(a.original_root / 'out/actual_stage_receipt.json') and
            joint['original_stage_exit_sha256'] == sha(a.original_root / 'natural_exit.json') and
            joint['whole_checkpoint_D_SHA_CRC_passed'] is True, 'Full original preservation gate before fresh construction')
    require(a.root.parent == Path('/data/coding') and a.root.name.startswith('paired_fulltrain_fresh_') and
            not (a.root / 'out').exists(), 'Fresh independent instance root required')
    selected = a.original_root / 'out/selected_best_full.pt'
    item = parent['selected_best_full']
    require(selected.stat().st_size == item['bytes'], 'Original whole selected byte size mismatch')
    zip_audit(selected, item['sha256'])
    for n, h in plan['asset_sha256'].items():
        require(sha(a.assets / n) == h, 'Fresh public asset mismatch')
    raw = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    compute = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    require(raw.strip() == plan['assigned_gpu_UUID'][a.method] and not compute.strip(), 'Actual fresh UUID/empty compute gate')
    require(shutil.disk_usage(a.root).free >= plan['remote_free_floor_bytes'], 'Fresh replay complete preservation space floor')
    out = a.root / 'out';out.mkdir()
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from paired_fulltrain_session_candidate import construct, predictions
    from fixed_flow_components_candidate import tensor_sha
    from paired_fulltrain_CPU_audit_candidate_v1 import rng_sha
    torch.set_num_threads(2)
    started = time.perf_counter()
    # New public/random instance and FIT statistics, without actual TRAIN or DEV target access.
    session = construct(a.method, a.bundle, a.assets, plan, supervision=False)
    clean = tensor_sha(session.model.state_dict())
    require(clean==read(a.original_root/'out/actual_construction_receipt.json')['initial_state_sha256'],
            'Fresh public/random initial entire state differs from original construction')
    stats_names = [n for n in session.model.state_dict() if n.startswith('dberta.v6_')]
    stats = tensor_sha({n: session.model.state_dict()[n] for n in stats_names})
    selected_value = torch.load(selected, map_location='cpu')
    require(selected_value['metadata'] == parent['metadata'] and tensor_sha(selected_value['model']) == parent['metadata']['best_state_SHA'], 'Original selected entire state binding')
    require(stats == tensor_sha({n: selected_value['model'][n] for n in stats_names}), 'Fresh FIT statistics differ from original selected statistics')
    session.model.load_state_dict(selected_value['model'], strict=True)
    session.clean_initial = None
    del selected_value
    state = tensor_sha(session.model.state_dict())
    def full_rng():
        return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().clone(),
                'cuda':[x.clone() for x in torch.cuda.get_rng_state_all()]}
    before_rng_sha=rng_sha(full_rng())
    def predict(dummy):
        session.model.eval(); pieces = []
        with torch.no_grad():
            for start in range(0,229,128):
                batch = [x[start:start+128].to(session.author.DEVICE) for x in session.dev_inputs.tensors]
                batch[3] = torch.full_like(batch[3],dummy)
                pieces.append(predictions(session,a.method,tuple(batch)).detach().cpu().numpy())
        return np.concatenate(pieces).astype(np.float32,copy=False)
    p0, p7 = predict(0), predict(7)
    with np.load(a.original_root/'out/selected_best_DEV_replay.npz',allow_pickle=False) as z:
        require(set(z.files)=={'row_ids','prediction','model_state_sha256'} and
                tuple(z['row_ids'].tolist())==session.guard.ids['dev'] and
                str(z['model_state_sha256'].item())==state and np.array_equal(p0,z['prediction']), 'Fresh entire selected original229 prediction replay differs')
    require(np.array_equal(p0,p7) and tensor_sha(session.model.state_dict())==state and
            before_rng_sha==rng_sha(full_rng()), 'Fresh dummy/state/allRNG invariance failure')
    require(not session.optimizer.state and session.scheduler.last_epoch==0 and
            not any(x.get('labels_read') for x in session.guard.journal), 'Fresh replay accessed supervision or optimizer state')
    torch.cuda.synchronize()
    peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
    require(max(peak.values())<=plan['max_gpu_peak_bytes'], 'Fresh cumulative construction/replay peak budget')
    np.savez(out/'fresh_selected_DEV_dummy_replay.npz',row_ids=np.asarray(session.guard.ids['dev']),
             prediction_dummy0=p0,prediction_dummy7=p7,model_state_sha256=np.asarray(state))
    result={'status':'ACTUAL_PAIRED_FRESH_PUBLIC_INSTANCE_WHOLE_SELECTED_REPLAY_COMPLETE_NO_LABELS_OR_FINAL_TEST',
            'actual_utc':now(),'pid':os.getpid(),'argv':sys.argv,'root':str(a.root),'source_bundle':str(a.bundle),
            'plan_sha256':a.plan_sha,'method':a.method,'original_training_joint_sha256':a.training_joint_sha,
            'original_stage_receipt_sha256':sha(a.original_root/'out/actual_stage_receipt.json'),
            'fresh_public_initial_state_sha256':clean,'selected_state_sha256':state,'fresh_statistics_sha256':stats,
            'external_checkpoint_reference':stable_file(selected),'full_checkpoint_new_download_this_replay':False,
            'original_prediction_replay_error':0,'dummy0vs7_error':0,'state_and_allRNG_unchanged':True,
            'guard_journal':session.guard.journal,'cumulative_peak_no_reset':peak,'elapsed_seconds':time.perf_counter()-started,
            'CPU_model_forward':False,'new_label_or_final_TEST_access':False,'actual_GPU_UUID_raw':raw,'actual_compute_raw':compute,
            'fresh_replay_D_capture_joint_pending':True}
    write(out/'actual_stage_receipt.json',result)
    print(__import__('json').dumps(result),flush=True)

if __name__=='__main__':main()
