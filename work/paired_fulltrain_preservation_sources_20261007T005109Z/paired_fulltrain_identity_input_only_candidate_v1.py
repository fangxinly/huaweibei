"""Actual prospective TRAIN/DEV IDs and raw input layout only, before final execution freeze.

The trusted pinned pickle materializes all entries; only TRAIN/DEV inputs and IDs are indexed.
No label, tokenizer, model, training, prediction or TEST entry indexing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write

def main():
    p = argparse.ArgumentParser()
    for n in ('root', 'bundle', 'assets'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--plan-sha', required=True)
    a = p.parse_args()
    file = a.bundle / 'paired_TRAIN_DEV_input_identity_plan.json'
    require(sha(file) == a.plan_sha, 'Input-only identity plan SHA mismatch')
    plan = read(file)
    require(plan.get('status') == 'PAIRED_TRAIN_DEV_IDENTITY_INPUT_ONLY_PROTOCOL_FROZEN' and
            plan.get('labels_enabled') is False and plan.get('model_or_training_enabled') is False and
            plan.get('TEST_entry_enabled') is False, 'Input-only identity protocol not frozen')
    for n, h in plan['source_sha256'].items():
        require(sha(a.bundle / n) == h, 'Exact identity source mismatch')
    require(a.root.parent == Path('/data/coding') and a.root.name.startswith('paired_fulltrain_identity_') and
            not (a.root / 'out').exists(), 'Fresh input-only root required')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    compute = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    require(gpu.strip() == plan['assigned_gpu_UUID'] and not compute.strip(), 'Actual assigned UUID/empty compute before identity access')
    require(shutil.disk_usage(a.root).free >= 4 * 1024**3, 'Actual identity preservation space floor')
    for n, h in plan['asset_sha256'].items():
        require(sha(a.assets / n) == h, 'Whole trusted input asset mismatch')
    out = a.root / 'out'; out.mkdir()
    import numpy as np
    from official_fulltrain_dev_guard_candidate import FullTrainDevGuard
    with (a.assets / 'assets/mosi.pkl').open('rb') as f:
        container = pickle.load(f)
    train, dev = container['train'], container['dev']
    del container
    guard = FullTrainDevGuard(train, dev)
    identity = hashlib.sha256(json.dumps(guard.ids, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    def input_layout(records):
        counts = {}
        for record in records:
            # record[1] is deliberately never accessed, even for diagnostics.
            inputs = record[0]
            signature = []
            for x in inputs:
                v = np.asarray(x)
                signature.append({'shape': list(v.shape), 'dtype': str(v.dtype)})
            key = json.dumps(signature, sort_keys=True)
            counts[key] = counts.get(key, 0) + 1
        return counts
    result = {'status': 'ACTUAL_OFFICIAL_TRAIN_DEV_INPUT_ONLY_IDENTITY_COMPLETE_NO_LABELS_OR_MODEL',
              'actual_utc': now(), 'pid': os.getpid(), 'argv': sys.argv, 'root': str(a.root),
              'source_bundle': str(a.bundle), 'plan_sha256': a.plan_sha,
              'ids': guard.ids, 'official_train_dev_ID_identity_sha256': identity,
              'original_public_asset_sha256': plan['asset_sha256'], 'train_input_layouts': input_layout(train),
              'dev_input_layouts': input_layout(dev), 'TRAIN_labels_read': False, 'DEV_labels_read': False,
              'TEST_entry_indexed': False, 'trusted_whole_pickle_other_role_bytes_materialized': True,
              'scale_semantics_not_established_by_shape_or_ID_only': True, 'actual_GPU_UUID_raw': gpu, 'actual_compute_raw': compute,
              'full_process_argv_raw': subprocess.run(['ps', '-eo', 'pid,ppid,args'], capture_output=True, text=True, check=True).stdout}
    write(out / 'actual_stage_receipt.json', result)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
