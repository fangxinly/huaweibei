"""Read-only identity/schema/source audit. No pickle, tensor/model, or probe fit.

Numeric arrays read here are existing row-index partitions only. Feature and
label payloads are not decoded. Candidate NPZ inspection reads TRAIN NPY headers.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np


REVIEW = Path(r'C:\Users\21234\Documents\Codex\2026-10-06\multimodal-flow-research-review')
NI = Path(r'C:\Users\21234\Documents\Codex\2026-10-05\ni')
OLD = Path(r'C:\Users\21234\Documents\Codex\2026-10-01\1-zhang-s-yang-y-chen')
SNAPSHOT = Path(r'D:\CodexBackups\selective_flow_20261003_1105\finite_c2_completed_snapshots_20261005T1627Z\a\snapshot.zip')
DATASET = OLD / 'work/careflow_data/mosi.pkl'
EXPECTED_DATA_SHA = '5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bsha(b):
    return hashlib.sha256(b).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    dataset_sha = sha(DATASET)
    assert dataset_sha == EXPECTED_DATA_SHA
    local_mapping = NI / 'work/train_row_video_mapping.json'
    local_source = NI / 'work/inspect_train_group_mapping_v1.py'
    with zipfile.ZipFile(SNAPSHOT) as z:
        receipt_bytes = z.read('train_group_mapping/receipt.json')
        receipt = json.loads(receipt_bytes)
        mapping_bytes = z.read('train_group_mapping/train_row_video_mapping.json')
        source_bytes = z.read('train_group_mapping_source/inspect_train_group_mapping_v1.py')
        rows = json.loads(mapping_bytes)
        assert receipt['dataset_sha256'] == dataset_sha
        assert bsha(mapping_bytes) == sha(local_mapping) == receipt['mapping_sha256']
        assert bsha(source_bytes) == sha(local_source) == receipt['source_sha256']
        assert not receipt['dev_requested'] and not receipt['test_requested']
        assert len(rows) == receipt['rows'] == 1281
        assert [r['row'] for r in rows] == list(range(1281))
        assert len({r['segment_id'] for r in rows}) == 1281
        assert len({r['video_id'] for r in rows}) == receipt['video_groups'] == 52
        for r in rows:
            match = re.fullmatch(r'(.+)\[(\d+)\]', r['segment_id'])
            assert match and match[1] == r['video_id'] and int(match[2]) == r['segment_index']
        folds = []
        all_held = []
        group_sets = []
        for f in receipt['folds']:
            k = f['fold']
            hb = z.read(f'train_group_mapping/outer_hold_{k}.npy')
            ob = z.read(f'train_group_mapping/outer_other_{k}.npy')
            assert bsha(hb) == f['hold_sha256'] and bsha(ob) == f['other_sha256']
            held = np.load(io.BytesIO(hb), allow_pickle=False)
            other = np.load(io.BytesIO(ob), allow_pickle=False)
            assert held.dtype.kind in 'iu' and other.dtype.kind in 'iu'
            assert len(held) == f['hold_rows'] and len(other) == f['other_rows']
            assert len(set(held)) == len(held) and len(set(other)) == len(other)
            assert not set(held) & set(other)
            assert set(held) | set(other) == set(range(1281))
            groups = {rows[int(i)]['video_id'] for i in held}
            assert groups == set(f['hold_video_ids'])
            assert not groups & {rows[int(i)]['video_id'] for i in other}
            folds.append({'fold': k, 'hold_rows': len(held), 'other_rows': len(other), 'hold_videos': len(groups), 'hold_SHA256': bsha(hb), 'other_SHA256': bsha(ob)})
            group_sets.append(groups)
            all_held.extend(held.tolist())
        assert sorted(all_held) == list(range(1281))
        assert all(not x & y for i, x in enumerate(group_sets) for y in group_sets[i+1:])

    plan_results = []
    for mode in ['factorized_aux', 'regression_aux']:
        folder = NI / f'work/polarity_intensity_{mode}_resume100_qualified_20261008T205936Z'
        path = folder / 'qualified_resume_plan.json'
        plan = load_json(path)
        train_ids = plan['official_train_dev_IDs']['train']
        assert train_ids == [r['segment_id'] for r in rows]
        assert plan['asset_sha256']['assets/mosi.pkl'] == dataset_sha
        selected_sources = ['encoder_adapter.py', 'candidate_adapter.py', 'train_official.py', 'resume_official_mse100_v1.py', 'careflow_train_dev_author_components_candidate.py']
        checks = {name: sha(folder/name) == plan['source_sha256'][name] for name in selected_sources}
        assert all(checks.values())
        plan_results.append({'candidate_mode': mode, 'plan_path': str(path), 'plan_SHA256': sha(path), 'TRAIN_ID_order_exactly_matches_mapping': True, 'dataset_SHA_bound': True, 'source_SHA_checks': checks, 'model_training_scope': 'full official TRAIN supervised; VAL checkpoint selection; final representation is not whole-pipeline OOF'})

    parallel = OLD / 'outputs/parallel_gpu_20261003_1800'
    folder = parallel / 'v0800/seven_aligned_seed2/attempt01'
    protocol = load_json(folder/'protocol.json')
    results = load_json(folder/'results.json')
    export_source = parallel / 'frozen_repeat_views_20261004T0800Z.py'
    training_source = parallel / 'matched_repeat_deployment_20261004T0650Z/seven_aligned/run_source_validation.py'
    assert sha(export_source) == protocol['audit_source_sha256']
    assert sha(training_source) == protocol['source_sha256']['run_source_validation.py']
    assert sha(folder/'protocol.json') == results['files_sha256']['protocol.json']
    feature_path = folder/'frozen_features.npz'
    assert sha(feature_path) == results['files_sha256']['frozen_features.npz']
    plan = load_json(parallel/'frozen_repeat_views_plan_20261004T0800Z.json')
    selection = plan['jobs']['seven_aligned_seed2']['selection']
    assert selection['checkpoint_sha256'] == protocol['checkpoint_sha256']
    assert selection['protocol_sha256'] == protocol['predecessor_protocol_sha256']
    headers = {}
    with zipfile.ZipFile(feature_path) as z:
        keys = z.namelist()
        for modality in ['text', 'audio', 'visual']:
            key = f'train__preflow_{modality}.npy'
            with z.open(key) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                else:
                    raise ValueError(f'Unsupported NPY version {version}')
            assert shape == (1281, 300) and not dtype.hasobject
            headers[key] = {'shape': list(shape), 'dtype': str(dtype), 'fortran_order': fortran}
        identity_keys = [k for k in keys if any(s in k.lower() for s in ['row_id', 'video_id', 'speaker_id', 'fold_id'])]

    out = {
        'status': 'TRAIN_IDENTITY_MAPPING_AND_SOURCE_BINDINGS_AUDITED_NO_PROBE_EXECUTED',
        'dataset': {'path': str(DATASET), 'SHA256': dataset_sha, 'bytes': DATASET.stat().st_size, 'pickle_materialized_in_this_audit': False},
        'mapping': {'local_path': str(local_mapping), 'original_snapshot': str(SNAPSHOT), 'original_receipt_member_SHA256': bsha(receipt_bytes), 'mapping_SHA256': bsha(mapping_bytes), 'source_SHA256': bsha(source_bytes), 'rows': 1281, 'videos': 52, 'unique_segment_ids': True, 'row_order_contiguous': True, 'segment_video_parse_passed': True, 'snapshot_and_local_bytes_match': True, 'original_dataset_binding_passed': True, 'official_dataset_IDs_redecoded_this_turn': False, 'scope': 'Reaudited saved source/receipt/mapping/partition chain, not a new decode of official pickle IDs.', 'folds': folds, 'whole_pipeline_OOF_predictions_exist_claimed': False},
        'current_AB_plan_bindings': plan_results,
        'old_feature_candidate': {'path': str(feature_path), 'SHA256': sha(feature_path), 'header_only_TRAIN_views': headers, 'explicit_identity_keys': identity_keys, 'export_source_path': str(export_source), 'export_source_SHA256': sha(export_source), 'training_source_path': str(training_source), 'training_source_SHA256': sha(training_source), 'bound_selected_epoch': selection['best_epoch'], 'bound_checkpoint_SHA256': protocol['checkpoint_sha256'], 'source_declared_export_order': 'shuffle=False, no explicit drop_last; concatenate in dataset iteration order', 'layout_by_source_not_numeric_rechecked': 'Three 100D blocks, one modal block retained, two zero blocks. Shape300 is padded, not 300 independent modal coordinates.', 'supervision': 'Seven availability task-MSE routes on full TRAIN, end-to-end parameters; VAL-selected model.', 'allowed_use': 'Candidate for fixed-trained-representation internal description after exact export-row lineage confirmation; not strict whole-pipeline OOF.', 'open_items': ['Recover and bind exact run_careflow dataset helper and predecessor training protocol to complete raw row-order lineage.', 'No per-row identity embedded in NPZ.', 'No fold-excluded supervision provenance; current source positively shows fullTRAIN supervision.', 'Video groups are not independently established speaker identities.']},
        'preexisting_pooled_split': {'path': str(NI/'work/anchored_increment_source_20261008T024739Z/split.json'), 'excluded_from_TRAIN_only_protocol': True, 'reason': 'Existing split pools original TRAIN/VAL/TEST2195 rows over93 videos; it is not the TRAIN1281/52-video population.'},
        'scope': {'feature_values_decoded': False, 'label_values_decoded': False, 'VAL_TEST_values_decoded': False, 'model_or_Torch_import': False, 'model_forward_or_training': False, 'new_probe_or_estimator_fit': False, 'remote_connection': False, 'credentials_processed': False, 'main_sources_changed': False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': out['status'], 'mapping_rows': 1281, 'mapping_videos': 52, 'fold_hold_rows': [f['hold_rows'] for f in folds], 'AB_official_TRAIN_ID_order_match': True, 'old_feature_bound_epoch': selection['best_epoch'], 'old_feature_strict_OOF': False, 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
