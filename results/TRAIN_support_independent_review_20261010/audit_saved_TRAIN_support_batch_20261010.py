"""Review saved reports only; never import model sources or decode raw/NPZ arrays."""
from pathlib import Path
import ast
import collections
import hashlib
import io
import json
import zipfile

BASE = Path('D:/CodexBackups/selective_flow_20261003_1105/candidate_final100_actual_20261009T012656Z/TRAIN_support_retry_actual_20261010T004247Z')
EXPECTED_SHA = '0f45efd12ff9a52361727018cd12c5aa4e08236aaea6c04b5be1910e791309ba'
OUT = Path(__file__).resolve().parents[1] / 'outputs/TRAIN支持诊断独立审阅_20261010.json'
DIMENSIONS = {'T': 257, 'T_A': 406, 'T_V': 352, 'T_A_V': 501}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def check_archive(z):
    names = z.namelist()
    manifest = json.loads(z.read('capture_member_manifest.json'))
    assert z.testzip() is None and len(names) == len(set(names))
    assert len(manifest) == len({r['name'] for r in manifest})
    assert set(names) == {r['name'] for r in manifest} | {'capture_member_manifest.json'}
    for row in manifest:
        data = z.read(row['name'])
        assert len(data) == row['bytes'] and digest(data) == row['sha256']
    return {'total_entries': len(names), 'content_members': len(manifest),
            'all_member_SHA_size_CRC_unique_exactset_passed': True}

def main():
    data = (BASE / 'complete_actual_TRAIN_support_original.zip').read_bytes()
    assert digest(data) == EXPECTED_SHA
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        archive_audit = check_archive(z)
        with zipfile.ZipFile(io.BytesIO(z.read('payload.zip'))) as inner:
            payload_audit = check_archive(inner)
            for name in inner.namelist():
                assert inner.read(name) == z.read('payload/' + name)
        extracted_diagnostic = []
        for name in z.namelist():
            if name.startswith('diagnostic/'):
                assert (BASE / 'verified_original' / name).read_bytes() == z.read(name)
                extracted_diagnostic.append(name)
        read = lambda name: json.loads(z.read(name))
        plan = read('diagnostic/execution_plan.json')
        protocol = read('payload/frozen_support_protocol.json')
        result = read('diagnostic/support_result.json')
        exit_record = read('natural_exit.json')
        once = read('once_support_diagnostic.json')
        ledger = read('diagnostic/decoder_ledger.json')
        source_hashes = {}
        for name, expected in plan['source_SHA'].items():
            source = z.read('diagnostic/' + name)
            assert digest(source) == expected and source == z.read('payload/' + name)
            source_hashes[name] = expected
        assert digest(z.read('diagnostic/execution_plan.json')) == exit_record['plan_SHA'] == once['plan_SHA']
        assert digest(z.read('payload/frozen_support_protocol.json')) == plan['protocol_SHA'] == result['input_SHA']['protocol']
        for name in ['data', 'rows', 'states', 'fits', 'predictions']:
            assert result['input_SHA'][name] == plan[name + '_SHA']
        assert exit_record['child'] == 1785 and exit_record['natural_exit'] == 0
        assert result['rows'] == 1281 and result['videos'] == 52 and result['fold_paths'] == 20
        assert result['max_original_prediction_reconstruction_error'] == 0.0
        assert result['TRAIN_labels_decoded'] == 0 and not result['VAL_TEST_numeric_decode']
        assert not result['new_solve_fit_score'] and not result['primary_results_changed']
        assert ledger == {'decoded_arrays': 2562, 'decoded_roles': ['train'], 'decoded_TRAIN_labels': 0}

        channels = read('diagnostic/per_channel_saved_fit_support.json')
        videos = read('diagnostic/per_video_saved_fit_support.json')
        counts = read('diagnostic/TRAIN_raw_nonfinite_and_zero_counts.json')
        assert len(channels) == 20 and len(videos) == 208 and len(counts) == 1281
        assert {(g['fold'], g['path']) for g in channels} == {(f, p) for f in range(5) for p in DIMENSIONS}
        for g in channels:
            assert [c['index'] for c in g['channels']] == list(range(DIMENSIONS[g['path']]))
        assert len({r['row_id'] for r in counts}) == 1281
        video_counts = collections.Counter(r['video_id'] for r in counts)
        assert len(video_counts) == 52
        assert len({(r['video_id'], r['path']) for r in videos}) == 208
        for path in DIMENSIONS:
            assert {r['video_id']: r['rows'] for r in videos if r['path'] == path} == dict(video_counts)
        video_folds = {}
        for row in videos:
            assert video_folds.setdefault(row['video_id'], row['fold']) == row['fold']
        nonfinite = {'A': 0, 'V': 0}
        for row in counts:
            assert row['word_rows'] > 0
            for name, width in [('A', 74), ('V', 47)]:
                c = row[name]
                assert c['elements_per_channel'] == row['word_rows']
                for key in ['nonfinite_counts', 'finite_zero_counts']:
                    assert len(c[key]) == width
                    assert all(type(v) is int and 0 <= v <= row['word_rows'] for v in c[key])
                nonfinite[name] += sum(c['nonfinite_counts'])
        assert nonfinite == {'A': 0, 'V': 0}
        inactive_missingness = []
        for g in channels:
            indices = {'T': [], 'T_A': [405], 'T_V': [351], 'T_A_V': [405, 500]}[g['path']]
            for k in indices:
                c = g['channels'][k]
                assert not c['active'] and c['coefficient'] == c['fit_mean'] == c['fit_min'] == c['fit_max'] == c['held_min'] == c['held_max'] == 0
                inactive_missingness.append({'fold': g['fold'], 'path': g['path'], 'index': k})

        summary = json.loads((BASE / 'saved_support_independent_summary.json').read_bytes())
        assert summary['result'] == result and summary['nonfinite_totals'] == nonfinite
        extrema = {}
        for path in DIMENSIONS:
            rows = [r for r in videos if r['path'] == path]
            largest = max(rows, key=lambda r: r['max_abs_coefficient_term'])
            assert summary['paths'][path]['max_term_video'] == largest
            max_z = max(r['max_abs_standardized_feature'] for r in rows)
            gt1000 = sum(r['standardized_elements_gt1000'] for r in rows)
            assert summary['paths'][path]['max_abs_standardized_feature'] == max_z
            assert summary['paths'][path]['elements_gt1000'] == gt1000
            extrema[path] = {'largest_saved_term': largest, 'max_saved_abs_Z': max_z,
                             'saved_elements_gt1000': gt1000}
        selected_channels = []
        for g in channels:
            indices = {'T': [], 'T_A': [290], 'T_V': [299], 'T_A_V': [290, 448]}[g['path']]
            selected_channels.extend({'fold': g['fold'], 'path': g['path'], **g['channels'][k]} for k in indices)
        selected_counts = [r for r in counts if r['row_id'] in ['Iu2PFX3z_1s[11]', '2WGyTLYerpo[46]']]
        calls = [ast.unparse(n.func) for n in ast.walk(ast.parse(z.read('diagnostic/TRAIN_saved_fit_support_diagnostic_v1.py').decode())) if isinstance(n, ast.Call)]
        assert 'np.linalg.solve' not in calls and 'core.fit_predict' not in calls
        release = json.loads((BASE / 'Release_receipt.json').read_bytes())
        assert release['source_SHA'] == EXPECTED_SHA and release['remote_digest_verified']
        output = {
            'role': 'INDEPENDENT_REVIEW_OF_SAVED_RECORDS_ONLY', 'date_Asia_Shanghai': '2026-10-10',
            'archive_SHA256': EXPECTED_SHA, 'archive_bytes': len(data), 'archive_audit': archive_audit,
            'nested_payload_audit': payload_audit, 'matching_extracted_diagnostic_members': extracted_diagnostic,
            'source_SHA': source_hashes, 'frozen_protocol': protocol, 'original_natural_exit': exit_record,
            'original_once_receipt': once, 'original_result': result, 'original_decode_ledger': ledger,
            'saved_record_shape_and_identity_consistency_passed': True,
            'saved_nonfinite_count_totals': nonfinite, 'saved_missingness_features_inactive': inactive_missingness,
            'independent_saved_summary_matches_originals': True, 'extrema_from_saved_records': extrema,
            'selected_saved_channels_all_folds': selected_channels, 'selected_saved_zero_count_records': selected_counts,
            'Release_verification_basis': 'READ_PRESERVED_RECEIPT_NO_NEW_REMOTE_ACCESS',
            'Release_receipt': release, 'review_computation': 'ZIP integrity and saved-JSON arithmetic/consistency only',
            'raw_or_NPZ_array_decodes_by_this_review': 0, 'model_source_imports_by_this_review': 0,
            'new_fit_forward_support_diagnostic_or_score_by_this_review': False,
            'single_term_is_error_or_causal_attribution': False,
            'upstream_units_centering_and_missingness_semantics_identified': False,
        }
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(OUT), 'output_SHA256': digest(OUT.read_bytes()),
                      'archive_audit': archive_audit, 'nested_payload_audit': payload_audit,
                      'saved_record_consistency_passed': True}, ensure_ascii=False))

if __name__ == '__main__':
    main()
