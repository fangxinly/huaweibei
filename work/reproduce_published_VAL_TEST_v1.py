"""Recompute saved official scores only; no model inference, fitting or selection."""
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'work/official_anchored_upgrade_20261008T053429Z/sentiment_metrics_careflow_v1.py'
    spec = importlib.util.spec_from_file_location('published_metrics', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads((root / 'results/official_upgrade_VAL_TEST.json').read_text(encoding='utf-8'))['roles']
    with np.load(root / 'results/official_upgrade_VAL_TEST_predictions.npz', allow_pickle=False) as archive:
        prediction = dict(archive)
    result = {}
    for role, count in [('VAL', 229), ('TEST', 685)]:
        with (root / 'results' / (role + '_actual_predictions.csv')).open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != count or len({r['row_id'] for r in rows}) != count:
            raise ValueError('Unexpected count or duplicate IDs')
        labels = [float(r['y']) for r in rows]
        if prediction[role + '_ids'].tolist() != [r['row_id'] for r in rows]:
            raise ValueError('CSV and frozen prediction IDs differ')
        result[role] = {}
        for column, reference in [('new', 'new'), ('messages_off', 'messages_off')]:
            # CSV float32 strings lose a few digits; retain frozen NPZ precision.
            key = role + ('_prediction' if column == 'new' else '_p0')
            scores = module.metrics(prediction[key], labels)
            for key in ['Acc7', 'Acc2', 'F1', 'MAE', 'Corr']:
                if abs(scores[key] - expected[role]['actual'][reference][key]) > 1e-12:
                    raise ValueError('Archived score mismatch: ' + role + '/' + column + '/' + key)
            result[role][column] = {k: scores[k] for k in ['Acc7', 'Acc2', 'F1', 'MAE', 'Corr']}
    fixed = json.loads((root / 'results/fixed_models_VAL_TEST_five_metrics.json').read_text(encoding='utf-8'))['models']
    with np.load(root / 'results/CaReFlow_and_fixed_F_TEST_predictions.npz', allow_pickle=False) as archive:
        if archive['row_ids'].tolist() != prediction['TEST_ids'].tolist() or not np.array_equal(archive['labels'], labels):
            raise ValueError('Baseline and upgrade TEST rows/labels differ')
        for name in ['careflow', 'minimal_fixed_F']:
            scores = module.metrics(archive[name], archive['labels'])
            for key in ['Acc7', 'Acc2', 'F1', 'MAE', 'Corr']:
                if abs(scores[key] - fixed[name]['TEST'][key]) > 1e-12:
                    raise ValueError('Archived baseline score mismatch')
            result['TEST'][name] = {k: scores[k] for k in ['Acc7', 'Acc2', 'F1', 'MAE', 'Corr']}
    print(json.dumps(dict(status='SAVED_PREDICTIONS_REPRODUCED_NOT_NEW_EXPERIMENT', roles=result), indent=2))


if __name__ == '__main__':
    main()
