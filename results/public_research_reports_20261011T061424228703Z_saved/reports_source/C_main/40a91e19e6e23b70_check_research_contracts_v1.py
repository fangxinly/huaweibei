"""Synthetic assertions only; does not open data, weights, or credentials."""
import contextlib
import importlib.util
import io
import json
import subprocess
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np
from external_pretraining_guard_v1 import check_metadata


def run():
    # A fixed representation creates no information, yet expands what a linear
    # readout can represent. This refutes the 'estimation error only' inference.
    x = np.array([-2., -1., 0., 1., 2.])
    y = x ** 2
    b = np.column_stack([np.ones(len(x)), x])
    z = np.column_stack([np.ones(len(x)), x ** 2])
    linear_risk = np.mean((b @ np.linalg.lstsq(b, y, rcond=None)[0] - y) ** 2)
    transformed_risk = np.mean((z @ np.linalg.lstsq(z, y, rcond=None)[0] - y) ** 2)
    assert linear_risk > 0 and transformed_risk < 1e-20
    # Independent modalities have zero mean predictability, but can jointly
    # determine the label. Residual size/conditional variance alone is not utility.
    a = np.array([-1., -1., 1., 1.])
    c = np.array([-1., 1., -1., 1.])
    for value in (-1., 1.):
        assert c[a == value].mean() == 0
    assert np.mean((a * c) ** 2) == 1
    target = [dict(sample_id='t', video_id='T', role='train'),
              dict(sample_id='v', video_id='V', role='val'),
              dict(sample_id='q', video_id='Q', role='test')]
    ext = [dict(sample_id='e', video_id='E', role='train')]
    result = check_metadata(target, ext, ['t'])
    assert not result['full_pretraining_qualification']
    bad_cases = [(target, ext, ['v']), (target, [dict(ext[0], video_id='Q')], ['t']),
                 (target, [dict(ext[0], role='test')], ['t']),
                 (target, [dict(ext[0], y=1.)], ['t']),
                 ([dict(r, media_sha256='a'*64) for r in target], [dict(ext[0], media_sha256='a'*64)], ['t'])]
    for args in bad_cases:
        try: check_metadata(*args)
        except ValueError: pass
        else: raise AssertionError('Leakage fixture was accepted')
    # Import and default CLI of the guarded helper must not contact the helper
    # or network. A real credential is never requested in this check.
    with patch.object(subprocess, 'run', side_effect=AssertionError('credential read')), \
         patch.object(urllib.request, 'urlopen', side_effect=AssertionError('network')):
        spec = importlib.util.spec_from_file_location('guarded_access', Path(__file__).with_name('github_access_check_guarded_v2.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = io.StringIO()
        with contextlib.redirect_stdout(out): module.main([])
        assert json.loads(out.getvalue())['credential_helper_read'] is False
    return dict(status='SYNTHETIC_CONTRACTS_PASS', fixed_feature_approximation_counterexample=True,
                conditional_novelty_not_task_utility_counterexample=True, rejected_leakage_fixtures=len(bad_cases),
                credential_import_and_default_no_private_access=True, real_data_or_model_execution=False)


if __name__ == '__main__': print(json.dumps(run()))
