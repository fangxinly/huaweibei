"""Correct Python sys.argv versus Popen executable argv in the original join.

The v1 assertion rejected before changing state or creating a completed archive.
Keep that original source and failure. Compare the actual executable separately.
"""
from pathlib import Path
p = Path(__file__).with_name('finalize_second_lease_activation_checkpoint_actual_v1.py')
s = p.read_text(encoding='utf-8')
old = "assert ex['child_pid'] == cpu['pid'] and ex['child_full_argv'] == cpu['argv']"
new = ("assert ex['child_pid'] == cpu['pid']\n"
       "assert ex['child_full_argv'][0] == '/data/coding/multimodal_flow_public_20261006T1341Z/.venv/bin/python'\n"
       "assert ex['child_full_argv'][1:] == cpu['argv']")
assert s.count(old) == 1
s = s.replace(old, new)
exec(compile(s, str(p), 'exec'))
