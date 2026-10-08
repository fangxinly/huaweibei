"""Prepare future reporting fixes; immutable historical sources are untouched."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
s = (root/'negative_gain_review_v1.py').read_text(encoding='utf-8-sig')
old = '本次四行c均大于a，凸混合最优均为b。'
assert s.count(old) == 1
s = s.replace(old, '本次各行的c、a及凸混合权重见上表；仅权重等于0的行以b为凸混合最优，不能预先断言所有行相同。')
ast.parse(s)
(root/'negative_gain_review_v2.py').write_text(s, encoding='utf-8')
s = (root/'message_output_backtrack_v1.py').read_text(encoding='utf-8-sig')
s = s.replace("'traces': traces, 'true_labels_consumed': False,", "'traces': traces, 'true_labels_consumed': None,\n            'target_provenance': 'UNVERIFIED_CALLER_ARRAY',\n            'target_provenance_verified': False,")
s = s.replace('Prepared inference primitive.', 'Unqualified inference primitive: caller target may contain labels; no provenance certificate.')
ast.parse(s)
(root/'message_output_backtrack_v2.py').write_text(s, encoding='utf-8')
print('Future versions prepared; no historical scoring or model execution')
