from pathlib import Path
import json,ast,hashlib
old=Path('work/paired_formal_actual_dispatch_v2_20261007T014523Z/actual_fresh_formal_dispatch.py')
d=Path('work/paired_formal_actual_dispatch_v3_20261007T015004Z')
rec=json.loads((d/'exact_formal_dispatch_arguments_receipt.json').read_text())
t=old.read_text().replace('paired_fulltrain100_','paired_fulltrain_train100_').replace('20261007T014523Z','20261007T015004Z')
for method,before in [('minimal_fixed_F','425e58b788e1cb1ca6df5f26b3d6404e2503b918866cb5e0cfcf3708ec62e472'),('careflow','ecf43a1d791c0b1d8b28e5eb716486ac366c40f2e69b7c867c599a04e0d56570')]:
    assert t.count(before)==1
    t=t.replace(before,rec['methods'][method]['child_arguments_sha256'])
t=t.replace("require(a.root==Path(","require(str(a.root).startswith('/data/coding/paired_fulltrain_'),'Frozen wrapper root prefix')\nrequire(a.root==Path(")
ast.parse(t)
assert 'paired_fulltrain100_' not in t
assert all(x['root'].startswith('/data/coding/paired_fulltrain_') for x in rec['methods'].values())
target=d/'actual_fresh_formal_dispatch.py';assert not target.exists();target.write_text(t)
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/d.name/target.name;assert not D.exists();D.write_bytes(target.read_bytes())
print('NEW_OPERATOR_SOURCE_AST_D_PREFIX_ARGV_PASSED',hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)
