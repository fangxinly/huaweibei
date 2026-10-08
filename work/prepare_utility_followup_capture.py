from pathlib import Path
import ast,hashlib
p=Path(__file__).resolve().parent
s=(p/'capture_utility_v4.py').read_text(encoding='utf-8').replace(" 'inflow_utility_v4_deployment_20261005T0346Z')]"," 'inflow_utility_v4_deployment_20261005T0346Z',\n 'utility_diagnostics_20261005T0439Z',\n 'utility_preservation_20261005T0439Z')]")
assert 'utility_diagnostics_20261005T0439Z' in s
ast.parse(s);target=p/'capture_utility_followup_v1.py';assert not target.exists();target.write_text(s,encoding='utf-8')
print(hashlib.sha256(target.read_bytes()).hexdigest())
