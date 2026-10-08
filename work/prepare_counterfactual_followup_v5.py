from pathlib import Path
import ast,hashlib,json
p=Path(__file__).parent
s=(p/'capture_counterfactual_followup_v4.py').read_text(encoding='utf-8')
s=s.replace("'counterfactual_provisional_preservation_20261005T0542Z')", "'counterfactual_provisional_preservation_20261005T0542Z','counterfactual_preservation_20261005T0604Z','counterfactual_diagnostics_20261005T0604Z')")
ast.parse(s);(p/'capture_counterfactual_followup_v5.py').write_text(s,encoding='utf-8')
s=(p/'run_lease_capture_pair_v2.py').read_text(encoding='utf-8').replace('capture_counterfactual_followup_v4_20261005T0545Z.py','capture_counterfactual_followup_v5_20261005T0604Z.py')
ast.parse(s);(p/'run_lease_capture_pair_v3.py').write_text(s,encoding='utf-8')
for n in ['capture_counterfactual_followup_v5.py','run_lease_capture_pair_v3.py']:
 print(json.dumps({'name':n,'sha256':hashlib.sha256((p/n).read_bytes()).hexdigest()}))
