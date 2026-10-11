from pathlib import Path
import ast,json,hashlib
root=Path(__file__).resolve().parent
capture=(root/'capture_gated_followup_v1.py').read_text(encoding='utf-8')
capture=capture.replace("'gated_preservation_20261005T0328Z')", "'gated_preservation_20261005T0328Z',\n 'inflow_utility_v4_deployment_20261005T0346Z')")
p=root/'capture_utility_v4.py'
with p.open('x',encoding='utf-8',newline='\n') as f:f.write(capture)
ast.parse(capture)
print(json.dumps({'capture_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
