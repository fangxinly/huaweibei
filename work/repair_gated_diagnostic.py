from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parent
src=(root/'diagnose_gated_conditions_v1.py').read_text(encoding='utf-8')
assert "(run/'training.log').read_text()" in src
src=src.replace("(run/'training.log').read_text()","(c.source_root/'training.log').read_text()")
ast.parse(src)
p=root/'diagnose_gated_conditions_v2.py'
with p.open('x',encoding='utf-8',newline='\n') as f:f.write(src)
digest=hashlib.sha256(p.read_bytes()).hexdigest()
launcher=(root/'launch_gated_diag.py').read_text(encoding='utf-8')
launcher=launcher.replace('0330Z','0332Z').replace('diagnose_gated_conditions_v1_','diagnose_gated_conditions_v2_').replace('f1067166d7da9297351f047d9921f82b9fbf0c09dae97e3771863704151b46fa',digest)
with (root/'launch_gated_diag_v2.py').open('x',encoding='utf-8',newline='\n') as f:f.write(launcher)
capture=(root/'capture_continuation.py').read_text(encoding='utf-8')
capture=capture.replace("'inflow_preservation_20261005022433Z')", "'inflow_preservation_20261005022433Z',\n 'gated_diagnostics_20261005T0330Z',\n 'gated_diagnostics_20261005T0332Z',\n 'gated_preservation_20261005T0328Z')")
ast.parse(capture)
with (root/'capture_gated_followup_v1.py').open('x',encoding='utf-8',newline='\n') as f:f.write(capture)
print(json.dumps({'diagnostic_v2_sha256':digest,'capture_sha256':hashlib.sha256((root/'capture_gated_followup_v1.py').read_bytes()).hexdigest(),'repair':'Only training marker log path corrected to deployment root; failed earlier roots preserved.'}))
