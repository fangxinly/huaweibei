from pathlib import Path
import hashlib,json
w=Path(__file__).parent
s=(w/'assemble_finite_full_checkpoint_v1.py').read_text(encoding='utf-8').replace("choices=['a','b']","choices=['c']").replace('finite_task_risk_runtime_v1','finite_task_risk_runtime_v2').replace('finite_formal_plan_v1.json','finite_formal_plan_v2.json').replace("mode={'a':'fixed','b':'finite_scalar'}[a.node]","mode={'c':'finite_vector'}[a.node]")
# Preserve strict whole-model disk reload/229 official-input verification.
s=s.replace("assembly_node='a',","assembly_node='a',amendment_id=plan['amendment_id'],")
p=w/'assemble_finite_full_checkpoint_v2.py';assert not p.exists();p.write_text(s,encoding='utf-8')
print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
