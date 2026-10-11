from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parent
src=(root/'diagnose_conditions.py').read_text(encoding='utf-8')
src=src.replace('seed=91811','seed=int(p[\'seed\'])').replace('author.set_random_seed(91811)',"assert p['seed']==91812\n    author.set_random_seed(int(p['seed']))")
src=src.replace("assert p['mode']==c.mode and sel['epochs']==100", "assert p['mode']==c.mode and sel['epochs']==100\n    history=json.loads((run/'history.json').read_text())\n    assert len(history)==100 and 'INFLOW_CONDITIONS_RUN_COMPLETE' in (run/'training.log').read_text()")
src=src.replace('result=.5*old+.5*torch.stack(features,1)', 'selected=torch.stack(features,1)*torch.tanh(self.feedback_gate)[None,:,None]\n            result=.5*old+.5*selected')
src=src.replace("('scale200',2.0,False),('donor_shift1'", "('scale200',2.0,False),('sign_flip',-1.0,False),('donor_shift1'")
src=src.replace("'stage1_heads':[", "'feedback_gates_raw':flow.feedback_gate.detach().cpu().tolist(),\n        'feedback_gates_tanh':torch.tanh(flow.feedback_gate).detach().cpu().tolist(),\n        'stage1_heads':[")
src=src.replace("'Ordinary unimodal predictions are not calibrated contribution/conflict truth.'", "'Ordinary unimodal predictions are not calibrated contribution/conflict truth.',\n                  'Gated donor intervention retains each trained receiver gate; sign flip is frozen inference only, not a sign-retrained model.'")
assert '91811' not in src
assert 'selected=torch.stack(features,1)*torch.tanh(self.feedback_gate)[None,:,None]' in src
assert 'sign_flip' in src
ast.parse(src)
target=root/'diagnose_gated_conditions_v1.py'
with target.open('x',encoding='utf-8',newline='\n') as f:f.write(src)
print(json.dumps({'file':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'status':'PREPARED_AST_ONLY_NOT_DEPLOYED_NOT_GPU_EXECUTED','constraints':'Requires full 100-epoch completion and idle matching GPU. No reuse of old v2 diagnostic results.'},indent=2))
