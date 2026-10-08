from pathlib import Path
import ast,hashlib,json
root=Path(__file__).resolve().parent
s=(root/'audit_gated_snapshot_v2.py').read_text(encoding='utf-8')
def edit(a,b):
 global s
 assert a in s,a
 s=s.replace(a,b)
edit('inflow_gated_v3_deployment_20261005T0241Z','inflow_utility_v4_deployment_20261005T0346Z')
edit("'state',5579","'fixed',6464");edit("'task',9328","'predicted',10379");edit("'none',5265","'none',6143")
edit('inflow_gated_v3_20261005T0241Z','inflow_utility_v4_20261005T0346Z');edit('capture_continuation.py','capture_utility_v4.py');edit('gated_checks','utility_checks');edit('inflow_conditions_v3','inflow_utility_v4');edit('91812','91813')
edit('0fd75bc483805cc495c6427897ecb75863fe30e3e3cb47542409f72d5e677c7c','2fbecdf63882b94317a7a6b8986e5edb66cb1b2679d7939a0530706f08da8af9')
edit('849ee16039c19e2fabd229102a824ec6d185f29f914a17ccd36a94b546825958','1c60592d3426320b61c67c3a739931703473bfe337d65bbc218884607f2b267b')
edit('8eb6e1577f0a8c407aab9f0d6c9dee22ec6bb96a20496e7ea8fec798334d529b','fb2341411057c1040ec4bf104bdd9edaf2c080a569e56a8bec6f5e0f84769b13')
edit("checks['initial_zero_gate_equality_verified']","checks['initial_zero_feedback_equality_verified'] and checks['label_isolation_verified'] and checks['utility_gradient_detach_verified']")
edit("assert len(row['feedback_gates'])==3 and np.isfinite(row['feedback_gates']).all()\n                assert np.max(np.abs(row['feedback_gates']))<=1\n                if mode=='none':assert row['feedback_gates']==[0,0,0] and row['context_norm']==0", "assert len(row['last_train_batch_utility_weights'])==6 and np.isfinite(row['last_train_batch_utility_weights']).all()\n                assert min(row['last_train_batch_utility_weights'])>=0 and max(row['last_train_batch_utility_weights'])<=1\n                if mode=='none':assert row['last_train_batch_utility_weights']==[0]*6 and row['context_norm']==0\n                if mode=='fixed':assert row['last_train_batch_utility_weights']==[.5]*6")
edit("+.01*row['variance_floor']","+.01*row['variance_floor']+.025*row['pair_sentiment']+.01*row['utility_calibration']")
edit("'feedback_gates':h[-1]['feedback_gates']", "'last_dev_batch_utility_weights':h[-1]['last_train_batch_utility_weights']")
edit('INFLOW_CONDITIONS_RUN_COMPLETE','INFLOW_UTILITY_RUN_COMPLETE');edit('GATED_SELECTED_NODES','UTILITY_SELECTED_NODES')
edit("if mode=='none':assert np.max(np.abs(on-off))<1e-5", """if mode=='none':assert np.max(np.abs(on-off))<1e-5
                    for key,width in [('own',3),('pair',6),('utility',6),('predicted_weights',6)]:
                        assert saved[key].shape==(229,width) and np.isfinite(saved[key]).all()
                    u=saved['utility'];w=saved['predicted_weights']
                    assert np.max(np.abs(u))<=1 and w.min()>=0 and w.max()<=1
                    assert np.allclose(w,1/(1+np.exp(-4*u)),atol=1e-7,rtol=1e-7)""")
edit("'limits':'Exploratory single seed", "'limits':'last_train_batch_utility_weights raw key is actually last DEV batch after eval; full utility calibration analysis remains separate. Exploratory single seed")
ast.parse(s)
p=root/'audit_utility_snapshot_v1.py'
with p.open('x',encoding='utf-8',newline='\n') as f:f.write(s)
print(json.dumps({'auditor_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'status':'PREPARED_AST_ONLY_LIVE_ACTUAL_AUDIT_PENDING_COMPLETE_BRANCH_FUTURE'}))
