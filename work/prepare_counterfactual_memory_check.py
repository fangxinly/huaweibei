from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
old=r/'inflow_counterfactual_v5_checks_20261005T0502Z';new=r/'inflow_counterfactual_v5_checks_20261005T0516Z'
new.mkdir(exist_ok=False)
for name in ['counterfactual_flow_model.py','encoder_adapter.py','legacy_flow_model.py']:(new/name).write_bytes((old/name).read_bytes())
s=(old/'check_counterfactual_v5.py').read_text()
needle="    assert all(g is None for g in initial_gradient[1:])\n"
assert s.count(needle)==1
s=s.replace(needle,needle+"    # Consume the retained probe graph before the first independent update.\n    objective(pred,batch[3],core.last_losses).backward()\n    optimizer.zero_grad(set_to_none=True)\n    del pred,stage,initial_gradient\n    torch.cuda.empty_cache()\n")
(new/'check_counterfactual_v5.py').write_text(s)
for p in new.glob('*.py'):ast.parse(p.read_text())
manifest={'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(new.glob('*.py'))}}
(new/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
plan=json.loads((old/'check_plan.json').read_text());plan['revision']='Probe graph consumed with backward but no optimizer update; no model/target/order change';plan['prior_check_root']=old.name
(new/'check_plan.json').write_text(json.dumps(plan,indent=2))
s=(r/'launch_counterfactual_checks_v1.py').read_text().replace('20261005T0502Z','20261005T0516Z')
(new/'launch_counterfactual_checks_v2.py').write_text(s)
s=(r/'capture_counterfactual_followup_v1.py').read_text().replace("'inflow_counterfactual_v5_checks_20261005T0502Z')]","'inflow_counterfactual_v5_checks_20261005T0502Z',\n 'inflow_counterfactual_v5_checks_20261005T0516Z')]")
(r/'capture_counterfactual_followup_v2.py').write_text(s)
print(json.dumps(manifest))
