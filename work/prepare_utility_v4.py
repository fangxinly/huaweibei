from pathlib import Path
import ast,datetime,hashlib,json
root=Path(__file__).resolve().parent
old=root/'inflow_gated_v3_20261005T0241Z';out=root/'inflow_utility_v4_20261005T0346Z';out.mkdir(exist_ok=False)
(out/'utility_flow_model.py').write_bytes((root/'utility_flow_model_v1.py').read_bytes())
for name in ('legacy_flow_model.py','encoder_adapter.py'):(out/name).write_bytes((old/name).read_bytes())
s=(old/'run_inflow_conditions_v3.py').read_text(encoding='utf-8')
def edit(a,b):
 global s
 assert a in s,a
 s=s.replace(a,b)
edit('conditional_flow_model','utility_flow_model');edit('run_inflow_conditions_v3.py','run_inflow_utility_v4.py');edit('inflow_conditions_v3','inflow_utility_v4');edit('91812','91813')
edit('("none", "state", "task")','("none", "fixed", "predicted")')
edit('+ CONFIG["variance_weight"] * auxiliary["variance_floor"])','+ CONFIG["variance_weight"] * auxiliary["variance_floor"]\n            + CONFIG["pair_weight"] * auxiliary["pair_sentiment"]\n            + CONFIG["utility_weight"] * auxiliary["utility_calibration"])')
edit('targets["feedback_gate"] = core.own_flow.feedback_gate','targets["feedback_output"] = core.own_flow.donor_feedback[0][-1].weight')
edit('"visual_head": core.own_flow.unimodal_heads[2].weight}', '"visual_head": core.own_flow.unimodal_heads[2].weight,\n                   "pair_head":core.own_flow.pair_heads[0][-1].weight,\n                   "utility_head":core.own_flow.utility_heads[0][-1].weight}')
edit('learned_feedback_gradient = core.own_flow.feedback[0][0].weight.grad','learned_feedback_gradient = core.own_flow.donor_feedback[0][1].weight.grad')
edit('"feedback_gates_after_check": torch.tanh(core.own_flow.feedback_gate).detach().cpu().tolist(),','"utility_weights_after_check":core.last_trace["utility_weights"].mean(0).cpu().tolist(),')
edit('"feedback_gates": torch.tanh(core.own_flow.feedback_gate).detach().cpu().tolist()', '"last_train_batch_utility_weights":core.last_trace["utility_weights"].mean(0).cpu().tolist()')
edit('"condition": "After first Euler stage, zero / ordinary 4D state / four scaled task predictions tiled to 4D; detached inputs, original context projection",','"condition": "After first Euler stage: none / fixed .5 / predicted sigmoid(4u), six directed detached-state donor feedback channels; zero feedback output init",')
edit('"feedback_gate_config": "Three signed tanh scalar gates, init zero; no extra loss, same optimizer/weight decay, optimization hypothesis not sample utility or semantic truth",','"utility_config": "Six tanh utility heads target detach(tanh(((p_m-y)^2-(p_mj-y)^2)/.5)); smoothL1 weight.01; pair MSE .025; every mode same auxiliary losses; labels absent from inference utility inputs",')
edit('"Same constructed parameters and order, but state and scalar conditions have different effective information capacity"','"Same directed feedback/head architecture and objectives; only conditioning weights differ"')
edit('        initial_zero_gate_equality_verified = True','''        initial_zero_gate_equality_verified = True
        model.eval()
        with torch.no_grad():
            small_equal=tuple(x[:4] for x in equal_batch)
            neutral=list(small_equal);neutral[3]=torch.zeros_like(neutral[3])
            base=forward_batch(model,tuple(neutral))[0]
            weights=core.last_trace['utility_weights'].clone()
            neutral[3]=torch.ones_like(neutral[3])*123.
            changed=forward_batch(model,tuple(neutral))[0]
            assert torch.equal(base,changed) and torch.equal(weights,core.last_trace['utility_weights'])
        label_isolation_verified=True''')
edit('        loss.backward()\n        targets =', '''        utility_grad=torch.autograd.grad(auxiliary['utility_calibration'],
            [core.own_flow.utility_heads[0][-1].weight,core.own_flow.pair_heads[0][-1].weight,core.proj_a.weight],
            retain_graph=True,allow_unused=True)
        assert utility_grad[0] is not None and utility_grad[0].abs().sum()>0
        assert utility_grad[1] is None and utility_grad[2] is None
        loss.backward()
        targets =''')
edit('"initial_zero_gate_equality_verified": initial_zero_gate_equality_verified,','"initial_zero_feedback_equality_verified": initial_zero_gate_equality_verified,\n                  "label_isolation_verified":label_isolation_verified,\n                  "utility_gradient_detach_verified":True,\n                  "feedback_output_gradient_first_backward_verified":cli.mode!="none",')
edit('        # First scheduled step has lr0. Second step learns a small gate; the\n        # third backward must reach the feedback projection through that gate.','        # First scheduled step has lr0. Second learns the zero feedback output;\n        # third backward must reach the hidden feedback projection.')
edit('INFLOW_CONDITIONS_','INFLOW_UTILITY_')
edit('    np.savez_compressed(cli.out / "predictions.npz", valid_pred=vp, valid_y=vy, condition_off_pred=offp)', '''    extra={k:[] for k in ('own','pair','utility','predicted_weights')}
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            batch=tuple(x.to(author.DEVICE) for x in batch)
            neutral=(*batch[:3],torch.zeros_like(batch[3]),batch[4])
            forward_batch(model,neutral)
            for name in extra:extra[name].append(core.own_flow._stage1[name].cpu().numpy())
    extra={k:np.concatenate(v) for k,v in extra.items()}
    np.savez_compressed(cli.out / "predictions.npz", valid_pred=vp, valid_y=vy, condition_off_pred=offp,**extra)''')
ast.parse(s)
(out/'run_inflow_utility_v4.py').write_text(s,encoding='utf-8',newline='\n')
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.py')}
for p in out.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
(out/'source_manifest.json').write_text(json.dumps({'source_sha256':sources},indent=2),encoding='utf-8')
plan={'prepared_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'AST_ONLY_REAL_GPU_CHECK_REQUIRED_BEFORE_TRAINING','question':'Does directed sample utility supervised by task-head error reduction improve feedback beyond fixed weights?','seed':91813,'epochs':100,'updates_per_epoch':40,'total_updates':4000,'train_samples':1281,'dev_samples':229,'TEST_forbidden':True,'modes':['none','fixed','predicted'],'init':'Same full parameters, six pair heads, six zero utility outputs, six zero feedback outputs; initial predictions identical','objective':'.02FM+.01cycle+.05unimodal(split half source/stage1)+.01variance+.025pairMSE+.01utilitySmoothL1, same all modes','utility_target':'detach(tanh(((p_m-y)^2-(p_mj-y)^2)/.5)), TRAIN only, not semantic ground truth','utility_weight':'sigmoid(4u); fixed .5; none0; average two directed donors per receiver then .5 context update','optimizer':'author groups lr1e-5 linear warmup10%, same encoder pretrained only, no old supervised checkpoint','check':'real first feedback-output/pair/utility gradients; utility-only gradient isolated from pair heads and encoder; after warmup hidden feedback gradient; exact same-model zero-feedback predictions across modes; labels mutation and batch/mask invariance; cross-node full initial SHA and100orders exact','limitations':['Single exploratory seed/node binding, no stable five-seed or semantic evidence claims','Task-head utility is endogenous proxy; poor pair teachers may make targets misleading','New objective/seed confound comparison to old runs; matched new none/fixed controls needed'],'source_sha256':sources}
(out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(plan,indent=2))
