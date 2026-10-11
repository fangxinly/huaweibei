from pathlib import Path
import ast,datetime,hashlib,json
OLD=Path('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen/outputs/own_flow_v6_followup/inflow_conditions_v2_20261004T2140Z')
OUT=Path('work/inflow_gated_v3_20261005T0241Z')
OUT.mkdir(exist_ok=False)
def edit(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)
model=(OLD/'conditional_flow_model.py').read_text(encoding='utf-8').split('\ndef check_synthetic_cpu():')[0]
model=edit(model,'        self.context_override = None','        self.context_override = None\n        self.feedback_gate = torch.nn.Parameter(torch.zeros(3))')
model=edit(model,'        return .5 * old + .5 * selected','        selected = selected * torch.tanh(self.feedback_gate)[None, :, None]\n        return .5 * old + .5 * selected')
model='"""Zero-initialized signed scalar gates per receiver, trained jointly from scratch.\n\nThis is an optimization control, not a calibrated semantic/conflict decision.\n"""\n'+model[model.index('import torch'):]
(OUT/'conditional_flow_model.py').write_text(model,encoding='utf-8',newline='\n')
runner=(OLD/'run_inflow_conditions_v2.py').read_text(encoding='utf-8')
runner=runner.replace('run_inflow_conditions_v2.py','run_inflow_conditions_v3.py').replace('inflow_conditions_v2','inflow_conditions_v3').replace('91811','91812')
runner=edit(runner,'    if cli.stage == "check":\n        batch', '''    if cli.stage == "check":
        model.eval()
        with torch.no_grad():
            equal_batch = tuple(x.to(author.DEVICE) for x in next(iter(valid_loader)))
            originals = []
            for equal_mode in ("none", "state", "task"):
                core.own_flow.mode = equal_mode
                originals.append(forward_batch(model, equal_batch)[0])
                assert float(core.last_trace["context_norm"]) == 0.0
            assert all(torch.equal(originals[0], p) for p in originals[1:])
            core.own_flow.mode = cli.mode
        initial_zero_gate_equality_verified = True
        batch''')
runner=edit(runner,'            targets["feedback"] = core.own_flow.feedback[0][0].weight','            targets["feedback_gate"] = core.own_flow.feedback_gate')
runner=edit(runner,'        assert not torch.equal(before, watched)\n        optimizer.zero_grad(set_to_none=True)', '''        optimizer.zero_grad(set_to_none=True)
        # First scheduled step has lr0. Second step learns a small gate; the
        # third backward must reach the feedback projection through that gate.
        prediction, _, _ = author._forward_eval(model, batch)
        loss = objective(prediction, batch[3], core.last_losses)
        loss.backward()
        learned_feedback_gradient = None
        if cli.mode != "none":
            learned_feedback_gradient = core.own_flow.feedback[0][0].weight.grad
            assert learned_feedback_gradient is not None
            assert torch.isfinite(learned_feedback_gradient).all() and learned_feedback_gradient.abs().sum() > 0
            learned_feedback_gradient = float(learned_feedback_gradient.abs().sum())
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        assert not torch.equal(before, watched)''')
runner=edit(runner,'"optimizer_change_verified": True, "warmup_check_steps": 2,','"optimizer_change_verified": True, "warmup_check_steps": 3,\n                  "initial_zero_gate_equality_verified": initial_zero_gate_equality_verified,\n                  "learned_feedback_gradient_sum": learned_feedback_gradient,\n                  "feedback_gates_after_check": torch.tanh(core.own_flow.feedback_gate).detach().cpu().tolist(),')
runner=edit(runner,'"peak_gpu_bytes": torch.cuda.max_memory_allocated()}\n        history.append(row)','"peak_gpu_bytes": torch.cuda.max_memory_allocated(),\n               "feedback_gates": torch.tanh(core.own_flow.feedback_gate).detach().cpu().tolist()}\n        history.append(row)')
runner=edit(runner,'"observer_auxiliary": "same .05 total coefficient:', '"feedback_gate_config": "Three signed tanh scalar gates, init zero; no extra loss, same optimizer/weight decay, optimization hypothesis not sample utility or semantic truth",\n                "observer_auxiliary": "same .05 total coefficient:')
(OUT/'run_inflow_conditions_v3.py').write_text(runner,encoding='utf-8',newline='\n')
for name in ('legacy_flow_model.py','encoder_adapter.py'):(OUT/name).write_bytes((OLD/name).read_bytes())
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.py')}
for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
(OUT/'source_manifest.json').write_text(json.dumps({'source_sha256':sources},indent=2),encoding='utf-8')
plan={'prepared_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'SOURCE_PREPARED_AST_ONLY_GPU_CHECK_AND_MATCHED_INITIAL_SHA_REQUIRED',
    'question':'Can zero-initialized trainable gates avoid premature conditional perturbations and support useful ordinary state/task conditioning?',
    'seed':91812,'epochs':100,'train_samples':1281,'dev_samples':229,'TEST_forbidden':True,
    'modes':['none','state','task'],'same_constructed_parameters':True,'optimizer':'same author groups lr1e-5 warmup10%',
    'batch_order':'100 precomputed 1280 sample orders, 40updates/epoch, same in three modes',
    'weights':'fresh encoder pretrained only, no old trained checkpoint initialization',
    'gate':'three signed tanh scalar gates, initial0, may suppress or reverse conditioning; no independent semantic guarantee',
    'check':'initial same-model predictions equal across modes at zero gates; gate gradient first backward; feedback gradient after warmup; real three updates; masking/batch checks; three initial state SHAs and 100order arrays exactly equal',
    'limitations':['Single new exploratory seed with mode/node binding; not a five-seed confirmation.',
        'Difference from v2 confounds gate mechanism and seed; compare only this new matched none student for main interpretation.',
        'Global scalar gate is an optimization control, not sample-specific shared/complementary/conflict classification.'],
    'source_sha256':sources}
(OUT/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
print(json.dumps(plan,indent=2))
