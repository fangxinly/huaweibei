"""Adapt the qualified complete-prefix runner to the previously approved pair."""
import argparse,ast,hashlib,json,shutil,zipfile
from pathlib import Path

def run(tag,stamp):
    base=Path(__file__).resolve().parents[1];root=base/'work'/('polarity_intensity_prefix_v2_'+tag);root.mkdir()
    original=base/'work/autonomous_mse100_resume_qualified_20261008T182923Z'
    candidate=base/'work/polarity_intensity_official_pipeline_preparation'
    for file in original.iterdir():
        if file.suffix in ('.py','.npy'):shutil.copy2(file,root/file.name)
    for n in ('candidate_adapter.py','polarity_intensity_flow.py','controlled_flow.py'):
        shutil.copy2(candidate/n,root/n)
    # Naming only; integer-safe sample and optimizer instrumentation stay intact.
    f=root/'training_health_v1.py';s=f.read_text(encoding='utf8');s=s.replace("def group_name(name):\n", "def group_name(name):\n    name=name.replace('.own_flow.core.','.own_flow.')\n    for head in ('polarity_delta','magnitude_delta'):\n        if '.own_flow.'+head+'.' in name:return head\n");f.write_text(s,encoding='utf8')
    f=root/'precheck16.py';s=f.read_text(encoding='utf8')
    s=s.replace('from official_upgrade import OfficialUpgrade','from candidate_adapter import make_candidate,optimizer_groups,CandidateTail\nfrom polarity_intensity_flow import train_objective')
    s=s.replace("assert p['task_loss']=='mse'", "assert p['task_loss']=='huber1_polarity_intensity_aux' and p['candidate_mode'] in ('regression_aux','factorized_aux')")
    s=s.replace('model.dberta.own_flow=OfficialUpgrade()',"model.dberta.own_flow=make_candidate(p['candidate_mode'])")
    begin=s.index('    gain=model.dberta.own_flow.gain;');end=s.index('    opt=torch.optim.AdamW(groups);',begin)
    s=s[:begin]+'    groups,named=optimizer_groups(model)\n'+s[end:]
    s=s.replace("task=F.mse_loss(pred,batch[3].view(-1));penalty=.01*model.dberta.own_flow.last_context.square().mean()", "loss,details=train_objective(model.dberta.own_flow,batch[3].view(-1),'TRAIN');penalty=.01*model.dberta.own_flow.last_context.square().mean();task=loss-penalty")
    s=s.replace('model.dberta.own_flow.gain.detach()', 'model.dberta.own_flow.core.gain.detach()')
    s=s.replace('flow_minus_base_RMS=float((model.dberta.own_flow.last_flow.detach()-model.dberta.own_flow.last_base.detach()).square().mean().sqrt())',"auxiliary_loss_components=details, magnitude_RMS=float(model.dberta.own_flow.last['magnitude'].detach().square().mean().sqrt())")
    begin=s.index('    flow_module=model.dberta.own_flow;message=');end=s.index('    assert np.array_equal(on,dummy)',begin)
    s=s[:begin]+'''    flow_module=model.dberta.own_flow
    with torch.no_grad():
        on=forward_batch(model,probe).cpu().numpy();off=flow_module.last_off['prediction'].cpu().numpy()
        base=flow_module.last['regression'].cpu().numpy();magnitude=flow_module.last['magnitude'].cpu().numpy()
        probe[3].fill_(7);dummy=forward_batch(model,probe).cpu().numpy()
'''+s[end:]
    s=s.replace('base=base,flow=flow,labels_not_used=True', 'original_regression=base,magnitude=magnitude,labels_not_used=True')
    s=s.replace("task_loss='mse',steps=16", "task_loss=p['task_loss'],candidate_mode=p['candidate_mode'],steps=16")
    s=s.replace("REAL_TRAIN_MSE_PREFIX16_COMPLETE_NOT_FULL_TRAIN", "REAL_TRAIN_POLARITY_INTENSITY_PREFIX16_COMPLETE_NOT_FULL_TRAIN")
    # Export an actual small source/tail for independent CPU replay of both outputs.
    needle="    metadata=dict(plan_SHA="
    at=s.index(needle)
    s=s[:at]+'''    with torch.no_grad():
        source,source_mask=source_features(model.dberta,probe)
        actual_tail=CandidateTail(model.dberta)
        tail_on,tail_off=actual_tail(source.cpu(),source_mask.cpu())
    maxerror=float(np.max(abs(tail_on.numpy()-on)))
    assert maxerror<1e-4 and float(np.max(abs(tail_off.numpy()-off)))<1e-4
    np.savez(a.out/'prefix16_source.npz',source=source.cpu().numpy(),mask=source_mask.cpu().numpy(),on=on,off=off)
    torch.save(dict(mode=p['candidate_mode'],state=actual_tail.state_dict()),a.out/'prefix16_tail.pt')
'''+s[at:]
    # CandidateTail constructors preserve RNG; CPU forward must not mutate state.
    s=s.replace("cp=a.out/'complete_resume_step16.pt'", "assert tensor_sha(model.state_dict())==saved_sha and torch.equal(saved_rng['torch'],torch.get_rng_state())\n    cp=a.out/'complete_resume_step16.pt'")
    f.write_text(s,encoding='utf8')
    wrapper=(base/'work/autonomous_mse16_healthfix_20261008T181431Z/run_prefix16_capture_v1.py').read_text(encoding='utf8')
    wrapper=wrapper.replace("uuid==p['GPU_UUID']['A']", "uuid==p['GPU_UUID'][p['execution_node']]")
    (root/'run_prefix16_capture_v1.py').write_text(wrapper,encoding='utf8')
    native=(original/'qualify_health_components_v2.py').read_text(encoding='utf8')
    begin=native.index('class Model(nn.Module):');end=native.index('torch.set_num_threads(2)',begin)
    native=native[:begin]+'''from candidate_adapter import make_candidate,optimizer_groups,CandidateTail
from polarity_intensity_flow import train_objective
class Model(nn.Module):
 def __init__(self):
  super().__init__();self.dberta=nn.Module();self.dberta.own_flow=make_candidate('factorized_aux');self.dberta.fusion=nn.Linear(300,100);self.dberta.predictor=nn.Linear(100,1)
 def forward(self,x,mask):return self.dberta.own_flow(x,mask,lambda h:self.dberta.predictor(self.dberta.fusion(h)))[0]

'''+native[end:]
    native=native.replace("groups=[dict(params=[p for n,p in m.named_parameters() if p.requires_grad and not n.endswith('gain')],lr=1e-5,weight_decay=.01),dict(params=[m.dberta.own_flow.gain],lr=.001,weight_decay=0.)]", "groups,_=optimizer_groups(m)")
    native=native.replace('tl=F.mse_loss(pl,y);rl=.01*left.dberta.own_flow.last_context.square().mean()',"loss,_=train_objective(left.dberta.own_flow,y,'TRAIN');rl=.01*left.dberta.own_flow.last_context.square().mean();tl=loss-rl")
    native=native.replace('(F.mse_loss(pr,y)+.01*right.dberta.own_flow.last_context.square().mean()).backward()',"train_objective(right.dberta.own_flow,y,'TRAIN')[0].backward()")
    # Qualify both same-capacity modes and exact tail serialization after real synthetic updates.
    at=native.index('a.out.mkdir(exist_ok=False)')
    native=native[:at]+'''for mode in ('regression_aux','factorized_aux'):
 model=Model();model.dberta.own_flow=make_candidate(mode);model.train();optimizer,_=opt(model)
 for step in range(3):
  optimizer.zero_grad(set_to_none=True);prediction=model(x,mask);train_objective(model.dberta.own_flow,y,'TRAIN')[0].backward()
  assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad);optimizer.step()
 model.eval();tail=CandidateTail(model.dberta);saved=tensor_sha(model.state_dict());rr=torch.get_rng_state().clone()
 with torch.no_grad():
  prediction=model(x,mask);on,off=tail(x,mask)
 assert torch.equal(prediction,on) and torch.equal(off,model.dberta.own_flow.last_off['prediction']) and saved==tensor_sha(model.state_dict()) and torch.equal(rr,torch.get_rng_state())
'''+native[at:]
    native=native.replace("'HEALTH_V2_NATIVE_CPU_SYNTHETIC_TRAJECTORY_IDENTICAL'", "'POLARITY_INTENSITY_HEALTH_NATIVE_CPU_SYNTHETIC_TRAJECTORY_IDENTICAL'")
    (root/'qualify_health_components_v2.py').write_text(native,encoding='utf8')
    p=json.loads((base/'work/autonomous_mse16_healthfix_20261008T181431Z/qualified_precheck_plan.json').read_text(encoding='utf8'))
    p.update(status='POLARITY_INTENSITY_PREFIX_PREPARATION_NOT_NATIVE_QUALIFIED',prepared_actualclock_UTC=stamp,task_loss='huber1_polarity_intensity_aux',candidate_mode='factorized_aux',execution_node='A',health_v2_native_qualified=False,continuation_runner_qualified=False,predeclared_matched_arms=['regression_aux','factorized_aux'],fixed_loss_coefficients=dict(task_huber1=1,abs_y_huber1=.1,weighted_sign_BCE=.1,context_square=.01),decision_basis='Previously approved original-flow polarity/intensity structure and TRAIN/VAL weakness; no TEST-based structure selection',source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir() if f.suffix in ('.py','.npy')})
    p.pop('qualification_sha256',None);(root/'preparation_plan.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf8')
    for f in root.glob('*.py'):ast.parse(f.read_text(encoding='utf8'))
    with zipfile.ZipFile(root/'preparation_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in root.iterdir():
            if f.suffix!='.zip':z.write(f,f.name)
    print(json.dumps(dict(root=str(root),bytes=(root/'preparation_source.zip').stat().st_size,actual_training=False,native_qualification=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--stamp',required=True);a=p.parse_args();run(a.tag,a.stamp)
