"""Prepare complete-prefix continuation and matched CandidateTail audit, not execute."""
import argparse,ast,hashlib,json,shutil,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]

def run(tag):
 source=base/'work/polarity_intensity_prefix_v2_20261008T203225Z';root=base/'work'/('polarity_intensity_resume_preparation_'+tag);root.mkdir()
 for f in source.iterdir():
  if f.suffix in ('.py','.npy'):shutil.copy2(f,root/f.name)
 s=(base/'work/resume_official_mse100_v1.py').read_text(encoding='utf8')
 s=s.replace('from official_upgrade import OfficialUpgrade','from candidate_adapter import make_candidate,optimizer_groups,CandidateTail\nfrom polarity_intensity_flow import train_objective')
 s=s.replace("p['prefix16_Release_B_CPU_qualified']", "p['prefix16_Release_othernode_CPU_qualified']")
 s=s.replace("assert p['task_loss']=='mse'", "assert p['task_loss']=='huber1_polarity_intensity_aux' and p['candidate_mode'] in ('regression_aux','factorized_aux')")
 s=s.replace('model.dberta.own_flow=OfficialUpgrade()',"model.dberta.own_flow=make_candidate(p['candidate_mode'])")
 begin=s.index('    gain=model.dberta.own_flow.gain;');end=s.index('    opt=torch.optim.AdamW(groups);',begin)
 s=s[:begin]+'    groups,named=optimizer_groups(model)\n'+s[end:]
 s=s.replace("assert pm['steps']==16", "assert pm['candidate_mode']==p['candidate_mode'] and pm['task_loss']==p['task_loss']\n    assert pm['steps']==16")
 s=s.replace("pred=forward_batch(model,batch);task=torch.nn.functional.mse_loss(pred,batch[3].view(-1));penalty=.01*model.dberta.own_flow.last_context.square().mean();loss=task+penalty", "pred=forward_batch(model,batch);loss,details=train_objective(model.dberta.own_flow,batch[3].view(-1),'TRAIN');penalty=.01*model.dberta.own_flow.last_context.square().mean();task=loss-penalty")
 s=s.replace('model.dberta.own_flow.gain.detach()', 'model.dberta.own_flow.core.gain.detach()')
 s=s.replace('flow_minus_base_RMS=float((model.dberta.own_flow.last_flow.detach()-model.dberta.own_flow.last_base.detach()).square().mean().sqrt())', "auxiliary_loss_components={k:float(v.detach()) for k,v in details.items()},magnitude_RMS=float(model.dberta.own_flow.last['magnitude'].detach().square().mean().sqrt())")
 s=s.replace("metadata=dict(plan_SHA=a.plan_sha,steps=400", "metadata=dict(plan_SHA=a.plan_sha,candidate_mode=p['candidate_mode'],task_loss=p['task_loss'],steps=400")
 s=s.replace("metadata=dict(prefix16_checkpoint_SHA=", "metadata=dict(candidate_mode=p['candidate_mode'],candidate_sources={n:p['source_sha256'][n] for n in ('candidate_adapter.py','polarity_intensity_flow.py','controlled_flow.py')},prefix16_checkpoint_SHA=")
 s=s.replace('tail=OriginalTail(model.dberta)', 'tail=CandidateTail(model.dberta)')
 s=s.replace("torch.save({'state':tail.state_dict()}", "torch.save({'mode':p['candidate_mode'],'state':tail.state_dict()}")
 s=s.replace('model.dberta.own_flow.message.state_dict()', 'model.dberta.own_flow.core.message.state_dict()')
 s=s.replace('OFFICIAL_ORIGINAL_FLOW_MSE_HEALTH_CANDIDATE_TRAIN100_COMPLETE','OFFICIAL_ORIGINAL_FLOW_POLARITY_INTENSITY_TRAIN100_COMPLETE')
 (root/'resume_official_mse100_v1.py').write_text(s,encoding='utf8')
 # Existing inference correctly reconstructs CandidateTail; replace stale D-only certificate.
 s=(base/'work/polarity_intensity_official_pipeline_preparation/infer_official.py').read_text(encoding='utf8')
 s=s.replace("assert p['training_D_B_CPU_gate'] and p['candidate_training_D_B_CPU_gate']", "assert p['training_Release_othernode_CPU_gate']")
 s=s.replace("full['metadata']['candidate_mode']==p['candidate_mode']", "full['metadata']['candidate_mode']==p['candidate_mode'] and full['metadata']['steps']==4000")
 s=s.replace("if i==0:\n", "if i==0:\n")
 s=s.replace("TRAIN_VAL_TEST_no_training_overlap=True", "TRAIN_VAL_TEST_no_training_overlap=True,candidate_mode=p['candidate_mode'],peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())")
 (root/'infer_official.py').write_text(s,encoding='utf8')
 # Future CSV writes exact binary32 values as Python floats; frozen prior scores untouched.
 s=(base/'work/official_mse_posttrain_tools_v1/score_official.py').read_text(encoding='utf8')
 s=s.replace("y[i],pred[name+'_prediction'][i],pred[name+'_p0'][i]", "float(y[i]),float(pred[name+'_prediction'][i]),float(pred[name+'_p0'][i])")
 s=s.replace("historical_TEST_access_disclosed=True", "candidate_mode=p['candidate_mode'],historical_TEST_access_disclosed=True")
 (root/'score_official.py').write_text(s,encoding='utf8')
 # Shared full-Adam/history audit, CandidateTail mappings and CPU ON replay.
 s=(base/'work/official_mse_posttrain_tools_v1/audit_official.py').read_text(encoding='utf8')
 s=s.replace('from incremental_message import OriginalTail,IncrementalMessage', 'from candidate_adapter import CandidateTail,make_candidate\nfrom types import SimpleNamespace\nfrom torch import nn')
 begin=s.index("    cache=dict(np.load(root/'out/selected_DEV_original_source.npz'");end=s.index("    write(a.out/'audit_result.json'",begin)
 s=s[:begin]+'''    assert m['candidate_mode']==p['candidate_mode']
    cache=dict(np.load(root/'out/selected_DEV_original_source.npz',allow_pickle=False))
    core=SimpleNamespace(own_flow=make_candidate(m['candidate_mode']),fusion=nn.Sequential(nn.Linear(300,150),nn.ReLU(),nn.Linear(150,100)),predictor=nn.Sequential(nn.Linear(100,150),nn.ReLU(),nn.Linear(150,1)))
    tail=CandidateTail(core);export=torch.load(root/'out/selected_original_tail.pt',map_location='cpu');assert export['mode']==m['candidate_mode'];tail.load_state_dict(export['state'],strict=True)
    selected=state['selected_model']
    for n,v in tail.state_dict().items():
        key='dberta.own_flow.'+n[len('flow.'):] if n.startswith('flow.') else 'dberta.'+n
        assert torch.equal(v,selected[key]),n
    expected={k.removeprefix('dberta.own_flow.core.message.'):v for k,v in selected.items() if k.startswith('dberta.own_flow.core.message.')}
    assert tensor_sha(expected)==tensor_sha(torch.load(root/'out/selected_message.pt',map_location='cpu')['state'])
    pieces=[];before=tensor_sha(tail.state_dict());rng=torch.get_rng_state().clone()
    with torch.no_grad():
        for i in range(0,229,32):
            src=torch.from_numpy(cache['source'][i:i+32]);mask=torch.from_numpy(cache['mask'][i:i+32]);on,off=tail(src,mask);assert torch.isfinite(on).all() and torch.isfinite(off).all();pieces.append(on.numpy())
    err=float(np.max(abs(np.concatenate(pieces)-cache['prediction'])));assert err<=1e-4
    assert before==tensor_sha(tail.state_dict()) and torch.equal(rng,torch.get_rng_state())
'''+s[end:]
 s=s.replace("CPU_encoder_forward=False", "candidate_mode=m['candidate_mode'],CandidateTail_all_selected_parameters_exact=True,CPU_encoder_forward=False")
 (root/'audit_official.py').write_text(s,encoding='utf8')
 # Genuine resume qualification must use candidate optimizer and fixed auxiliary losses.
 s=(base/'work/qualify_resume_official_v1.py').read_text(encoding='utf8')
 s=s.replace('from official_upgrade import OfficialUpgrade', 'from candidate_adapter import make_candidate,optimizer_groups\nfrom polarity_intensity_flow import train_objective')
 s=s.replace("self.dberta.own_flow=OfficialUpgrade()", "self.dberta.own_flow=make_candidate(mode)")
 begin=s.index(' special={id(');end=s.index(' o=torch.optim.AdamW(groups);',begin)
 s=s[:begin]+' groups,_=optimizer_groups(m)\n'+s[end:]
 s=s.replace("loss=(pred-y).square().mean()+.01*m.dberta.own_flow.last_context.square().mean()", "loss=train_objective(m.dberta.own_flow,y,'TRAIN')[0]")
 # Serialize and compare each mode independently with identical original init RNG.
 start=s.index('random.seed(128)');end=s.index("sha=lambda p:")
 # Definitions and body are mixed in old script; wrap all within each mode loop.
 body=s[start:end].replace("a.out.mkdir(exist_ok=False);path=a.out/'toy_prefix.pt'", "path=a.out/('toy_prefix_'+mode+'.pt')")
 s=s[:start]+"a.out.mkdir(exist_ok=False)\nfor mode in ('regression_aux','factorized_aux'):\n"+'\n'.join(' '+line if line else '' for line in body.splitlines())+'\n'+s[end:]
 s=s.replace("'OFFICIAL_FLOW_PREFIX16_RESUME_TO40_CPU_TRAJECTORY_IDENTICAL'", "'POLARITY_INTENSITY_BOTH_MODES_PREFIX16_RESUME_TO40_CPU_TRAJECTORY_IDENTICAL'")
 s=s.replace('prefix_counted_once=True', "qualified_modes=['regression_aux','factorized_aux'],prefix_counted_once=True")
 (root/'qualify_resume_official_v1.py').write_text(s,encoding='utf8')
 wrapper=(base/'work/run_resumed100_capture_v1.py').read_text(encoding='utf8')
 wrapper=wrapper.replace("p['prefix16_Release_B_CPU_qualified']", "p['prefix16_Release_othernode_CPU_qualified']")
 wrapper=wrapper.replace("bundle/'B_prefix16_audit_result.json'", "bundle/'prefix16_original_othernode_CPU_result.json'")
 wrapper=wrapper.replace("'B_ORIGINAL_PREFIX16_COMPLETE_MODEL_ADAM_RNG_ORDER_SHA_CRC_PASSED'", "'CANDIDATE_OTHER_NODE_PREFIX16_COMPLETE_MODEL_ADAM_RNG_ORDER_CPU_ON_OFF_PASSED'")
 wrapper=wrapper.replace("p['GPU_UUID']['A']", "p['GPU_UUID'][p['execution_node']]")
 (root/'run_resumed100_capture_v1.py').write_text(wrapper,encoding='utf8')
 for f in root.glob('*.py'):ast.parse(f.read_text(encoding='utf8'))
 p=json.loads((source/'native_preparation_plan.json').read_text(encoding='utf8'));p.update(status='CANDIDATE_RESUME_PREPARATION_NO_REAL_RESUME_QUALIFIED_YET',source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.iterdir() if f.suffix in ('.py','.npy')},resume_native_trajectory_qualified=False,prefix16_Release_othernode_CPU_qualified=False)
 (root/'resume_preparation_plan.json').write_text(json.dumps(p,indent=2),encoding='utf8')
 with zipfile.ZipFile(root/'resume_source.zip','x',zipfile.ZIP_DEFLATED) as z:
  for f in root.iterdir():
   if f.suffix!='.zip':z.write(f,f.name)
 print(json.dumps(dict(root=str(root),resume_qualification=False,real100_started=False)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);run(p.parse_args().tag)
