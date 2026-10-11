import ast,hashlib,json
from pathlib import Path
F=Path('outputs/parallel_gpu_20261003_1800');S=F/'source_train_shuffle_deployment_20261004T0016Z'
dep=F/'dual_source_deployment_20261004T0044Z';dep.mkdir(exist_ok=False)
old='assert not active,\'Independent GPU is occupied; do not share training GPU\''
gate="""assert authorization.get('shared_gpu_pair_authorized') is True and authorization['maximum_training_processes']==2
        if active:
            assert cli.stage=='run' and len(active.splitlines())==1,'Only the authorized one peer may be active'
            peer=authorization['peer']
            expected=[str(root/'.venv/bin/python'),'-u',peer['runner'],'--stage','run','--baseline-root',str(root),'--out',peer['attempt'],'--feedback','none','--rho','0','--guard','on','--epochs','100','--seed','128','--device','cuda','--parallel-authorization',peer['authorization']]
            import shlex
            rows=subprocess.check_output(['ps','-ww','-eo','pid,args'],text=True).splitlines()[1:]
            matches=[row for row in rows if shlex.split(row.split(None,1)[1])==expected]
            assert len(matches)==1,'GPU peer actual command must match the unique authorized assignment'
            peer_manifest=json.loads(Path(peer['manifest']).read_text())
            assert all(helpers_digest_early(Path(peer['runner']).parent/n)==h for n,h in peer_manifest.items())
        memory_free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        assert memory_free>=7500,'Insufficient free VRAM for unchanged batch32/dev128'
"""
helper="""def helpers_digest_early(path):
    import hashlib
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

"""
records={}
for case in ['full_null','text_one']:
    out=dep/case;out.mkdir()
    if case=='full_null':
        flow=(S/'source_view_flow.py').read_text(encoding='utf-8')
        runner=(S/'run_source_validation.py').read_text(encoding='utf-8')
        name='v8_matched_full_route_train_weak_source_permutation_control_v1'
        loss='full source route only: (final MSE + 0.1 first-phase MSE)/1.1; seven routes still computed and evaluated'
        flow=flow.replace("((final-labels.reshape(-1,1)).square().mean()+.1*(first-labels.reshape(-1,1)).square().mean())","((final[:,6]-labels.reshape(-1)).square().mean()+.1*(first[:,6]-labels.reshape(-1)).square().mean())")
        runner=runner.replace('v8_matched_seven_route_train_weak_source_permutation_control_v1',name).replace('equal average over seven source availability routes of (final MSE + 0.1 first-phase MSE)/1.1',loss)
    else:
        flow=(F/'source_view_deployment_20261003T2246Z/source_view_flow.py').read_text(encoding='utf-8')
        runner=(F/'source_full_control_deployment_20261003T2317Z/run_source_validation.py').read_text(encoding='utf-8')
        name='v8_matched_text_only_route_supervision_v1'
        loss='text source route code 1 only: (final MSE + 0.1 second-Euler-step MSE)/1.1; seven routes still computed and evaluated; natural full-route dev selection'
        flow=flow.replace("((final-labels.reshape(-1,1)).square().mean()+.1*(first-labels.reshape(-1,1)).square().mean())","((final[:,0]-labels.reshape(-1)).square().mean()+.1*(first[:,0]-labels.reshape(-1)).square().mean())")
        runner=runner.replace('v8_matched_full_route_only_supervision_control_v1',name).replace('full source route only: (final MSE + 0.1 first-phase MSE)/1.1; seven routes still computed and evaluated',loss)
        runner=runner.replace("(core.proj_a.weight,core.proj_v.weight,core.model.embeddings.word_embeddings.weight,core.own_flow.field.net[-1].weight)","(core.model.embeddings.word_embeddings.weight,core.own_flow.field.net[-1].weight)")
        runner=runner.replace("optimizer.step()","if cli.stage=='check':\n                for parameter in (core.proj_a.weight,core.proj_v.weight):\n                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()==0\n            optimizer.step()")
    assert old in runner
    runner=runner.replace(old,gate);runner=runner.replace('def evaluate(',helper+'def evaluate(')
    for filename,content in [('run_source_validation.py',runner),('source_view_flow.py',flow),('native_flow.py',(S/'native_flow.py').read_text(encoding='utf-8'))]:
        ast.parse(content);(out/filename).write_text(content,encoding='utf-8',newline='\n')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
    (out/'source_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    records[case]={'name':name,'task_loss':loss,'source_sha256':hashes,'root':'/data/coding/selective_flow/strong_baselines/own_flow_v8_identifiability/dual_source_20261004T0044Z/'+case}
for case in records:
    v=dep/case
    (v/'predecessor_local_verification.json').write_bytes(Path('D:/CodexBackups/selective_flow_20261003_1105/completed_refresh_20261004T0037Z/source_train_shuffle_seed128_4060Ti/local_checkpoint_verification.json').read_bytes())
    (v/'predecessor_diagnostic_verification.json').write_bytes((F/'source_train_shuffle_frozen_view_local_audit_20261004T0040Z.json').read_bytes())
(F/'dual_source_candidate_plan_20261004T0044Z.json').write_text(json.dumps({'purpose':'Complete missing comparisons: full-route-loss train AV permutation for the aligned-vs-null objective factorial, and pure text-source supervision under same seven-route compute/capacity to compare text-four and seven objectives. User explicitly authorized two concurrent runs on 4060Ti.','cases':records,'seed':128,'epochs':100,'train_batch':32,'dev_batch':128,'test_evaluated':False,'five_seed_promoted':False,'source_alignment_causal_claim':False,'hardware_limits':'Same physical GPU; concurrent execution is newly authorized and recorded. Do not equate trajectories or runtime to previous serial runs.'},indent=2),encoding='utf-8')
print(json.dumps(records))

