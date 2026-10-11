"""New MSE health prefix; frozen originals unchanged; no execution here."""
import ast,hashlib,json,shutil,sys,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]
tag,stamp=sys.argv[1:]
root=base/'work'/('autonomous_mse16_'+tag);root.mkdir(exist_ok=False)
src=base/'work/official_flow_health_candidate_v1'
for p in src.iterdir():
    if p.suffix in ('.py','.npy'):shutil.copyfile(p,root/p.name)
shutil.copyfile(base/'work/training_health_components_v2.py',root/'training_health_components_v2.py')
old=json.loads((base/'work/official_anchored_upgrade_20261008T053146Z/official_native_protocol.json').read_text(encoding='utf8'))
text=(root/'train_official.py').read_text(encoding='utf8')
text=text[:text.index('    history=[];best=')]
text=text.replace('from incremental_message import OriginalTail','from incremental_message import OriginalTail\nfrom training_health_components_v2 import component_gradients,before_optimizer_step,after_optimizer_step')
text=text.replace("assert p['native_CPU_D_qualification'] and p['new_candidate_full_encoder_precheck_D_B']", "assert p['precheck_steps']==16 and p['storage_transport_qualified'] and p['health_v2_native_qualified']")
text=text.replace("    def predict(dummy=0):", "    def unused_DEV_predict(dummy=0):")
text+= '''    # A fixed TRAIN probe with no label use; same state/gain, functional message OFF.
    from torch.nn import functional as F
    steps=0; records=[];batches,omitted=train_batches([int(x) for x in order[0]])
    for rows in batches[:16]:
        model.train();opt.zero_grad(set_to_none=True);batch=[v[list(rows)].cuda() for v in train.tensors]
        torch.cuda.synchronize();tick=time.monotonic();pred=forward_batch(model,batch)
        task=F.mse_loss(pred,batch[3].view(-1));penalty=.01*model.dberta.own_flow.last_context.square().mean()
        pieces=component_gradients(model,task,penalty);loss=task+penalty
        assert torch.isfinite(loss);loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for _,v in named)
        record=dict(actual_UTC=utc(),step=steps+1,rows=list(rows),task_loss=float(task.detach()),context_penalty=float(penalty.detach()),component_gradients=pieces,total_gradients=gradient_summary(model),learning_rates=[g['lr'] for g in opt.param_groups],gain=float(model.dberta.own_flow.gain.detach().tanh()),context_RMS=float(model.dberta.own_flow.last_context.detach().square().mean().sqrt()),flow_minus_base_RMS=float((model.dberta.own_flow.last_flow.detach()-model.dberta.own_flow.last_base.detach()).square().mean().sqrt()))
        before=before_optimizer_step(model,opt);opt.step();scheduler.step();steps+=1
        record['actual_updates']=after_optimizer_step(model,before);del before
        torch.cuda.synchronize();record['seconds']=time.monotonic()-tick;records.append(record)
        with (a.out/'TRAIN_health.jsonl').open('a') as f:f.write(json.dumps(record)+'\\n')
        budget();assert time.monotonic()-start < p['precheck_budget_seconds']
    assert steps==16 and scheduler.last_epoch==16
    saved_rng=rng(); saved_sha=tensor_sha(model.state_dict());model.eval()
    probe=[v[list(batches[0])].cuda() for v in train_inputs.tensors];probe[3].fill_(0)
    flow_module=model.dberta.own_flow;message=flow_module.message
    with torch.no_grad():
        on=forward_batch(model,probe).cpu().numpy();base=flow_module.last_base.cpu().numpy();flow=flow_module.last_flow.cpu().numpy()
        forward_saved=message.forward
        message.forward=lambda slots: torch.zeros((len(slots),3,100),device=slots.device,dtype=slots.dtype)
        try: off=forward_batch(model,probe).cpu().numpy()
        finally: message.forward=forward_saved
        probe[3].fill_(7);dummy=forward_batch(model,probe).cpu().numpy()
    assert np.array_equal(on,dummy) and tensor_sha(model.state_dict())==saved_sha
    rr=rng();assert torch.equal(saved_rng['torch'],rr['torch']) and all(torch.equal(x,y) for x,y in zip(saved_rng['cuda'],rr['cuda']))
    np.savez(a.out/'fixed_TRAIN_probe.npz',ids=np.asarray(guard.ids['train'])[list(batches[0])],p=on,p0_same_flow=off,base=base,flow=flow,labels_not_used=True)
    metadata=dict(plan_SHA=a.plan_sha,task_loss='mse',steps=16,next_epoch=0,next_batch=16,scheduler_total_steps=4000,warmup_steps=400,orders_SHA=p['orders_SHA'],parameter_info=info,optimizer_index_to_name=indices,initial_public_state_SHA=initial,final_state_SHA=saved_sha,official_train_dev_IDs=p['official_train_dev_IDs'],normalization='TRAIN1281 only',validation_labels_used=False,TEST_entry_not_indexed=True,not_selected_checkpoint=True)
    state=dict(metadata=metadata,model=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=cpu(scheduler.state_dict()),rng=saved_rng,prefix_records=records,omitted_rows=list(omitted))
    cp=a.out/'complete_resume_step16.pt';torch.save(state,cp);assert cp.stat().st_size<=p['checkpoint_bytes_ceiling']
    write(a.out/'guard_journal.json',guard.journal)
    result=dict(status='REAL_TRAIN_MSE_PREFIX16_COMPLETE_NOT_FULL_TRAIN',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,metadata=metadata,checkpoint_SHA=sha(cp),checkpoint_bytes=cp.stat().st_size,probe_dummy_state_RNG_pass=True,peak=budget(),wall_seconds=time.monotonic()-start,continuation_must_count_first16_in4000=True)
    write(a.out/'precheck_result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','out'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--plan-sha',required=True);run(parser.parse_args())
'''
ast.parse(text);(root/'precheck16.py').write_text(text,encoding='utf8')
# Only retain provenance/IDs/assets/budget, never old lease execution fields.
keep=('asset_sha256','runtime_versions','official_train_dev_IDs','official_role_IDs','official_ID_identity_SHA','orders_file','orders_SHA','fixed_CaReFlow_five','CaReFlow_baseline_original_reference')
plan={k:old[k] for k in keep}
plan.update(status='AUTONOMOUS_MSE_PREFIX16_PROTOCOL_FROZEN_GATE_NOT_YET_PASSED',actualclock_preparation_UTC=stamp,
 precheck_steps=16,precheck_budget_seconds=900,epochs=100,updates=4000,task_loss='mse',training_budget_seconds=7200,
 GPU_peak_ceiling_bytes=6442450944,checkpoint_bytes_ceiling=3200000000,remote_free_floor_bytes=10000000000,
 conservative_lease_end_UTC='2026-10-09T15:00:00+00:00',lease_platform_confirmed=False,
 GPU_UUID={'A':'GPU-cf38498b-a118-7e1a-45f1-0b5502b36cde','B':'GPU-609b23d6-282d-8a2b-23f5-9433b824d212'},
 storage_transport_qualified=False,health_v2_native_qualified=False,
 storage_strategy='GitHub Release complete SHA plus other-node CPU and C small originals; latest human autonomous mandate; no frozen deletions',
 TEST_previously_seen=True,VAL_selection='original batch128/101 mean MSE strict earliest; no precheck VAL label use',
 continuation_runner_qualified=False,method_changes=['Huber1 to MSE only; new instrumentation'],
 source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
(root/'precheck_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
# B gets a separate public-only restore root; A's completed restore remains intact.
rp=json.loads((base/'work/fourth_public_restore_20261008T174420Z/public_restore_plan.json').read_text(encoding='utf8'))
rp['nodes']={'B':{'uuid':plan['GPU_UUID']['B']}};rp['actualclock_preparation_UTC']=stamp
rp['public_root']='/data/coding/multimodal_flow_public_'+tag+'_B';rp['remote_record_prefix']='/data/coding/fourth_public_restore_'+tag
shutil.copyfile(base/'work/fourth_public_restore_20261008T174420Z/restore_public.py',root/'restore_public.py')
(root/'B_public_restore_plan.json').write_text(json.dumps(rp,ensure_ascii=False,indent=2),encoding='utf8')
with zipfile.ZipFile(root/'source_bundle.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in root.iterdir():
  if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
print(json.dumps({'root':str(root),'bundle_SHA':hashlib.sha256((root/'source_bundle.zip').read_bytes()).hexdigest(),'execution_enabled':False}))
