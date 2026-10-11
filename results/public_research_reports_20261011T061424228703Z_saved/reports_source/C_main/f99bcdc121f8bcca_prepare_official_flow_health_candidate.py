"""Single-variable MSE candidate plus non-RNG-changing TRAIN health logging."""
import ast,hashlib,json,shutil
from pathlib import Path
base=Path(__file__).resolve().parents[1]
source=base/'work/official_anchored_upgrade_20261008T053146Z'
dest=base/'work/official_flow_health_candidate_v1';dest.mkdir(exist_ok=False)
for p in source.iterdir():
    if p.suffix in ('.py','.npy') and p.name!='run_stage.py':shutil.copyfile(p,dest/p.name)
shutil.copyfile(base/'work/training_health_v1.py',dest/'training_health_v1.py')
text=(dest/'train_official.py').read_text(encoding='utf-8-sig')
def change(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old))
    text=text.replace(old,new)
change('from official_upgrade import OfficialUpgrade,objective','from official_upgrade import OfficialUpgrade\nfrom training_health_v1 import gradient_summary,before_update,after_update,original_task_objective')
change("assert p['native_CPU_D_qualification'];assert p['epochs']", "assert p['native_CPU_D_qualification'] and p['new_candidate_full_encoder_precheck_D_B'];assert p['task_loss']=='mse';assert p['epochs']")
change('loss=objective(model,pred,batch[3])', "loss=original_task_objective(model.dberta.own_flow,pred,batch[3].view(-1),kind=p['task_loss'],role='TRAIN')")
change('            opt.step();scheduler.step();steps+=1;', '''            health=None
            if steps<16 or (steps+1)%40==0:
                health=dict(actual_UTC=utc(),step=steps+1,gradients=gradient_summary(model),learning_rates_before_update=[g['lr'] for g in opt.param_groups],
                            gain_tanh_before_update=float(model.dberta.own_flow.gain.detach().tanh()),
                            context_RMS=float(model.dberta.own_flow.last_context.detach().square().mean().sqrt()),
                            flow_minus_base_RMS=float((model.dberta.own_flow.last_flow.detach()-model.dberta.own_flow.last_base.detach()).square().mean().sqrt()))
                before=before_update(model)
            opt.step();scheduler.step();steps+=1;
            if health is not None:
                health['sampled_actual_Adam_updates']=after_update(model,before);del before
                with (a.out/'TRAIN_health.jsonl').open('a') as healthfile:healthfile.write(json.dumps(health)+'\\n')
            ''')
change('metadata=dict(plan_SHA=a.plan_sha', "metadata=dict(task_loss=p['task_loss'],TRAIN_health='First16 updates plus last update of each epoch; full gradient RMS/parameter counts and fixed-sample actual Adam updates; update includes decay',plan_SHA=a.plan_sha")
text=text.replace('CLEAN_OFFICIAL_ANCHORED_UPGRADE_TRAIN100_COMPLETE','OFFICIAL_ORIGINAL_FLOW_MSE_HEALTH_CANDIDATE_TRAIN100_COMPLETE')
ast.parse(text);(dest/'train_official.py').write_text(text,encoding='utf-8')
for p in dest.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
receipt=dict(status='MSE_ONLY_OFFICIAL_FLOW_CANDIDATE_SOURCE_PREPARED_NOT_FULL_ENCODER_QUALIFIED',
    task_loss='mse',structural_or_learning_rate_changes=False,polarity_intensity_aux_heads=False,
    same_seed_orders_100epochs_4000updates=True,same_official_TRAIN_VAL_TEST_roles=True,
    same_VAL_batch_MSE_strict_earliest_selection=True,core_alignment_loss_added=False,
    new_candidate_full_encoder_precheck_D_B=False,permanent_backup_capacity_passed=False,
    original_source_unchanged=True,no_training_or_new_VAL_TEST_scores=True,
    source_SHA256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()})
(dest/'preparation_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(status=receipt['status'],files=len(receipt['source_SHA256']))))
