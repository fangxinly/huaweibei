import hashlib,json,shutil,sys
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
ws=Path(__file__).resolve().parent.parent;stamp=sys.argv[1];clock=sys.argv[2];pointer=read(ws/'work/official_upgrade_pointer.json');source=Path(pointer['source']);d=Path(pointer['D']);old=source/'official_train_protocol.json';p=read(old);failure=d/'A_train_failed';verdict=read(failure/'actual_D_verification.json');assert verdict['natural_exit']==1 and verdict['source_and_argv_passed']
lines=[json.loads(v) for v in (failure/'extracted/out/TRAIN_steps.jsonl').read_text().splitlines()];assert len(lines)==2 and [x['step'] for x in lines]==[1,2]
projection=max(v['seconds'] for v in lines)*4000*1.5+1900;assert 7200<projection<10800
assert not (failure/'extracted/out/DEV_selection_targets.npy').exists()
p['training_budget_seconds']=10800;p['stage_budget_seconds']['train']=10800;p['train_once_token']='/data/coding/official_anchored_upgrade_'+stamp+'_timing_once.lock';p['timing_revision_only']={'actualclock_UTC':clock,'old_plan_SHA':sha(old),'actual_failed_original_archive_SHA':verdict['archive_SHA'],'actual_projection_seconds':projection,'two_update_failure_before_any_VAL_label_or_TEST_access':True,'no_failed_training_weight_reused':True,'all_method_data_budget_order_selection_metric_settings_unchanged':True}
f=source/('official_train_timing10800_'+stamp+'_protocol.json')
with f.open('x',encoding='utf8') as out:json.dump(p,out,ensure_ascii=False,indent=2)
shutil.copy2(f,d/f.name);pointer['active_training_plan']=str(f);pointer['active_training_plan_SHA']=sha(f);pointer['actual_train_root']='/data/coding/official_upgrade_train_A_'+stamp;pointer['budget_revision_original_failure']=verdict
(ws/'work/official_upgrade_pointer.json').write_text(json.dumps(pointer,ensure_ascii=False,indent=2),encoding='utf8')
with (d/('timing_revision_receipt_'+stamp+'.json')).open('x',encoding='utf8') as out:json.dump(pointer,out,ensure_ascii=False,indent=2)
print(json.dumps({'plan':str(f),'plan_SHA':sha(f),'root':pointer['actual_train_root'],'budget_seconds':10800,'projection':projection}))
