import argparse,json,hashlib,shutil
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--actualclock',required=True);p.add_argument('--phase',required=True);p.add_argument('--evidence',type=Path);a=p.parse_args()
ws=Path(__file__).resolve().parent.parent;pointer=json.loads((ws/'work/official_upgrade_pointer.json').read_text(encoding='utf8'));d=Path(pointer['D']);snapshot=d/('progress_'+a.actualclock.replace('-','').replace(':','').replace(' ','_')+'.json')
plan=Path(pointer['active_training_plan']);assert sha(plan)==pointer['active_training_plan_SHA']
evidence=json.loads(a.evidence.read_text(encoding='utf8')) if a.evidence else None
value={'actualclock_UTC':a.actualclock,'phase':a.phase,'pointer':pointer,'train_plan_SHA':sha(plan),'native_D_CPU_complete':True,'remote_root':pointer['actual_train_root'],'physical_progress_original':{'local_path':str(a.evidence),'SHA':sha(a.evidence),'content':evidence} if a.evidence else None,'active_current_connections':{'A_SSH':89189,'B_SSH':39816,'A_SFTP':9027,'B_SFTP':9666},'current_upgrade_official_VAL_TEST_five_complete':False,'whole_goal_complete':False,'CaReFlow_fixed':{'VAL_MAE':.604523826276,'TEST_MAE':.619535293923368},'C_D_part_capacity':{'C_free':shutil.disk_usage(pointer['C_part_backup']).free,'D_free':shutil.disk_usage(d).free},'prior_merged_model_excluded_from_comparison':True}
with snapshot.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
(ws/'outputs/官方划分原方案消息改进与CaReFlow对齐实际接续.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'phase':a.phase,'snapshot':str(snapshot)},ensure_ascii=False))
