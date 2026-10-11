from pathlib import Path
import shutil,json,hashlib,datetime
w=Path(__file__).parent;repo=w.parent
now=datetime.datetime.now(datetime.timezone.utc)
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('group_teacher_local_records_'+now.strftime('%Y%m%dT%H%M%SZ'))
space={drive:shutil.disk_usage(drive).free for drive in ['C:/','D:/']}
files=list((w/'group_teacher_plan_20261005T1650Z').glob('*'))
files += [w/name for name in ['group_teacher_runtime_v1.py','check_group_teacher_v1.py','freeze_group_teacher_splits_v1.py','inspect_group_teacher_assets_v1.py','finalize_group_teacher_precheck_plan_v1.py','audit_group_teacher_local_plan_v1.py','preserve_group_teacher_local_v1.py']]
files += [repo/'outputs'/name for name in ['研究接续状态.md','按视频分组TRAIN内教师隔离设计.md','视频分组任务教师本地冻结源与划分独立核验.json','视频分组任务教师划分与预检冻结计划.json','当前情况核对_20261006T005415BJ.json']]
files += [w/'audit_group_teacher_mechanism_receipts_v1.py',repo/'outputs/视频分组教师机制回执审核准备.md']
files += [w/'compact_group_teacher_continuation_v1.py',repo/'outputs/隔离教师标量目标与流反馈可识别边界.md']
files += [w/'train_group_teacher_v1_draft.py',repo/'outputs/视频分组教师训练器草稿状态.md']
files += sorted((repo/'outputs').glob('研究接续状态_教师本地准备历史_*.md'))
assert space['D:/']>sum(p.stat().st_size for p in files)+1024*1024
dest.mkdir();members=[]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p in files:
 relative=p.relative_to(repo);target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
 expected=sha(p);assert sha(target)==expected
 members.append({'path':str(relative),'bytes':p.stat().st_size,'sha256':expected})
result={'status':'LOCAL_RESEARCH_RECORDS_PERMANENT_D_ALL_SHA_VERIFIED','utc':now.isoformat(),'destination':str(dest),'fresh_free_bytes':space,'members':members,'GPU_preflight_or_training_started':False,'remote_capture_new_root_covered':False,'transport_restriction_unchanged':True}
proof=repo/'outputs/视频分组任务教师本地准备永久D保存核验.json'
proof.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copyfile(proof,dest/proof.name);assert sha(proof)==sha(dest/proof.name)
print(json.dumps({'status':result['status'],'destination':str(dest),'files':len(members),'fresh_free_bytes':space}))
