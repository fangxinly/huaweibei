import sys,json,hashlib,shutil,zipfile,ast,datetime
from pathlib import Path
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'work/anchored_increment_candidate'))
from common import sha,read,write
stamp=sys.argv[1];clock=sys.argv[2];old=root/'work/weak_retention_train40_20261007T145201Z/bundle';orig=read(old/'training_execution_protocol.json')
source=root/'work'/('anchored_increment_source_'+stamp);source.mkdir();d=Path('D:/CodexBackups/selective_flow_20261003_1105')/('anchored_increment_actual_'+stamp);d.mkdir()
for name,h in orig['source_sha256'].items():assert sha(old/name)==h;shutil.copy2(old/name,source/name)
for f in (root/'work/anchored_increment_candidate').glob('*.py'):shutil.copy2(f,source/f.name)
shutil.copy2(old/'split.json',source/'split.json');assert sha(source/'split.json')==orig['split_sha256']
csvdir=root/'outputs/best20_best36_FIT_INNER_review_20261008T012035Z'
for role,h in [('fit','e80f286aeb2526e22956f3b0077ef89b97415dda11c6b59a44b3905fc0c1e994'),('inner','36bc71f28d0c5a29df9b3aff2a774ab97ff4183515f96720f85596d1c3908cbe')]:
    f=csvdir/('best36_'+role+'_y_b_p_video.csv');assert sha(f)==h;shutil.copy2(f,source/('best36_'+role+'.csv'))
parentd=Path('D:/CodexBackups/selective_flow_20261003_1105/weak_retention_train40_20261007T145201Z');j=parentd/'training_D_B_CPU_joint.json';joint=read(j);assert joint['status']=='PILOT40_D_ORIGINAL_CPU_CAPTURE_COMPLETE';shutil.copy2(j,source/'parent_joint.json')
matches=list(parentd.rglob('resume_and_selected_full.pt'));assert len(matches)==1;parent=matches[0];assert parent.stat().st_size==2966459315 and sha(parent)=='e173bf8b56327161a7fff430597cf7441e6d51d4a35381fa14c3b8743a01d7e6'
capacity={'actual_local_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'D_free_bytes':shutil.disk_usage(d).free,'C_free_bytes':shutil.disk_usage(root).free,'prospective_new_artifacts_budget_bytes':600000000,'additional_margin_bytes':1000000000,'original_parent_fresh_SHA':sha(parent),'actualclock_scope':clock};assert capacity['D_free_bytes']>=1600000000;write(d/'actual_local_capacity.json',capacity)
for f in source.glob('*.py'):ast.parse(f.read_text(encoding='utf8'),filename=str(f))
p=dict(status='ORIGINAL_ANCHORED_INCREMENT_SINGLE20_PROTOCOL_FROZEN',actualclock_source_freeze_UTC=clock,allowed_stages=['native','cache'],real_train_enabled=False,source_sha256={f.name:sha(f) for f in source.iterdir()},asset_sha256=orig['asset_sha256'],runtime_versions=orig['runtime_exact_versions'],GPU_UUID=orig['assigned_gpu_uuid'],remote_space_floor_bytes=2147483648,conservative_lease_end_UTC=orig['conservative_lease_end_UTC'],stage_budget_seconds={'native':180,'cache':900,'train':1800,'audit':900},parent={'full_SHA':capacity['original_parent_fresh_SHA'],'full_bytes':2966459315,'selected_state_SHA':'ad9c35a4b7900dbd8806f11a2466bbdc771211f52c44d09729b725fd10245a24','selected_epoch':36,'D_path':str(parent),'remote_path':'/data/coding/weak_retention_train40_run_20261007T145201Z/out/resume_and_selected_full.pt'},parent_D_CPU_joint_reference={'path':str(j),'SHA':sha(j)},human_authorization='Improve original scheme; original supplied third servers and24h; retain appropriate data; only four previously named state deletions authorized',native_D_other_CPU_reference=None,cache_D_reference=None,cache_p0_tolerance=1e-5,CPU_replay_tolerance=1e-4,training=dict(task_seed=128,FIT=1494,INNER=264,epochs=20,updates=940,batch=32,tail=22,selection='fixed final epoch20; no INNER selection',optimizer='AdamW lr.001 wd.01 clip1',loss='Huber delta1 +.01 mean context_squared',changes='Only donor adapters400->8->100, six directions, .125 original scale; donor-minus-zero contrast. Frozen original encoder, fields, reader, role head, fusion, predictor, gain.',not_whole_pipeline_OOF=True),reporting='All five metrics from fixed final checkpoint; same-flow zero-message p0, saved original best36 p1 and new p1; FIT/INNER and weak/strong video bootstrap; INNER already explored, original TEST participates pooled FIT, no independent official benchmark/full5fold claim',composite_state_not_standalone_full_model=True,no_additional_deletion=True)
write(source/'increment_protocol.json',p)
zp=d/'frozen_original_increment_source.zip'
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
    for f in source.iterdir():z.write(f,f.name)
with zipfile.ZipFile(zp) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for f in source.iterdir():assert hashlib.sha256(z.read(f.name)).hexdigest()==sha(f)
ref=dict(source=str(source),D=str(d),ZIP=str(zp),ZIP_SHA=sha(zp),plan_SHA=sha(source/'increment_protocol.json'),source_CRC_SHA_unique=True,actualclock=clock,capacity=capacity)
write(d/'source_freeze_reference.json',ref);write(root/'work/anchored_increment_pointer.json',ref);print(json.dumps(ref))
