from pathlib import Path
import datetime, hashlib, json, shutil

w=Path(__file__).parent
prep=w/'correction_calibration_collection_v2_20261006T0414Z'
plan=json.loads((prep/'same_teacher_collection_plan_v2.json').read_text())
now=datetime.datetime.now(datetime.timezone.utc)
space={drive:shutil.disk_usage(drive).free for drive in ['C:/','D:/']}
assert space['D:/'] >= plan['minimum_local_permanent_free_bytes']
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('same_teacher_collection_actual_'+now.strftime('%Y%m%dT%H%M%SZ'))
dest.mkdir()
files=[p for p in prep.iterdir() if p.is_file()]
files += [w/'check_same_teacher_deployment_v1.py']
names=[p.name for p in files]
for selected in plan['folds']:
    node=dest/selected['node'];node.mkdir()
    for p in files:
        target=node/p.name;shutil.copyfile(p,target)
        assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(target.read_bytes()).digest()
    auth={'status':'FRESH_LOCAL_AND_REMOTE_COLLECTION_SPACE_VERIFIED','utc':now.isoformat(),
          'local_permanent_free_bytes':space['D:/'],'expected_uuid':selected['expected_uuid'],'deletion_required':False}
    (node/'fresh_space_authorization.json').write_text(json.dumps(auth,indent=2))
context={'utc':now.isoformat(),'destination':str(dest),'fresh_free_bytes':space,'source_names':names,
         'remote_root':plan['teacher_root']+'/'+plan['collection_subdirectory']}
out=w/'same_teacher_actual_context_20261006T0536Z.json'
assert not out.exists()
out.write_text(json.dumps(context,indent=2))
print(json.dumps(context))
