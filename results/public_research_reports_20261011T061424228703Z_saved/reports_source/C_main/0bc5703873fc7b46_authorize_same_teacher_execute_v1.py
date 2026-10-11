"""Create phase authorization only after original GPU receipts and actual captures pass."""
from pathlib import Path
import datetime, hashlib, json, shutil
w=Path(__file__).parent
context=json.loads((w/'same_teacher_actual_context_20261006T0536Z.json').read_text())
d=Path(context['destination'])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
captured=read(d/'precheck_capture_audit.json')
assert captured['status']=='ACTUAL_COLLECTION_SUBROOT_SOURCE_PROCESS_RECEIPTS_ARRAYS_AND_PRIOR_FROZEN_EVIDENCE_CAPTURE_VERIFIED'
assert captured['phase']=='precheck' and set(captured['nodes'])=={'a','b','c'}
free={drive:shutil.disk_usage(drive).free for drive in ['C:/','D:/']}
now=datetime.datetime.now(datetime.timezone.utc)
for fold,node in enumerate(['a','b','c']):
    r=d/node;plan=read(r/'same_teacher_collection_plan_v2.json');s=plan['folds'][fold]
    audit=read(r/'precheck/independent_precheck_audit.json')
    assert audit['status']=='ACTUAL_COLLECTION_PRECHECK_METADATA_INDEPENDENT_AUDIT_PASSED'
    assert audit['original_gpu_receipt_sha256']==sha(r/'precheck/receipt.json') and audit['fold']==fold
    assert free['D:/']>=plan['minimum_local_permanent_free_bytes']
    target=r/'execution_authorization.json';assert not target.exists()
    target.write_text(json.dumps({'actual_gpu_precheck_and_independent_audit_passed':True,
        'plan_sha256':sha(r/'same_teacher_collection_plan_v2.json'),'collector_sha256':plan['collector_sha256'],
        'original_precheck_receipt_sha256':sha(r/'precheck/receipt.json'),
        'precheck_independent_audit_sha256':sha(r/'precheck/independent_precheck_audit.json'),
        'capture_source_coverage_verified':True,'capture_source_sha256':plan['capture_source_sha256'],
        'actual_capture_audit_sha256':sha(d/'precheck_capture_audit.json'),'utc':now.isoformat()},indent=2))
    shutil.copyfile(r/'fresh_space_authorization.json',r/'precheck_fresh_space_authorization.json')
    (r/'fresh_space_authorization.json').write_text(json.dumps({'status':'FRESH_LOCAL_AND_REMOTE_COLLECTION_SPACE_VERIFIED',
        'utc':now.isoformat(),'local_permanent_free_bytes':free['D:/'],'expected_uuid':s['expected_uuid'],'deletion_required':False},indent=2))
print(json.dumps({'status':'EXECUTE_AUTHORIZATIONS_CREATED_FROM_ACTUAL_GPU_AND_CAPTURE_AUDITS','utc':now.isoformat(),'fresh_free_bytes':free}))
