from pathlib import Path
import datetime,hashlib,json
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105/inflow_conditions_20261005022433Z')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for node in ('a','b','c'):
    source=json.loads((BASE/node/'completion_audit.json').read_text())
    manifest_path=BASE/node/'independent_manifest.json'
    m=json.loads(manifest_path.read_text())
    target=json.loads((BASE/node/'destination_verification.json').read_text())
    assert target['status']=='NEW_INFLOW_INDEPENDENT_LEASED_COPY_ALL_SEVEN_FILES_SHA_VERIFIED'
    assert target['manifest_sha256']==sha(manifest_path)
    assert target['files']==m['files'] and target['selection']==m['selection']==source['selection']
    assert target['local_audit_sha256']==sha(BASE/node/'completion_audit.json')
    assert target['source_gpu_uuid']==m['source_gpu_uuid'] and target['source_gpu_uuid']!=target['target_gpu_uuid']
    assert source['checkpoint_sha256']==target['files']['best.pt']['sha256']
    rows.append({'node':node,'mode':source['mode'],'checkpoint':source['checkpoint'],'checkpoint_sha256':source['checkpoint_sha256'],
        'checkpoint_bytes':source['checkpoint_bytes'],'destination_gpu_uuid':target['target_gpu_uuid'],'destination_directory':target['directory'],
        'destination_proof_sha256':sha(BASE/node/'destination_verification.json'),'selection':source['selection']})
r={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'THREE_NEW_FULL_CHECKPOINTS_LOCAL_AND_INDEPENDENT_LEASED_SEVEN_FILE_SHA_VERIFIED','rows':rows,
    'limits':'Independent leased copies are temporary; local whole checkpoints remain required. All prior artifacts retained; this registry is additive.'}
(Path('outputs')/'新增三份权重保存核验.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r,indent=2))
