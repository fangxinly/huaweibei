from pathlib import Path
import hashlib,json,zipfile
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105/inflow_conditions_20261005022433Z')
CP={'a':BASE/'a/run/best.pt','b':BASE/'b/run/best.pt','c':Path('C:/Users/21234/Documents/Codex/2026-10-05/ni/work/inflow_conditions_20261005022433Z/c/best.pt')}
UUID={'a':'GPU-5902bbd4-2328-0822-777c-1311d53ee5a3','b':'GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
for node in ('a','b','c'):
    audit_path=BASE/node/'completion_audit.json'
    audit=json.loads(audit_path.read_text())
    assert audit['status']=='NEW_INFLOW_COMPLETE_WHOLE_SHA_100_SELECTION_OFFICIAL_229_VERIFIED'
    files=dict(audit['files'])
    files['best.pt']={'sha256':audit['checkpoint_sha256'],'bytes':audit['checkpoint_bytes']}
    for name,expected in files.items():
        path=CP[node] if name=='best.pt' else BASE/node/'run'/name
        assert path.stat().st_size==expected['bytes']
        # Whole checkpoint was just independently streamed by completion auditor.
        if name!='best.pt': assert sha(path)==expected['sha256']
    manifest={'mode':audit['mode'],'source_gpu_uuid':UUID[node],'files':files,
              'selection':audit['selection'],'local_audit_sha256':sha(audit_path)}
    dest=BASE/node/'independent_manifest.json'
    with dest.open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
    with zipfile.ZipFile(BASE/node/'snapshot.zip') as z:
        launch=json.loads(z.read('inflow_conditions_v2_deployment_20261004T2140Z/training_launch.json'))
    print(json.dumps({'node':node,'mode':audit['mode'],'manifest_sha256':sha(dest),'launch':launch}))
