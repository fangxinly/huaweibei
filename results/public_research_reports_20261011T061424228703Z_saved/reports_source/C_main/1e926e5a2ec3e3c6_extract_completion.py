from pathlib import Path
import datetime,hashlib,json,zipfile
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105/inflow_conditions_20261005022433Z')
DEP='inflow_conditions_v2_deployment_20261004T2140Z'
ASSIGN={'a':('none','GPU-5902bbd4-2328-0822-777c-1311d53ee5a3',4473),
        'b':('state','GPU-c65ea6c9-c9b4-d2d8-a1ab-5867662ed26d',4787),
        'c':('task','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa',7776)}
rows=[]
for node,(mode,uuid,pid) in ASSIGN.items():
    root=BASE/node
    p=json.loads((root/'proof.json').read_text())
    data=(root/'snapshot.zip').read_bytes()
    assert len(data)==p['archive_bytes'] and hashlib.sha256(data).hexdigest()==p['archive_sha256']
    with zipfile.ZipFile(root/'snapshot.zip') as z:
        members=json.loads(z.read('member_manifest.json'))
        assert members==p['members']
        assert set(z.namelist())==set(members)|{'member_manifest.json'}
        assert len(z.namelist())==len(set(z.namelist()))
        for name,entry in members.items():
            raw=z.read(name)
            assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'],name
        resources=json.loads(z.read('actual_resources.json'))
        assert uuid in resources['gpu']['output']
        assert not resources['compute']['output'].strip()
        assert not any(line.split() and line.split()[0]==str(pid) for line in resources['processes']['output'].splitlines())
        prefix=DEP+'/run_'+mode+'/'
        log=z.read(DEP+'/training.log').decode()
        assert 'INFLOW_CONDITIONS_RUN_COMPLETE' in log and 'Traceback' not in log
        history=json.loads(z.read(prefix+'history.json'))
        assert [r['epoch'] for r in history]==list(range(1,101))
        out=root/'run'
        out.mkdir(exist_ok=False)
        for name in ['protocol.json','batch_orders.npy','history.json','selection.json','results.json','predictions.npz']:
            (out/name).write_bytes(z.read(prefix+name))
        selection=json.loads(z.read(prefix+'selection.json'))
        results=json.loads(z.read(prefix+'results.json'))
        excluded=[r for r in p['excluded_weights'] if r['path'].endswith('/run_'+mode+'/best.pt')]
        assert len(excluded)==1
        rows.append({'node':node,'mode':mode,'epochs':100,'captured_at':resources['captured_at'],'gpu':resources['gpu']['output'],
            'remote_free_bytes':resources['disk_free_bytes'],'checkpoint_bytes':excluded[0]['bytes'],
            'selection':selection,'results':results,'archive_sha256':p['archive_sha256']})
report={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'METADATA_AND_COMPLETION_PRECHECK_PASSED_FULL_CHECKPOINT_PENDING','rows':rows}
(BASE/'completion_precheck.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
