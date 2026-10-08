from pathlib import Path
import datetime,hashlib,json,shutil
r=Path(__file__).resolve().parent.parent;out=r/'outputs';target=r/'work/utility_completed_snapshots_20261005T0439Z'
target.mkdir(exist_ok=False);rows=[];audits=[]
for node,stamp in [('a','202610050439AC'),('b','202610050436B'),('c','202610050439AC')]:
    p=out/f'逐样本效用完整审核_{node.upper()}_{stamp}.json';data=json.loads(p.read_text(encoding='utf-8'))
    assert data['status']=='UTILITY_SELECTED_NODES_COMPLETED_WHOLE_SHA_100_SELECTION_OFFICIAL_DEV_VERIFIED'
    assert data['nodes_verified']==[node] and data['rows'][0]['full_checkpoint_verified']
    row=data['rows'][0];rows.append(row);audits.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    dest=target/node;dest.mkdir(exist_ok=False)
    for name in ['snapshot.zip','proof.json']:
        src=r/'work'/('utility_live_'+stamp)/node/name;dst=dest/name
        shutil.copyfile(src,dst);assert hashlib.sha256(src.read_bytes()).hexdigest()==hashlib.sha256(dst.read_bytes()).hexdigest()
    assert hashlib.sha256((dest/'snapshot.zip').read_bytes()).hexdigest()==row['archive_sha256']
report={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'UTILITY_SELECTED_NODES_COMPLETED_WHOLE_SHA_100_SELECTION_OFFICIAL_DEV_VERIFIED','nodes_verified':['a','b','c'],'rows':rows,'individual_audits':audits,'limits':'Combined separately audited actual completion snapshots. Independent seven-file target copies and donor diagnostics are separate.'}
dest=out/'逐样本效用三组完整权重核验.json'
with dest.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':report['status'],'snapshots':str(target)}))
