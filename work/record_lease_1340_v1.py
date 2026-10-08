from pathlib import Path
import datetime,hashlib,json
r=Path(__file__).resolve().parents[1];o=r/'outputs';load=lambda p:json.loads(p.read_text(encoding='utf-8'))
audit=load(o/'租期1340真实动态保存快照核验.json');local=load(o/'租期1340永久本地资料保存核验.json');remote=load(o/'租期1340独立节点资料保存核验.json');live=load(o/'主任务反事实效用运行核验_1340保存时.json')
assert audit['status']=='SIX_REAL_DEADLINE_OLD_AND_NEW_CAPTURES_ALL_SHA_VERIFIED' and len(audit['rows'])==6
assert local['archive_sha256']==remote['archive_sha256'] and local['manifest_sha256']==remote['manifest_sha256'] and local['members']==remote['members']==190
pairsha=hashlib.sha256((r/'work/run_lease_capture_pair_v1.py').read_bytes()).hexdigest()
for node in ['a','b','c']:
 d=r/'work/lease_202610050540Z'/node;p=load(d/'pair_receipt.json');assert p['node']==node and p['deadline']=='2026-10-05T05:40:00+00:00' and p['capture_pair_source_sha256']==pairsha
 for rec in p['records']:assert rec['returncode']==0 and rec['proof_sha256']==hashlib.sha256((d/rec['kind']/'proof.json').read_bytes()).hexdigest()
proofs={name:hashlib.sha256((o/name).read_bytes()).hexdigest() for name in ['租期1340真实动态保存快照核验.json','租期1340永久本地资料保存核验.json','租期1340独立节点资料保存核验.json','主任务反事实效用运行核验_1340保存时.json']}
report={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline_beijing':'2026-10-05 13:40','status':'REAL_1340_OLD_NEW_CAPTURES_PERMANENT_D_AND_INDEPENDENT_CPU_METADATA_SAVED','captures':audit['rows'],'permanent_metadata':local,'independent_cpu':remote,'run_epochs_at_capture':{x['node']:x['epochs'] for x in live['rows']},'proof_sha256':proofs,'pending_deadlines':['2026-10-05 14:10','2026-10-05 14:30'],'limits':'Six captures were actualUTC05:40:17/18. Complete weights separately preserved. New v5 still training, C provisional supplement separately audited. Overall research incomplete.'}
p=o/'租期1340动态保存完成记录.json';assert not p.exists();p.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'status':report['status'],'epochs':report['run_epochs_at_capture']}))
