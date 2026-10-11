from pathlib import Path
import argparse,datetime,hashlib,json
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--stamp',required=True);a.add_argument('--label',required=True);a.add_argument('--pair-source',type=Path,required=True);c=a.parse_args()
o=Path('outputs');captures=o/('租期'+c.label+'真实动态保存快照核验.json');local=o/('租期'+c.label+'永久本地资料保存核验.json');ind=o/('租期'+c.label+'独立节点资料保存核验.json');r=json.loads(captures.read_text(encoding='utf-8'));l=json.loads(local.read_text(encoding='utf-8'));i=json.loads(ind.read_text(encoding='utf-8'))
assert r['status']=='SIX_REAL_DEADLINE_OLD_AND_NEW_CAPTURES_ALL_SHA_VERIFIED' and l['status']=='LOCAL_ORIGINAL_AND_PERMANENT_D_ZIP_ALL_SHA_VERIFIED'
assert i['status']=='INDEPENDENT_CPU_METADATA_ZIP_AND_ALL_MEMBERS_SHA_VERIFIED' and i['archive_sha256']==l['archive_sha256']
rows=[]
for node in ['a','b','c']:
 p=Path('work')/('lease_'+c.stamp)/node/'pair_receipt.json';receipt=json.loads(p.read_text());assert receipt['node']==node and receipt['deadline']==r['deadline'] and receipt['capture_pair_source_sha256']==sha(c.pair_source)
 for entry in receipt['records']:
  proof=p.parent/entry['kind']/'proof.json';assert entry['returncode']==0 and entry['proof_sha256']==sha(proof)
 rows.append({'node':node,'pair_receipt_sha256':sha(p),'started_at':receipt['started_at'],'finished_at':receipt['finished_at']})
result={'status':'REAL_DEADLINE_OLD_NEW_CAPTURES_PERMANENT_LOCAL_AND_INDEPENDENT_CPU_SAVED','verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deadline':r['deadline'],'pair_receipts':rows,'capture_audit_sha256':sha(captures),'local_receipt_sha256':sha(local),'independent_receipt_sha256':sha(ind),'permanent_directory':l['destination'],'archive_sha256':l['archive_sha256'],'members':l['members'],'limits':'Full weights stored separately. This deadline completion does not imply whole research completion.'}
with (o/('租期'+c.label+'动态保存完成记录.json')).open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result))
