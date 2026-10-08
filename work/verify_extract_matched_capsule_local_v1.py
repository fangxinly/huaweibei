import hashlib,json,sys,zipfile
from pathlib import Path
p=Path(sys.argv[1]);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());r=read(p/'capture_receipt.json');ex=read(p/'capture_actual_exit.json')
assert ex['exit_code']==0 and ex['natural_wait_verified'] and ex['original_receipt_sha256']==sha(p/'capture_receipt.json') and ex['child_pid']==r['pid'] and ex['child_full_argv'][1:]==r['argv']
assert sha(p/'snapshot.zip')==r['snapshot_sha256']
with zipfile.ZipFile(p/'snapshot.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==r['members'] and all('..' not in Path(n).parts and not n.startswith('/') for n in z.namelist())
 z.extractall(p);m=json.loads(z.read('member_manifest.json'))
 for n,i in m['small_members'].items():assert hashlib.sha256(z.read(n)).hexdigest()==i['sha256']==sha(p/n)
print(json.dumps({'status':'ACTUAL_COMPLETE_CAPSULE_FULL_SHA_CRC_UNIQUE_EXTRACTED_D','D':str(p),'members':r['members'],'snapshot_sha256':r['snapshot_sha256']}))
