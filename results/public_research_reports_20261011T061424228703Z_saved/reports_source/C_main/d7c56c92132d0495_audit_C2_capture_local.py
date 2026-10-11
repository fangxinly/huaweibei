import json,hashlib,zipfile,sys
from pathlib import Path
r=json.loads(Path('work/C2_current_dispatch.json').read_text(encoding='utf8'));d=Path(r['D'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prefix=sys.argv[1]
archive=d/(prefix+'_complete_capture.zip');receipt=json.loads((d/(prefix+'_capture_receipt.json')).read_text(encoding='utf8'))
assert sha(archive)==receipt['sha256']
out=d/(prefix+'_original_capsule');out.mkdir()
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
 z.extractall(out)
m=json.loads((out/'manifest.json').read_text(encoding='utf8'))
for n,h in m['member_sha256'].items():assert sha(out/n)==h
read=lambda p:json.loads(p.read_text(encoding='utf8'))
g=read(out/'original/actual_stage_receipt.json');c=read(out/'execution/actual_child.json');n=read(out/'execution/natural_exit.json')
assert n['natural_exit']==0 and n['pid']==c['pid']==g['pid'] and n['fullargv']==c['fullargv']==g['fullargv']
assert g['plan_sha256']==r['plan_sha'] and sha(out/'source/protocol.json')==r['plan_sha']
if prefix=='B':
 a=read(d/'A_original_capsule/original/actual_stage_receipt.json')
 assert g['original_receipt_sha256']==sha(d/'A_original_capsule/original/actual_stage_receipt.json') and g['scores']==a['scores'] and g['CPU_model_forward'] is False
print(json.dumps({'prefix':prefix,'archive_SHA':sha(archive),'pid':g['pid'],'exit_utc':n['actual_exit_utc'],'scores':g['scores']},ensure_ascii=False))
