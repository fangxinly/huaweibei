import sys,json,hashlib,zipfile,shutil,base64,ast
from pathlib import Path
clock=sys.argv[1]
base=Path('D:/CodexBackups/selective_flow_20261003_1105')
src=base/'old598_requested_VAL_TEST_source_20261007T112358Z/C2_budget_exception_candidate'
stamp=clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
dst=base/('C2_authorized_VAL_TEST_'+stamp);dst.mkdir()
for p in src.iterdir():
 if p.is_file() and p.name!='protocol_candidate.json':shutil.copy2(p,dst/p.name)
plan=json.loads((src/'protocol_candidate.json').read_text(encoding='utf8'))
plan.update(status='C2_ONLY_HUMAN_AUTHORIZED_EXECUTION_FROZEN',human_request='你只要测出来就行',human_execution_authorized=True,human_authorization_record={'verbatim':'你只要测出来就行','context':'Reply to C2-only save-budget exception explanation; prioritize completing measurement and immediate saving in current remaining lease'},actualclock_source_freeze_UTC=clock,save_reserve_seconds=3000)
(dst/'protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
for n,h in plan['source_sha256'].items():assert hashlib.sha256((dst/n).read_bytes()).hexdigest()==h
ast.parse((dst/'old598_C2_remaining_budget_v2.py').read_text(encoding='utf8'))
archive=dst/'source_frozen.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for p in dst.iterdir():
  if p.is_file() and p!=archive:z.write(p,p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
remote='/data/coding/C2_authorized_source_'+stamp
payload=base64.b64encode(archive.read_bytes()).decode()
upload="python -c \"import base64,io,zipfile,pathlib; p=pathlib.Path('"+remote+"'); p.mkdir(); z=zipfile.ZipFile(io.BytesIO(base64.b64decode('"+payload+"'))); assert z.testzip() is None; z.extractall(p)\""
(dst/'upload_command.txt').write_text(upload,encoding='utf8')
record={'D':str(dst),'remote_source':remote,'stamp':stamp,'plan_sha':hashlib.sha256((dst/'protocol.json').read_bytes()).hexdigest(),'source_zip_sha':hashlib.sha256(archive.read_bytes()).hexdigest(),'reserve_seconds':3000,'actualclock':clock}
(dst/'freeze_receipt.json').write_text(json.dumps(record,indent=2),encoding='utf8')
Path('work/C2_current_dispatch.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record))
