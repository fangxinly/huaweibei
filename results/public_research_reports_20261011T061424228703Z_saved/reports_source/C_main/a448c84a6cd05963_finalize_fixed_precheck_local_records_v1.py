import hashlib,json,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
BUNDLE=ROOT/'work/minimal_fixed_precheck_local_20261006T1240Z'
D=Path(json.loads((BUNDLE/'permanent_D_location.json').read_text())['directory'])
utc=datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
official=D/'official_automation_and_preservation_tool_receipts.json'
data=json.loads(official.read_text(encoding='utf-8'))
result=data['official_automation_update']['result']
assert result.get('isError') is False
result_status=json.loads(result['content'][1]['text'])
assert result_status['automationId']=='automation' and result_status['status']=='ACTIVE'
ledger_path=ROOT/'outputs/研究建议交流接续.json'
ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
ledger['updated_at_utc']=utc;ledger['updated_utc']=utc
ledger['automation'].update({'updated_at_utc':utc,'prompt_sha256':hashlib.sha256(data['official_automation_update']['arguments']['prompt'].encode()).hexdigest(),'original_tool_receipt':str(official),'interval_minutes':10,'duplicate_automation_created':False})
ledger['local_fixed_precheck_candidate']['official_update_receipt']=str(official)
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state_path=ROOT/'outputs/研究接续状态.md'
text=state_path.read_text(encoding='utf-8')
assert text.count('独立正规方程/14代数见证')==1
text=text.replace('独立正规方程/14代数见证','独立正规方程及后续JSON代数核验')
state_path.write_text(text,encoding='utf-8')
out=D/'final_records';out.mkdir(exist_ok=False)
paths=[ledger_path,ROOT/'outputs/研究接续状态.md',ROOT/'outputs/完整流fixed_v2预检与全状态重放本地接续.md',official,Path(__file__)]
members={}
for p in paths:
 q=out/p.name;shutil.copyfile(p,q);assert sha(p)==sha(q)
 members[q.name]={'bytes':q.stat().st_size,'sha256':sha(q)}
package=D/'final_records.zip'
with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(out.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(package) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 assert set(z.namelist())==set(members)
 for name,meta in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==meta['sha256']
receipt={'status':'LOCAL_PREPARATION_FINAL_RECORDS_AND_OFFICIAL_AUTOMATION_D_SHA_ZIP_CRC_PASSED','actual_utc':utc,'members':members,'package_sha256':sha(package),'new_GPU_or_scores':False,'new_remote_capture':False,'whole_research_or_lease_complete':False}
(D/'final_records_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'actual_utc':utc,'new_GPU':False,'new_scores':False}))
