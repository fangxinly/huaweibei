"""Preserve research decisions and original tool receipts on D, without credentials."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,zipfile
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
project=Path(__file__).resolve().parents[1]
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review')
names=['研究接续状态.md','研究建议交流接续.json','文献证据与双向研究决策_20261006T0742Z.md','文献交流与自动化原回执_20261006T0744Z.json','视频隔离完整流参考最小目标修订草案_v2.md','匹配消息候选对照实际结果与优化决策.md','原生位移半径CAL零标签机制实际结果.md','租期末强化保存与下一阶段研究决策.md','权限异常恢复偏好与官方依据.md']
files=[(project/'outputs'/n,'main_outputs/'+n) for n in names]
files += [(review/'outputs/第五次独立审视_CAL覆盖完成与隔离参考预检.md','review/第五次独立审视_CAL覆盖完成与隔离参考预检.md'),(review/'work/audit_fifth_coverage_and_fit_roles.py','review/audit_fifth_coverage_and_fit_roles.py'),(review/'work/fifth_coverage_and_fit_roles_audit.json','review/fifth_coverage_and_fit_roles_audit.json')]
files += [(project/'work/audit_late_lease_capture_v1.py','audit_late_lease_capture_v1.py'),(Path(__file__),'preserve_lease_research_records_v1.py')]
a.directory.mkdir(exist_ok=False,parents=True)
assert shutil.disk_usage(a.directory).free>1024**3
manifest={}
for src,name in files:
 assert src.is_file(),src
 dst=a.directory/name;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,dst)
 b=dst.read_bytes();assert b==src.read_bytes();manifest[name]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'source':str(src)}
zpath=a.directory/'research_records.zip'
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
 for name in manifest:z.write(a.directory/name,name)
 z.writestr('member_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 assert set(z.namelist())==set(manifest)|{'member_manifest.json'}
 for name,it in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==it['sha256']
proof={'status':'LOCAL_RESEARCH_AND_REVIEW_RECORDS_PERMANENT_D_SHA_CRC_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':manifest,'zip_sha256':hashlib.sha256(zpath.read_bytes()).hexdigest(),'zip_bytes':zpath.stat().st_size,'scope':'Local research/review preservation, not another remote capture or CPU model verification.'}
(a.directory/'preservation_receipt.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':proof['status'],'files':len(manifest),'zip_bytes':proof['zip_bytes'],'zip_sha256':proof['zip_sha256']}))
