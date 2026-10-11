import json, hashlib, shutil, zipfile
from pathlib import Path
from datetime import datetime, timezone

base=Path("C:/Users/21234/Documents/Codex/2026-10-05/ni")
record_path=base/"outputs/研究建议交流接续.json"
state_path=base/"outputs/研究接续状态.md"
record=json.loads(record_path.read_text(encoding="utf-8"))
now=datetime.now(timezone.utc)
stamp=now.strftime("%Y%m%dT%H%M%SZ")
free={root:shutil.disk_usage(root).free for root in ("C:/","D:/")}
assert free["C:/"]>100*1024*1024 and free["D:/"]>100*1024*1024
dest=Path("D:/CodexBackups/selective_flow_20261003_1105")/("independent_review_coordination_"+stamp)
dest.mkdir(exist_ok=False)
original_state=state_path.read_bytes()
(dest/"研究接续状态_交流建立前.md").write_bytes(original_state)
record["first_sent_report_sha256"]={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in record["source_reports"]}
record["initial_prompt_sha256"]=hashlib.sha256(record["first_sent_prompt"].encode("utf-8")).hexdigest()
record["permanent_preservation_directory"]=str(dest)
record["updated_at_utc"]=now.isoformat()
record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
note="\n研究建议交流："+now.isoformat()+"，用户授权的独立建议聊天“多模态消息控制研究建议”01a10fcb-6663-70a2-9a76-60e5634d0c03已创建并在审视首批真实结果。主实验仍在本聊天；后续实质结果保存后发送同一建议聊天，读取建议并记录采纳/暂缓/拒绝理由，不自动改冻结源。接续见研究建议交流接续.json。原十分钟自动任务已通过官方工具更新，未新增自动任务；本次协调不产生新科学分数。\n"
state_path.write_bytes(original_state+note.encode("utf-8"))
for p in (record_path,state_path,Path(__file__)):
    target=dest/p.name
    data=p.read_bytes()
    target.write_bytes(data)
    assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(data).digest()
members={p.name:{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in dest.iterdir() if p.is_file()}
manifest={"status":"INDEPENDENT_REVIEW_COORDINATION_AND_CONTINUATION_D_SHA_VERIFIED","actual_utc":now.isoformat(),"fresh_free_bytes":free,"files":members,"not_scientific_result":True,"no_old_files_deleted":True}
(dest/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
package=dest/"coordination.zip"
all_files=sorted(p for p in dest.iterdir() if p.is_file() and p!=package)
with zipfile.ZipFile(package,"x",compression=zipfile.ZIP_DEFLATED) as z:
    for p in all_files:z.write(p,p.name)
with zipfile.ZipFile(package) as z:
    names=z.namelist()
    assert len(names)==len(set(names))==len(all_files)
    assert z.testzip() is None
    for p in all_files: assert z.read(p.name)==p.read_bytes()
evidence={"status":manifest["status"],"actual_utc":now.isoformat(),"directory":str(dest),"manifest_sha256":hashlib.sha256((dest/"manifest.json").read_bytes()).hexdigest(),"zip_sha256":hashlib.sha256(package.read_bytes()).hexdigest(),"zip_crc_and_unique_members_verified":True}
(base/"outputs/独立研究建议交流与永久D保存.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(evidence,ensure_ascii=True))

