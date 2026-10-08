import hashlib,json,shutil,zipfile
from pathlib import Path
b=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');o=b/'outputs'
D=Path('D:/CodexBackups/selective_flow_20261003_1105/official_baseline_artifact_guard_preparation_20261007T000208TZ')
clock='2026-10-07 00:02:34 UTC'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=read(o/'正式CaReFlow原整模型与标签守卫准备接续.json')
link={'status':audit['status'],'actual_clock_start_utc':audit['clock_start_utc'],
      'control_record_clock_utc':clock,'D':str(D),'byte_audit':str(D/'preparation/artifact_byte_audit.json'),
      'whole_checkpoint_sha256':audit['sha256'],'whole_checkpoint_bytes':audit['bytes'],
      'ZIP_CRC_unique_members':333,'large_original_local_copy_not_remote_redownload':True,
      'formal_cache_reuse_approved':False,'formal_runtime_end_to_end_label_guard_passed':False,
      'directory_stamp_typo':'TZ suffix is naming transformation error; actual clock fields authoritative, no timestamp backfill'}
pro=o/'正式CaReFlow比较来源与预算本地准备接续.json';v=read(pro);v['latest_original_checkpoint_byte_preservation_and_guard_preparation']=link
write(pro,v)
md=o/'正式CaReFlow比较来源与预算本地准备接续.md'
md.write_text(md.read_text(encoding='utf-8')+'\n补充实际clock '+clock+'：已从原保存manifest定位C的指定seed128 epoch72完整模型，fresh SHA/ZIP CRC/333唯一成员通过，原742279993字节已本地完整复制D并复核；原缓存小目录无best.pt并不代表原完整模型缺失。此为字节保存门，未Torch加载/张量或模型重放，不冒正式缓存可复用。train/dev-only守卫仅合成检查，未接入完整runner。续读正式CaReFlow原整模型与标签守卫准备接续.md/json。新目录TZ后缀为命名变换错误，实际clock字段为准。\n',encoding='utf-8')
state=o/'完整流匹配U_W头实际接续.json';v=read(state);v['official_baseline_artifact_guard_preparation']=link;write(state,v)
ledger=o/'研究建议交流接续.json';v=read(ledger);v['official_baseline_artifact_guard_preparation']=link
v['no_advisor_message_for_this_local_preparation']=True;write(ledger,v)
short=o/'研究接续状态.md';text=short.read_text(encoding='utf-8');rest=text.split('\n\n',1)[1]
short.write_text('准备实际clock '+clock+'：指定CaReFlow原C完整best742279993字节fresh SHA/CRC/333唯一成员过并本地完整D保存，无远程下载/新训练/标签/成绩。原小缓存目录缺best不代表整模型缺失；未Torch张量/模型重放，正式cache复用仍未通过。train/dev-only守卫合成检查过，完整正式runner/共同订单预算选模未冻。先读正式CaReFlow比较来源与预算本地准备接续.md/json及正式CaReFlow原整模型与标签守卫准备接续.md/json。16建议已failed不轮询/重发；201原结果不变、整体五项目标与租期保存未完成。\n\n'+rest,encoding='utf-8')
control=D/'control';assert not control.exists();control.mkdir()
for name in ['研究接续状态.md','正式CaReFlow比较来源与预算本地准备接续.md','正式CaReFlow比较来源与预算本地准备接续.json',
             '正式CaReFlow原整模型与标签守卫准备接续.md','正式CaReFlow原整模型与标签守卫准备接续.json',
             '完整流匹配U_W头实际接续.json','研究建议交流接续.json']:
    shutil.copy2(o/name,control/name)
shutil.copy2(__file__,control/Path(__file__).name)
members={p.name:sha(p) for p in control.iterdir() if p.is_file()};write(control/'member_SHA.json',members)
with zipfile.ZipFile(control/'snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in [*members,'member_SHA.json']:z.write(control/name,name)
with zipfile.ZipFile(control/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)+1
    for name,h in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
write(control/'archive_receipt.json',{'record_clock_utc':clock,'zip_sha256':sha(control/'snapshot.zip'),
      'unique_members':len(members)+1,'CRC_all_member_SHA':True,'no_new_experiment_or_remote_capture':True})
print(json.dumps({'D_control':str(control),'members':len(members)+1,'whole_checkpoint_bytes_saved_separately':True},ensure_ascii=False))
