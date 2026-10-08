from pathlib import Path
import json,hashlib,shutil,zipfile
src=Path('work/paired_physical_health_20261007T024247Z.json');r=json.loads(src.read_text(encoding='utf-8'))
assert all(x['result']['status']=='fulfilled' and x['result']['value']['exit_code']==0 for x in r['session_closures'])
d=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_physical_health_actual_20261007T024247Z');assert not d.exists();d.mkdir()
shutil.copyfile(src,d/src.name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
note='原物理观察UTC02:42:47/48：F完成69轮/2776行更新，CaReFlow完成76轮/3077行更新，原child fullargv/全sourceSHA/各自UUID和compute/空间通过、stderr无Traceback、累计reserved4525654016/4481613824低6GiB，natural_exit仍不存在。健康继续，不重复启动/停止。\n这是原工具观察转存D，不是remote COMPLETE capture/整weight保存/CPU前向/新五项成绩；无本轮B fresh。SSH34456/60339已明确exit实际0、全部旧ID禁复用。整体目标/100完成与保存/最终TEST/租期强化保存未完成。\n'
p=Path('outputs/研究接续状态.md');p.write_text(note+'先读正式双方官方fullTRAIN100实际训练接续.json最新physical_health指针。\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
p=Path('outputs/正式双方官方fullTRAIN100实际训练接续.json');s=json.loads(p.read_text(encoding='utf-8'));s['latest_actual_physical_health']={'clock_checked_utc':r['clock_checked_utc'],'D_raw_record':str(d/src.name),'D_raw_record_sha256':sha(d/src.name),'F_completed_epoch':69,'F_observed_update_lines':2776,'CaReFlow_completed_epoch':76,'CaReFlow_observed_update_lines':3077,'not_COMPLETE_capture_or_CPUs_or_final_weights':True,'all_session_closures_actual_zero':True};p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(d/'actual_health_note.md').write_text(note,encoding='utf-8')
for p in [Path('outputs/研究接续状态.md'),Path('outputs/正式双方官方fullTRAIN100实际训练接续.json')]:shutil.copyfile(p,d/p.name)
files=[p for p in d.iterdir() if p.is_file()];manifest={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in files};(d/'local_member_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files.append(d/'local_member_manifest.json')
with zipfile.ZipFile(d/'local_observation_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,p.name)
with zipfile.ZipFile(d/'local_observation_snapshot.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
receipt={'status':'ACTUAL_LOCAL_D_ORIGINAL_TOOL_TRANSCRIPT_HEALTH_SAVE_SHA_CRC_PASSED_NOT_REMOTE_CAPTURE','clock_checked_utc':r['clock_checked_utc'],'zip_SHA':sha(d/'local_observation_snapshot.zip'),'members':len(files),'formal100_complete':False,'final_weights_downloaded':False,'CPU_forward':False,'final_TEST':False}
(d/'actual_local_D_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
