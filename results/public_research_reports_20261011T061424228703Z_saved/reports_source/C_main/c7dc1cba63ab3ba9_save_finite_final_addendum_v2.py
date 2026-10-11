from pathlib import Path
import json,datetime,hashlib,shutil
rpath=Path('outputs/有限任务风险研究资料与接续永久保存核验.json');r=json.loads(rpath.read_text(encoding='utf-8'));dest=Path(r['directory']);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,x in r['files'].items():
 assert sha(Path(name))==sha(dest/name)
 if name!='outputs\\研究接续状态.md' and name!='outputs/研究接续状态.md':assert sha(Path(name))==x['sha256']
paths=[Path('outputs/研究接续状态.md'),Path('work/new_p4_checks/finite_final_addendum_20261005T1544Z.json'),Path('work/save_finite_final_addendum_v1.py'),Path('work/save_finite_final_addendum_v2.py')]
for path in paths:
 assert not path.is_absolute();d=dest/path;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,d);assert sha(path)==sha(d);r['files'][str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
r['last_addendum_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();r['addendum_repair']='First addendum copied state and evidence but source __file__ absolute joined back into its executing original Windows path and CopyFile2 rejected it. Original v1 kept; v2 uses explicit relative source path, then revalidates every record.'
rpath.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(rpath,dest/'backup_receipt.json')
for name,x in r['files'].items():assert sha(Path(name))==x['sha256']==sha(dest/name)
print('FINAL_ADDENDUM_ALL_RECORDS_SHA_VERIFIED',len(r['files']))
