"""New version embeds actual capture source and SHA in the original ZIP."""
from pathlib import Path
p=Path(__file__).with_name('capture_staged_reference_v24.py')
s=p.read_text(encoding='utf-8')
needle="manifest={'actual_utc':"
insert="""tool=Path(__file__).resolve();before=tool.stat();digest=sha(tool);after=tool.stat()
assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
members['capture_tool/capture_staged_reference_v25.py']={'original_path':str(tool),'sha256':digest,'bytes':before.st_size,'inode':before.st_ino,'device':before.st_dev,'mtime_ns':before.st_mtime_ns,'stable_before_after':True}
"""
assert s.count(needle)==1;s=s.replace(needle,insert+needle)
q=p.with_name('capture_staged_reference_v25.py');assert not q.exists();q.write_text(s,encoding='utf-8');compile(s,str(q),'exec')
print(q)
