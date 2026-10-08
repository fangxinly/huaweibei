"""Reversible NTFS compression; preserve every checkpoint path and byte SHA."""
import ctypes,datetime,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path
base=Path('D:/CodexBackups/selective_flow_20261003_1105').resolve()
root=base/('retained_checkpoint_lossless_storage_'+sys.argv[1]);root.mkdir()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.GetCompressedFileSizeW.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)]
kernel.GetCompressedFileSizeW.restype=ctypes.c_ulong
def allocated(p):
 high=ctypes.c_ulong();low=kernel.GetCompressedFileSizeW(str(p),ctypes.byref(high))
 if low==0xffffffff and ctypes.get_last_error():raise ctypes.WinError(ctypes.get_last_error())
 return (high.value<<32)|low
targets=sorted([p for p in base.rglob('*.pt') if p.stat().st_size>1500000000],key=lambda p:p.stat().st_size,reverse=True)
before=shutil.disk_usage(base).free;records=[]
for i,p in enumerate(targets):
 assert p.resolve().is_relative_to(base) and p.is_file()
 entry={'path':str(p),'logical_bytes':p.stat().st_size,'before_sha256':sha(p),'before_allocated':allocated(p),'actual_start_utc':utc()}
 argv=['compact.exe','/C','/I','/Q',str(p)]
 child=subprocess.run(argv,capture_output=True)
 entry.update(fullargv=argv,natural_exit=child.returncode,stdout=child.stdout.decode('utf8',errors='replace'),stderr=child.stderr.decode('utf8',errors='replace'))
 assert child.returncode==0
 entry.update(after_sha256=sha(p),after_allocated=allocated(p),actual_finish_utc=utc())
 assert entry['after_sha256']==entry['before_sha256'] and p.stat().st_size==entry['logical_bytes']
 records.append(entry)
 (root/f'file_{i:02d}.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps({'file':str(p),'SHA_unchanged':True,'saved_allocated_bytes':entry['before_allocated']-entry['after_allocated'],'D_free_bytes':shutil.disk_usage(base).free}),flush=True)
 if shutil.disk_usage(base).free>=12*1024**3:break
result={'status':'RETAINED_ALL_PATHS_AND_CONTENT_BYTES_NTFS_COMPRESSION_VERIFIED','actual_finish_utc':utc(),'files':records,'D_free_before':before,'D_free_after':shutil.disk_usage(base).free,'no_file_deleted':True,'no_original_content_bytes_changed':True}
(root/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
Path('outputs/保留原件无损压缩实际接续.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'status':result['status'],'D_free_after':result['D_free_after'],'D_record':str(root)}),flush=True)
