"""NTFS compression preserves original paths/content; no deletion or repacking.

Before/after content SHA256 are computed. Only an explicitly named original .pt
within the designated D backup root is accepted. No recursive mutation.
"""
import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def allocated(path):
    high = ctypes.c_ulong()
    fn = ctypes.windll.kernel32.GetCompressedFileSizeW
    fn.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    fn.restype = ctypes.c_ulong
    ctypes.set_last_error(0)
    low = fn(str(path), ctypes.byref(high))
    if low == 0xffffffff and ctypes.get_last_error(): raise ctypes.WinError()
    return (high.value << 32) | low


def run(a):
    root = Path('D:/CodexBackups/selective_flow_20261003_1105').resolve()
    path = a.path.resolve(strict=True)
    if not path.is_relative_to(root) or path.suffix != '.pt' or path.is_symlink():
        raise ValueError('Explicit regular checkpoint under fixed backup root required')
    if a.receipt.exists(): raise FileExistsError(a.receipt)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    before = dict(SHA256=sha(path), bytes=path.stat().st_size,
                  allocated_bytes=allocated(path), D_free_bytes=shutil.disk_usage(root).free)
    pre = dict(status='BEFORE_CONTENT_VERIFIED_COMPRESSION_NOT_COMPLETE', path=str(path),
               actualclock_before_dispatch_UTC=a.clock, pid=os.getpid(), fullargv=[sys.executable]+sys.argv,
               before=before, file_deleted=False, content_rewritten=False)
    a.receipt.with_suffix('.before.json').write_text(json.dumps(pre, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=pre['status'], logical_bytes=before['bytes'], allocated_bytes=before['allocated_bytes'])), flush=True)
    tick = time.monotonic()
    proc = subprocess.run(['compact.exe', '/C', '/A', '/I', '/Q', str(path)], capture_output=True)
    after = dict(SHA256=sha(path), bytes=path.stat().st_size,
                 allocated_bytes=allocated(path), D_free_bytes=shutil.disk_usage(root).free)
    valid = before['SHA256'] == after['SHA256'] and before['bytes'] == after['bytes']
    rec = dict(pre, status='NTFS_COMPRESSION_COMPLETE_CONTENT_SHA_UNCHANGED' if valid and proc.returncode == 0 else 'COMPRESSION_CHECK_FAILED',
               command=['compact.exe', '/C', '/A', '/I', '/Q', str(path)], natural_exit_code=proc.returncode,
               after=after, saved_allocated_bytes=before['allocated_bytes']-after['allocated_bytes'],
               content_SHA_unchanged=valid, elapsed_seconds=time.monotonic()-tick,
               stdout=proc.stdout.decode('utf-8', errors='replace'), stderr=proc.stderr.decode('utf-8', errors='replace'))
    a.receipt.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: rec[k] for k in ('status','natural_exit_code','saved_allocated_bytes','content_SHA_unchanged')}), flush=True)
    if not valid or proc.returncode: raise RuntimeError('Compression audit failed; preserve receipt')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--path', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--clock', required=True)
    run(p.parse_args())
