"""Create and verify a recovery copy; never deletes the source."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference_first10_actual_20261006T151954Z/a/selected_best_full.pt').resolve(strict=True)
dest=root/'work/checkpoint_recovery_20261008T170308Z/selected_best_full.pt'
dest.parent.mkdir(exist_ok=False)
assert source.is_relative_to(Path('D:/CodexBackups/selective_flow_20261003_1105').resolve())
assert shutil.disk_usage(dest.parent).free>source.stat().st_size+300000000
expected='6949dccb34e93ab4642a7b670844ca41c94524e479ab32456299c118e28711fc'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(source)==expected
shutil.copyfile(source,dest)
assert sha(dest)==expected and source.stat().st_size==dest.stat().st_size
receipt=dict(status='EXACT_RECOVERY_COPY_ORIGINAL_SHA_VERIFIED_NO_DELETION',
    actualclock_before_copy_UTC='2026-10-08 17:03:08 UTC',source=str(source),destination=str(dest),
    original_SHA256=expected,bytes=dest.stat().st_size,source_still_present=source.exists(),
    source_deleted=False,permission_to_remove_D_duplicate_pending=True,
    prior_user_constraint='Do not delete or modify old frozen materials; four other exact files only were authorized for removal')
(root/'outputs/retention_compression_20261008T165711Z/exact_recovery_copy.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(dict(status=receipt['status'],bytes=receipt['bytes'],source_deleted=False)))
