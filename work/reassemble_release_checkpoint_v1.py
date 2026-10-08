"""Restore one checkpoint from verified release ranges; never overwrite a file."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore(manifest, folder, name, target):
    rows = sorted((r for r in manifest['assets'] if r['name'].startswith(name + '.part')), key=lambda r: r['source_offset'])
    if not rows or len({r['source_sha256'] for r in rows}) != 1:
        raise ValueError('Missing or inconsistent ranges')
    offset = 0
    for row in rows:
        part = folder / row['name']
        if offset != row['source_offset'] or part.stat().st_size != row['bytes'] or sha(part) != row['sha256']:
            raise ValueError('Part identity/size/SHA mismatch')
        offset += row['bytes']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or shutil.disk_usage(target.parent).free < offset + 500_000_000:
        raise ValueError('Destination must be new with full checkpoint plus 500MB free')
    with target.open('xb') as output:
        for row in rows:
            with (folder / row['name']).open('rb') as source:
                shutil.copyfileobj(source, output, 8 * 1024**2)
    if target.stat().st_size != offset or sha(target) != rows[0]['source_sha256']:
        raise ValueError('Restored checkpoint failed SHA; preserve for diagnosis')
    print(json.dumps(dict(status='RESTORED_EXACT_ORIGINAL_BYTES', bytes=offset, sha256=rows[0]['source_sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('parts_folder', type=Path)
    parser.add_argument('checkpoint_name')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    restore(json.loads(args.manifest.read_text(encoding='utf-8')), args.parts_folder, args.checkpoint_name, args.output)
