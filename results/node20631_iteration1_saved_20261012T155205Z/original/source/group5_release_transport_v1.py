"""Streaming immutable originals to Release ranges, with full download restoration.

Default commands only inspect local bytes. Publication never deletes or replaces
assets. HTTP failures are recorded and surfaced; no hidden upload retries.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BLOCK = 8 * 1024**2
PART = 512 * 1024**2


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    tmp.replace(path)


def blocks(stream, length=None):
    while length is None or length:
        data = stream.read(BLOCK if length is None else min(BLOCK, length))
        if not data:
            if length:
                raise EOFError('Declared source range is incomplete')
            return
        yield data
        if length is not None:
            length -= len(data)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in blocks(stream):
            h.update(data)
    return h.hexdigest()


def plan(path, name, part_bytes=PART):
    path = Path(path)
    if not path.is_file() or not 0 < part_bytes < 2*1024**3 or not name or '/' in name or '\\' in name:
        raise ValueError('Invalid original or part naming/size')
    size = path.stat().st_size
    if size == 0:
        raise ValueError('Empty original')
    h = hashlib.sha256()
    parts = []
    with path.open('rb') as stream:
        offset = 0
        while offset < size:
            length = min(part_bytes, size-offset)
            ph = hashlib.sha256()
            for data in blocks(stream, length):
                h.update(data); ph.update(data)
            parts.append(dict(name=f'{name}.part{len(parts):04d}', offset=offset, bytes=length, SHA=ph.hexdigest()))
            offset += length
    if len(parts) > 1000:
        raise ValueError('Too many parts')
    return dict(schema='GROUP5_CONTIGUOUS_ORIGINAL_V1', path=str(path), bytes=size, whole_SHA=h.hexdigest(), parts=parts)


def verify_zip(path, manifest_name='member_manifest.json'):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP member')
        manifest = json.loads(z.read(manifest_name))
        if set(names) != {r['name'] for r in manifest}|{manifest_name}:
            raise ValueError('ZIP exact member inventory differs')
        for row in manifest:
            n = row['name']
            if Path(n).is_absolute() or '..' in Path(n).parts or '\\' in n or ':' in n:
                raise ValueError('Unsafe member name')
            h = hashlib.sha256(); size = 0
            with z.open(n) as stream:
                for data in blocks(stream):
                    h.update(data); size += len(data)
            # ZipExtFile checks member CRC at EOF, including the manifest itself.
            if size != row['bytes'] or h.hexdigest() != row['sha256']:
                raise ValueError('ZIP member SHA/size differs: '+n)
    return dict(all_member_SHA_CRC_unique_exact_set_passed=True, members=len(manifest))


def seal(root, archive):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if archive.exists() or archive.parent != root:
        raise ValueError('Fresh archive inside the exact capture root required')
    files = sorted(p for p in root.rglob('*') if p.is_file())
    manifest = []
    for p in files:
        if p.is_symlink():
            raise ValueError('Symbolic source is not an immutable capture')
        manifest.append(dict(name=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=digest(p)))
    if any(r['name']=='member_manifest.json' for r in manifest):
        raise ValueError('Reserved manifest filename')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p, row in zip(files, manifest):
            z.write(p, row['name'])
        z.writestr('member_manifest.json', json.dumps(manifest, ensure_ascii=False).encode())
    proof = verify_zip(archive)
    return dict(actual_UTC=utc(), archive=str(archive), archive_SHA=digest(archive), bytes=archive.stat().st_size, **proof)


def asset_inventory(url, headers):
    from publish_release_assets_v1 import request
    result=[]; page=1
    while True:
        rows=request(url+f'?per_page=100&page={page}', headers)
        result.extend(rows)
        if len(rows)<100:
            break
        page+=1
    if len({a['name'] for a in result})!=len(result):
        raise ValueError('Ambiguous Release asset names')
    return result


def upload(manifest, receipt):
    from publish_release_assets_v1 import credential_headers, request
    from group5_test_selected_contract_v1 import validate_release_parts
    source=Path(manifest['path'])
    if source.stat().st_size!=manifest['bytes'] or digest(source)!=manifest['whole_SHA']:
        raise ValueError('Frozen original changed')
    headers=credential_headers()
    release=request('https://api.github.com/repos/fangxinly/huaweibei/releases/tags/autonomous-flow-health-20261008', headers)
    inventory=asset_inventory(release['assets_url'], headers)
    by_name={a['name']:a for a in inventory}
    if len(inventory)+sum(p['name'] not in by_name for p in manifest['parts'])>1000:
        raise PermissionError('Release asset capacity absent')
    state=dict(status='RANGE_PUBLICATION_STARTED', actual_UTC=utc(), whole_SHA=manifest['whole_SHA'], parts=[])
    write(receipt,state)
    try:
        with source.open('rb') as stream:
            for part in manifest['parts']:
                stream.seek(part['offset'])
                remote=by_name.get(part['name'])
                if remote is None:
                    h=hashlib.sha256(); sent=0
                    def body():
                        nonlocal sent
                        for data in blocks(stream,part['bytes']):
                            h.update(data);sent+=len(data);yield data
                    url=release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(part['name'])
                    req=urllib.request.Request(url,data=body(),method='POST',headers=dict(headers, **{
                        'Content-Type':'application/octet-stream','Content-Length':str(part['bytes'])}))
                    with urllib.request.urlopen(req,timeout=180) as response:
                        remote=json.load(response)
                    if sent!=part['bytes'] or h.hexdigest()!=part['SHA']:
                        raise ValueError('Uploaded range differs from frozen bytes')
                row=dict(name=part['name'],bytes=remote['size'],digest=remote.get('digest'),state=remote['state'],
                         id=remote['id'],url=remote['browser_download_url'],offset=part['offset'])
                if row['state']!='uploaded' or row['bytes']!=part['bytes'] or row['digest']!='sha256:'+part['SHA']:
                    raise ValueError('Remote digest/size/state differs')
                state['parts'].append(row);write(receipt,state)
                print(json.dumps(dict(name=part['name'],remote_digest_verified=True)),flush=True)
        validate_release_parts(manifest,state['parts'])
        # Inspect fresh publication metadata, including all pages.
        final={a['id']:a for a in asset_inventory(release['assets_url'],headers)}
        for row in state['parts']:
            if final[row['id']]['digest']!=row['digest'] or final[row['id']]['size']!=row['bytes']:
                raise ValueError('Final Release inventory changed')
        if digest(source)!=manifest['whole_SHA']:
            raise ValueError('Original changed during publication')
        state.update(status='ALL_RANGES_REMOTE_DIGEST_VERIFIED_RESTORE_PENDING',actual_UTC=utc())
        write(receipt,state)
        return state
    except Exception as exc:
        state.update(status='RANGE_PUBLICATION_FAILED_ORIGINAL_PRESERVED',actual_UTC=utc(),error_type=type(exc).__name__)
        write(receipt,state)
        raise RuntimeError(state['status']) from None


def restore(manifest, receipt, destination, opener=None):
    from group5_test_selected_contract_v1 import validate_release_parts
    validate_release_parts(manifest,receipt['parts'])
    if receipt['whole_SHA']!=manifest['whole_SHA']:
        raise ValueError('Publication refers to a different original')
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256(); size=0
    by_name={r['name']:r for r in receipt['parts']}
    with destination.open('xb') as target:
        for part in manifest['parts']:
            row=by_name[part['name']];ph=hashlib.sha256();length=0
            url=row['url']
            if opener is None:
                if not url.startswith('https://github.com/fangxinly/huaweibei/releases/download/'):
                    raise ValueError('Unexpected original download location')
                stream=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Research-original-restore'}),timeout=180)
            else:
                stream=opener(url)
            with stream:
                for data in blocks(stream):
                    length+=len(data)
                    if length>part['bytes']:
                        raise ValueError('Downloaded range exceeds declared length')
                    target.write(data);ph.update(data);h.update(data);size+=len(data)
            if length!=part['bytes'] or ph.hexdigest()!=part['SHA']:
                raise ValueError('Downloaded range SHA/length differs')
        target.flush();os.fsync(target.fileno())
    if size!=manifest['bytes'] or h.hexdigest()!=manifest['whole_SHA']:
        raise ValueError('Restored full original SHA differs')
    proof=verify_zip(destination)
    return dict(actual_UTC=utc(),status='RELEASE_FULL_ORIGINAL_RESTORED_SHA_ZIP_VERIFIED',
                destination=str(destination),whole_SHA=h.hexdigest(),bytes=size,**proof)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','upload','restore'])
    p.add_argument('--original',type=Path);p.add_argument('--name');p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--receipt',type=Path);p.add_argument('--destination',type=Path);p.add_argument('--proof',type=Path)
    a=p.parse_args()
    if a.action=='plan':write(a.manifest,plan(a.original,a.name))
    elif a.action=='upload':upload(json.loads(a.manifest.read_bytes()),a.receipt)
    else:write(a.proof,restore(json.loads(a.manifest.read_bytes()),json.loads(a.receipt.read_bytes()),a.destination))
