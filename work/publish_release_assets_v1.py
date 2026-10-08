"""Stream explicit SHA-verified assets to a GitHub draft release; no local deletion.

Credential helper access requires --publish. Credentials stay in memory and are
never placed in a plan, receipt, URL or command. Import/default invocation is inert.
Large ZIP members can be published in ranges without extracting onto a full disk.
"""
import argparse
import base64
import contextlib
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


@contextlib.contextmanager
def open_source(asset):
    if 'parts' in asset:
        from ingest_split_capture import Parts
        with Parts([Path(p) for p in asset['parts']]) as raw, zipfile.ZipFile(raw) as archive:
            with archive.open(asset['member']) as stream:
                yield stream
    elif 'archive' in asset:
        with zipfile.ZipFile(asset['archive']) as archive, archive.open(asset['member']) as stream:
            yield stream
    else:
        with Path(asset['path']).open('rb') as stream:
            yield stream


def blocks(stream, length=None):
    while length is None or length:
        block = stream.read(8 * 1024**2 if length is None else min(length, 8 * 1024**2))
        if not block:
            if length:
                raise EOFError('Source shorter than declared range')
            break
        yield block
        if length is not None:
            length -= len(block)


def verify_source(asset):
    digest = hashlib.sha256()
    size = 0
    with open_source(asset) as stream:
        for block in blocks(stream):
            digest.update(block)
            size += len(block)
    if size != asset['bytes'] or digest.hexdigest() != asset['sha256']:
        raise ValueError('Source size/SHA mismatch: ' + asset['name'])


def credential_headers():
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
    result = subprocess.run(['git', '-c', 'credential.interactive=never', 'credential', 'fill'],
        input='protocol=https\nhost=github.com\n\n', text=True, capture_output=True, env=env, timeout=25)
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode or not fields.get('password'):
        raise RuntimeError('GitHub credential helper unavailable')
    key = base64.b64encode((fields.get('username', '') + ':' + fields['password']).encode()).decode()
    return {'Authorization': 'Basic ' + key, 'User-Agent': 'Research-asset-publication',
            'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}


def request(url, headers, payload=None, method=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=dict(headers, **({'Content-Type': 'application/json'} if body else {})), method=method)
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def run(plan, receipt):
    if receipt.exists():
        state = json.loads(receipt.read_text(encoding='utf-8'))
        if state['repository'] != plan['repository'] or state['tag'] != plan['tag']:
            raise ValueError('Receipt identity mismatch')
    else:
        state = {'repository': plan['repository'], 'tag': plan['tag'], 'assets': [], 'status': 'VERIFYING_SOURCES'}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    def save():
        receipt.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    headers = credential_headers()
    base = 'https://api.github.com/repos/' + plan['repository']
    releases = request(base + '/releases?per_page=100', headers)
    release = next((r for r in releases if r['tag_name'] == plan['tag']), None)
    if release is None:
        release = request(base + '/releases', headers, dict(tag_name=plan['tag'], target_commitish=plan['commit'],
            name=plan['title'], body=plan['body'], draft=True))
    state.update(release_id=release['id'], release_url=release['html_url'], status='UPLOADING_DRAFT')
    save()
    try:
        for asset in plan['assets']:
            verify_source(asset)
            offset = 0
            chunk_size = asset.get('chunk_bytes', 1_600_000_000)
            with open_source(asset) as stream:
                while offset < asset['bytes']:
                    length = min(chunk_size, asset['bytes'] - offset)
                    name = asset['name'] if asset['bytes'] <= chunk_size else asset['name'] + f'.part{offset // chunk_size:02d}'
                    old = next((a for a in request(release['assets_url'] + '?per_page=100', headers) if a['name'] == name), None)
                    if old is not None:
                        digest = hashlib.sha256()
                        for block in blocks(stream, length):
                            digest.update(block)
                        if old['state'] != 'uploaded' or old['size'] != length or old.get('digest') != 'sha256:' + digest.hexdigest():
                            raise ValueError('Existing remote asset cannot be verified: ' + name)
                        uploaded = old
                    else:
                        digest = hashlib.sha256()
                        sent = 0
                        last_log = 0
                        def body():
                            nonlocal sent, last_log
                            for block in blocks(stream, length):
                                digest.update(block)
                                sent += len(block)
                                yield block
                                if sent - last_log >= 128 * 1024**2:
                                    print(json.dumps(dict(asset=name, sent_bytes=sent, total_bytes=length)), flush=True)
                                    last_log = sent
                        url = release['upload_url'].split('{')[0] + '?name=' + urllib.parse.quote(name)
                        req = urllib.request.Request(url, data=body(), headers=dict(headers,
                            **{'Content-Type': 'application/octet-stream', 'Content-Length': str(length)}), method='POST')
                        with urllib.request.urlopen(req, timeout=180) as response:
                            uploaded = json.load(response)
                        if sent != length or uploaded['state'] != 'uploaded' or uploaded['size'] != length:
                            raise ValueError('Remote size/state mismatch: ' + name)
                        if uploaded.get('digest') != 'sha256:' + digest.hexdigest():
                            raise ValueError('Remote SHA256 not verified: ' + name)
                    row = dict(name=name, bytes=length, sha256=digest.hexdigest(), source_sha256=asset['sha256'],
                               source_offset=offset, url=uploaded['browser_download_url'], id=uploaded['id'], remote_digest_verified=True)
                    state['assets'] = [r for r in state['assets'] if r['name'] != name] + [row]
                    save()
                    print(json.dumps(dict(status='ASSET_REMOTE_SHA_VERIFIED', name=name, bytes=length)), flush=True)
                    offset += length
        release = request(base + '/releases/' + str(release['id']), headers, {'draft': False}, method='PATCH')
        state.update(status='PUBLISHED_ALL_ASSETS_REMOTE_SHA_VERIFIED', release_url=release['html_url'])
        save()
        print(json.dumps(dict(status=state['status'], url=state['release_url'], assets=len(state['assets']))), flush=True)
    except Exception as exc:
        state.update(status='FAILED_OR_PARTIAL_DRAFT_PRESERVED', error_type=type(exc).__name__)
        save()
        raise RuntimeError('Asset publication failed; inspect private receipt; no source was deleted') from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if not args.publish:
        print(json.dumps(dict(status='SKIPPED_NO_PUBLICATION_REQUEST', credential_access=False)))
        return
    if args.plan is None or args.receipt is None:
        parser.error('--publish requires --plan and --receipt')
    run(json.loads(args.plan.read_text(encoding='utf-8')), args.receipt)


if __name__ == '__main__':
    main()
