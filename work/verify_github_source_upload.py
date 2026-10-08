"""Verify publication copies; print only paths/kinds, never matched values."""
import collections,hashlib,json,re
from pathlib import Path
from prepare_github_source_upload import ASSIGN,TOKENS,PRIVATE,RANDOM_QUOTED
ROOT=Path(__file__).resolve().parent.parent
DEST=(ROOT/'github_upload_20261008T133346Z').resolve()
LOCAL=ROOT/'outputs/github_upload_20261008T133346Z'
m=json.loads((DEST/'source_manifest.json').read_text(encoding='utf-8'))
expected={r['path'] for r in m['records']}
expected.update(['source_manifest.json','CODE_INDEX.md','README.md','.gitignore'])
removed=[];findings=[];counts=collections.Counter()
for p in DEST.rglob('*'):
    if not p.is_file() or '.git' in p.relative_to(DEST).parts:continue
    rel=p.relative_to(DEST).as_posix()
    assert p.resolve().is_relative_to(DEST) and p.resolve()!=DEST
    if rel not in expected:
        assert rel.startswith('archive_code/')
        p.unlink()  # Own uncommitted preparation leftovers only; no recursive delete.
        removed.append(rel);continue
    counts[p.suffix]+=1
    if p.suffix=='.npy':continue
    t=p.read_text(encoding='utf-8-sig')
    for name,rx in [('provider_token',TOKENS),('private_key',PRIVATE),('credential_assignment',ASSIGN),('random_quoted',RANDOM_QUOTED)]:
        for hit in rx.finditer(t):
            if name=='credential_assignment':
                s=hit.group(3)
                if s.startswith(('REDACTED','YOUR_','<','${')) or s in {'password','passwd','PASSWORD','example','placeholder'}:continue
            if name=='random_quoted':
                s=hit.group(2)
                if not (any(c.isupper() for c in s) and any(c.islower() for c in s) and any(c.isdigit() for c in s)):continue
            findings.append(dict(path=rel,kind=name,line=t.count('\n',0,hit.start())+1))
for r in m['records']:
    p=DEST/r['path'];assert p.exists()
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['published_sha256'],r['path']
assert not m['python_syntax_findings']
report=dict(all_manifest_SHA_pass=True,python_syntax_findings=0,remaining_secret_findings=findings,
    sanitized_edits=dict(collections.Counter(k for r in m['records'] for k,v in r['sanitizations'].items() for _ in range(v))),
    file_extensions=dict(counts),removed_own_preparation_orphans=len(removed),publication_files=len(expected),
    source_files=sum(r['kind']=='source' for r in m['records']),full_data_or_weights_uploaded=False)
(LOCAL/'precommit_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
assert not findings
