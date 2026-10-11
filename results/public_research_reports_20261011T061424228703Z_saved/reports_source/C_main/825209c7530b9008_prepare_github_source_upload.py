"""Build a source-only Git checkout. Original frozen evidence is read-only."""
import ast,collections,hashlib,json,re,shutil,zipfile
from pathlib import Path,PurePosixPath
from github_source_inventory import CODE
ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'github_upload_20261008T133346Z'
LOCAL=ROOT/'outputs/github_upload_20261008T133346Z'
CONFIG=re.compile(r'(?i)(?:^|[_-])(plan|protocol|config|schema)(?:[_-]|\.)')
HOST=re.compile(r'[A-Za-z0-9]+snow\.deepln\.com')
RANDOM_QUOTED=re.compile(r'([\"\'])([A-Za-z0-9]{16})\1')
ASSIGN=re.compile(r'(?i)((?:password|passwd|pwd|api_key|access_token|auth_token|secret_key)\s*[:=]\s*)([\"\'])([^\"\'\r\n]{6,})(\2)')
TOKENS=re.compile(r'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})')
PRIVATE=re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')

def h(b):return hashlib.sha256(b).hexdigest()
def sanitize(b):
    t=b.decode('utf-8-sig');changes=collections.Counter()
    def host(m):changes['server_endpoint']+=1;return 'REDACTED_SERVER_HOST.invalid'
    t=HOST.sub(host,t)
    def assign(m):
        value=m.group(3)
        if value.startswith(('REDACTED','YOUR_','<','${')) or value in {'password','passwd','PASSWORD','example','placeholder'}:return m.group()
        changes['credential_assignment']+=1
        return m.group(1)+m.group(2)+'REDACTED_SECRET_REQUIRE_ENV'+m.group(4)
    t=ASSIGN.sub(assign,t)
    def token(m):changes['provider_token']+=1;return 'REDACTED_PROVIDER_TOKEN'
    t=TOKENS.sub(token,t)
    def random(m):
        value=m.group(2)
        if any(c.isupper() for c in value) and any(c.islower() for c in value) and any(c.isdigit() for c in value):
            changes['possible_random_credential']+=1
            return m.group(1)+'REDACTED_SECRET_REQUIRE_ENV'+m.group(1)
        return m.group()
    t=RANDOM_QUOTED.sub(random,t)
    if PRIVATE.search(t) or re.search(r'sshpass\s+-p\s+\S+',t):raise ValueError('manual secret review required; value omitted')
    return (t.encode('utf-8') if changes else b),dict(changes)

def actual_labels(obj,key=''):
    if isinstance(obj,dict):return any(actual_labels(v,str(k)) for k,v in obj.items())
    if isinstance(obj,list):
        if key.lower() in {'y','truth','labels','true_labels','y_true','targets'} and any(isinstance(v,(int,float)) for v in obj):return True
        return any(actual_labels(v,key) for v in obj)
    return False

def main():
    assert (DEST/'.git').is_dir()
    assert not (DEST/'source_manifest.json').exists()
    LOCAL.mkdir(exist_ok=True)
    records=[];archive_refs=[];excluded=[];original_map={};syntax=[]
    def put(b,relative,origin,kind):
        relative=PurePosixPath(relative)
        assert not relative.is_absolute() and '..' not in relative.parts
        if len(b)>5*1024*1024:raise ValueError('source over 5MiB: '+str(relative))
        original=h(b)
        if kind=='generated_order':clean,edits=b,{}
        else:clean,edits=sanitize(b)
        p=DEST/str(relative);p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or p.read_bytes()!=clean:p.write_bytes(clean)
        if p.suffix=='.py':
            try:ast.parse(clean.decode('utf-8-sig'),filename=str(relative))
            except SyntaxError as e:syntax.append(dict(path=str(relative),line=e.lineno,reason=e.msg))
        records.append(dict(path=str(relative),origin=origin,kind=kind,bytes=len(clean),
                            original_sha256=original,published_sha256=h(clean),sanitizations=edits))
        original_map.setdefault(original,str(relative))
    for top in ('work','outputs'):
        for p in sorted((ROOT/top).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            rel=p.relative_to(ROOT).as_posix()
            if p.suffix.lower() in CODE:put(p.read_bytes(),rel,rel,'source')
            elif p.name=='offline_requirements.txt':put(p.read_bytes(),rel,rel,'dependency_lock')
            elif p.name=='prospective_common_TRAIN_position_orders_seed128_100.npy':put(p.read_bytes(),rel,rel,'generated_order')
            elif top=='work' and p.suffix=='.json' and CONFIG.search(p.name):
                try:data=json.loads(p.read_text(encoding='utf-8-sig'))
                except (UnicodeDecodeError,json.JSONDecodeError):excluded.append(dict(path=rel,reason='invalid config JSON'));continue
                if actual_labels(data):excluded.append(dict(path=rel,reason='contains numeric labels'));continue
                put(p.read_bytes(),rel,rel,'historical_configuration')
    inventory=json.loads((ROOT/'outputs/github_upload_inventory_20261008T133346Z.json').read_text(encoding='utf-8'))
    for archive_number,row in enumerate(inventory['archives']):
        if row.get('error'):excluded.append(dict(path=row['path'],reason=row['error']));continue
        p=ROOT/row['path']
        with zipfile.ZipFile(p) as z:
            for entry in z.infolist():
                member=PurePosixPath(entry.filename.replace('\\','/'))
                if entry.is_dir() or '__pycache__' in member.parts:continue
                suffix=member.suffix.lower()
                kind='source' if suffix in CODE else 'historical_configuration' if suffix=='.json' and CONFIG.search(member.name) else None
                if kind is None:continue
                b=z.read(entry)
                if kind=='historical_configuration':
                    try:data=json.loads(b.decode('utf-8-sig'))
                    except (UnicodeDecodeError,json.JSONDecodeError):continue
                    if actual_labels(data):excluded.append(dict(path=row['path']+'::'+str(member),reason='contains numeric labels'));continue
                origin=row['path']+'::'+str(member)
                original=h(b)
                if original in original_map:archive_refs.append(dict(origin=origin,published_path=original_map[original],original_sha256=original));continue
                # Keep sibling imports together without exceeding Windows path limits.
                group=h(str(member.parent).encode())[:12]
                relative=f'archive_code/z{archive_number:03d}/{group}/{member.name}'
                put(b,relative,origin,kind)
    docs=[
        ('outputs/official_upgrade_completion_20261008T112205Z/官方对齐改进完成总结.md','docs/official_VAL_TEST_results.md'),
        ('outputs/flexible_official_calibration_20261008T125206Z/官方VAL_TEST与非线性校准说明.md','docs/flexible_calibration_review.md'),
        ('outputs/coupling_residual_math_review_20261008T120037Z/耦合与任务残差几何核对.md','docs/coupling_residual_math_review.md'),
        ('outputs/polarity_intensity_review_20261008T130609Z/REPORT.md','docs/polarity_intensity_review.md'),
    ]
    for rel,target in docs:
        p=ROOT/rel
        if p.exists():put(p.read_bytes(),target,rel,'selected_documentation')
    requirements=(ROOT/'work/second_lease_linux_wheels_20261006t1401z/offline_requirements.txt').read_bytes()
    put(requirements,'requirements-research-lock.txt','work/second_lease_linux_wheels_20261006t1401z/offline_requirements.txt','dependency_lock')
    manifest=dict(scope='All unpacked workspace source plus unique source/config versions from local archives',
                  records=records,duplicate_archive_references=archive_refs,excluded_configurations=excluded,
                  excluded_binary_types=['model checkpoints','datasets and labels','predictions','raw logs','wheel files','bytecode','base64 payloads'],
                  source_guard_note='Historical plans retain original source hashes; sanitization requires a new explicit source/config freeze before execution.',
                  python_syntax_findings=syntax,local_originals_modified=False)
    (DEST/'source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    groups=collections.Counter('/'.join(PurePosixPath(r['path']).parts[:-1]) for r in records)
    lines=['# 源码目录索引','','完整逐文件来源和SHA见 `source_manifest.json`。','','|目录|文件数|','|---|---:|']
    for folder,count in sorted(groups.items()):lines.append(f'|[{folder}]({folder}/)|{count}|')
    (DEST/'CODE_INDEX.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (LOCAL/'preparation_result.json').write_text(json.dumps(dict(files=len(records),bytes=sum(r['bytes'] for r in records),
        archive_duplicate_refs=len(archive_refs),sanitized_files=[dict(path=r['path'],edits=r['sanitizations']) for r in records if r['sanitizations']],
        syntax_findings=syntax,excluded_configurations=excluded),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(records),bytes=sum(r['bytes'] for r in records),archive_duplicate_refs=len(archive_refs),
        sanitized_files=sum(bool(r['sanitizations']) for r in records),syntax_findings=syntax,excluded_configs=len(excluded)),ensure_ascii=False))

if __name__=='__main__':main()
