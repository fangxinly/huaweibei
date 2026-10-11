"""Read-only inventory. Never print matched secret values."""
import ast, collections, json, re, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
CODE={'.py','.ps1','.sh','.bash','.js','.ts','.tsx','.jsx','.toml','.yaml','.yml','.ini','.cfg','.bat','.cmd','.c','.cpp','.h','.cu','.ipynb'}
SECRETS=[
 ('credential_assignment',re.compile(r'(?i)(?:password|passwd|pwd|api_key|access_token|auth_token|secret_key)\s*[:=]\s*[\"\']([^\"\'\r\n]{6,})[\"\']')),
 ('provider_token',re.compile(r'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})')),
 ('private_key',re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')),
 ('sshpass',re.compile(r'sshpass\s+-p\s+\S+')),
]
def scan(text):
    out=[]
    for kind,rx in SECRETS:
        for m in rx.finditer(text):
            value=m.group(1) if kind=='credential_assignment' else m.group()
            if kind=='credential_assignment' and (value.startswith(('REDACTED','YOUR_','<','${')) or value in {'password','passwd','PASSWORD','example','placeholder'}):continue
            out.append(dict(kind=kind,line=text.count('\n',0,m.start())+1))
    for m in re.finditer(r'(?<![A-Za-z0-9])[A-Za-z0-9]{16}(?![A-Za-z0-9])',text):
        s=m.group()
        if any(c.isupper() for c in s) and any(c.islower() for c in s) and any(c.isdigit() for c in s):
            out.append(dict(kind='possible_random_credential',line=text.count('\n',0,m.start())+1))
    return out
def main():
    rows=[];archives=[];findings=[];total=0;imports=collections.Counter()
    for top in ('work','outputs'):
        for p in (ROOT/top).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            rel=p.relative_to(ROOT).as_posix()
            if p.suffix.lower() in CODE:
                b=p.read_bytes();t=b.decode('utf-8-sig',errors='replace');total+=len(b)
                rows.append(dict(path=rel,bytes=len(b)))
                hits=scan(t)
                if hits:findings.append(dict(path=rel,hits=hits))
                if p.suffix=='.py':
                    try:
                        for n in ast.walk(ast.parse(t)):
                            if isinstance(n,ast.Import):
                                for a in n.names:imports[a.name.split('.')[0]]+=1
                            elif isinstance(n,ast.ImportFrom) and n.module:imports[n.module.split('.')[0]]+=1
                    except SyntaxError:pass
            elif p.suffix=='.zip':
                try:
                    with zipfile.ZipFile(p) as z:
                        entries=[i for i in z.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() in CODE]
                        if entries:archives.append(dict(path=rel,code_members=len(entries),code_bytes=sum(i.file_size for i in entries)))
                except zipfile.BadZipFile:archives.append(dict(path=rel,error='BadZipFile'))
    out=dict(source_count=len(rows),source_bytes=total,sources=rows,secret_findings=findings,archives=archives,imports=imports.most_common(50))
    dest=ROOT/'outputs/github_upload_inventory_20261008T133346Z.json'
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(source_count=len(rows),source_bytes=total,secret_findings=findings,archives=archives,imports=imports.most_common(25)),ensure_ascii=False))
if __name__=='__main__':main()
