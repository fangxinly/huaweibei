"""Copy existing source/result members from large archives. Never decode arrays."""
import collections
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import build_complete_research_archive as base


def main():
    out = base.WS / 'outputs' / ('research_archive_members_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    secrets = base.gather_secrets(base.discover())
    redactions = 0

    def redact(s):
        nonlocal redactions
        old = s
        for secret in secrets:
            redactions += s.count(secret)
            s = s.replace(secret, '[REDACTED_CREDENTIAL]')
        s,n = base.TOKEN.subn('[REDACTED_TOKEN]',s)
        redactions += n
        return s, s != old

    sources = []
    for root_id, root in base.ROOTS.items():
        for d, dirs, names in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in base.SKIP_DIR and not x.startswith(('research_linux_handoff','complete_research_archive','research_archive_members','complete_research_transfer')))
            for name in sorted(names):
                p = Path(d)/name
                if p.suffix.lower() == '.zip' and p.stat().st_size > base.MAX_ZIP:
                    sources.append((root_id,p))
    target = out/'archived_members_supplement.zip'
    mapping=[]; objects={}; manifest=[]; archives=[]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as dest:
        def put(name,data):
            dest.writestr(name,data)
            manifest.append({'path':name,'bytes':len(data),'SHA256':base.digest(data)})
        for i,(root_id,p) in enumerate(sources):
            assert shutil.disk_usage(out).free > base.MIN_FREE + 256*1024**2
            st=p.stat()
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
            item={'root':root_id,'original_archive_path':str(p),'archive_relative_path':p.relative_to(base.ROOTS[root_id]).as_posix(),'archive_bytes':st.st_size,'archive_SHA256':h.hexdigest(),'selected_members_verified':0,'unselected_members':0,'whole_archive_member_CRC_not_rerun':True}
            with zipfile.ZipFile(p) as src:
                names=src.namelist();assert len(names)==len(set(names))
                item['archive_unique_members_verified']=True;item['archive_members']=len(names)
                for m in src.infolist():
                    rel=PurePosixPath(m.filename.replace('\\','/'))
                    ext=rel.suffix.lower()
                    sensitive=rel.name.lower() in {'auth.json','credentials','credentials.json','.env','id_rsa','id_ed25519'} or ext in {'.pem','.key','.p12','.pfx'}
                    selected=not m.is_dir() and ext in base.TEXT_EXT|base.ARRAY_EXT and not sensitive
                    if not selected:
                        item['unselected_members']+=1;continue
                    assert not rel.is_absolute() and '..' not in rel.parts
                    data=src.read(m) # ZipFile verifies this selected member's original CRC.
                    q={'root':root_id,'archive_relative_path':item['archive_relative_path'],'archive_SHA256':item['archive_SHA256'],'original_member':m.filename,'member_CRC32':format(m.CRC,'08x'),'original_bytes':len(data),'original_SHA256':base.digest(data),'selected_original_member_CRC_verified':True}
                    changed=False
                    if ext in base.TEXT_EXT:
                        s,changed=redact(data.decode('utf-8',errors='replace'))
                        if changed:data=s.encode('utf-8')
                    dh=base.digest(data);name='archive_objects/'+dh[:2]+'/'+dh
                    if dh not in objects:
                        put(name,data);objects[dh]=len(data)
                    q.update(object=name,exported_SHA256=dh,exported_bytes=len(data),status='redacted_derivative' if changed else 'exact_original_member_copy')
                    mapping.append(q);item['selected_members_verified']+=1
            assert st.st_size==p.stat().st_size and st.st_mtime_ns==p.stat().st_mtime_ns
            archives.append(item)
            print('ARCHIVE_MEMBER_PROGRESS',i+1,'of',len(sources),'members',len(mapping),'unique_objects',len(objects),'ZIP_bytes',target.stat().st_size,flush=True)
        coverage={'actual_UTC':datetime.now(timezone.utc).isoformat(),'archives':archives,'selected_members':len(mapping),'unique_objects':len(objects),'unique_object_bytes':sum(objects.values()),'credential_redaction_occurrences':redactions,'no_array_numeric_decode':True,'no_models_loaded':True,'no_original_archives_modified':True,'no_scientific_execution':True}
        put('ARCHIVED_MEMBER_MAP.json',base.jsonbytes(mapping))
        put('ARCHIVED_MEMBER_COVERAGE.json',base.jsonbytes(coverage))
        put('RESTORE_ARCHIVE_MEMBERS.py',RESTORE.encode('utf-8'))
        put('VERIFY_ALL.py',VERIFY.encode('utf-8'))
        dest.writestr('ARCHIVED_MEMBER_MANIFEST.json',base.jsonbytes(manifest))
    with zipfile.ZipFile(target) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        assert set(z.namelist())=={m['path'] for m in manifest}|{'ARCHIVED_MEMBER_MANIFEST.json'}
        for q in manifest:
            b=z.read(q['path']);assert len(b)==q['bytes'] and base.digest(b)==q['SHA256']
        assert z.testzip() is None
    h=hashlib.sha256()
    with target.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    receipt={'actual_UTC':datetime.now(timezone.utc).isoformat(),'archive':str(target),'bytes':target.stat().st_size,'SHA256':h.hexdigest(),'ZIP_entries':len(manifest)+1,'all_exported_members_SHA256_CRC_unique_exactset_verified':True,'coverage':coverage}
    (out/'SUPPLEMENT_RECEIPT.json').write_bytes(base.jsonbytes(receipt))
    (out/(target.name+'.sha256')).write_text(h.hexdigest()+'  '+target.name+'\n',encoding='ascii')
    (base.WS/'work/research_archive_members_pointer.json').write_bytes(base.jsonbytes({k:v for k,v in receipt.items() if k!='coverage'}))
    print('SUPPLEMENT_COMPLETE',json.dumps({k:v for k,v in receipt.items() if k!='coverage'}),flush=True)


VERIFY='''import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
count=0
for mf in ['MANIFEST.json','ARCHIVED_MEMBER_MANIFEST.json']:
 for q in json.loads((root/mf).read_text(encoding='utf-8')):
  p=(root/q['path']).resolve();assert root.resolve() in p.parents
  h=hashlib.sha256();size=0
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b);size+=len(b)
  assert size==q['bytes'] and h.hexdigest()==q['SHA256'],q['path']
  count+=1
print('VERIFIED_ALL',count,'members. No research code executed.')
'''
RESTORE=VERIFY+'''import shutil
from pathlib import PurePosixPath
rows=json.loads((root/'ARCHIVED_MEMBER_MAP.json').read_text(encoding='utf-8'))
dest=root/'restored_archive_members';assert not dest.exists(),'Use a fresh destination.'
for q in rows:
 rel=PurePosixPath(q['archive_relative_path']);member=PurePosixPath(q['original_member'].replace('\\\\','/'))
 assert not rel.is_absolute() and not member.is_absolute() and '..' not in rel.parts and '..' not in member.parts
 p=dest/q['root']/Path(*rel.parts)/Path(*member.parts);assert dest.resolve() in p.resolve().parents
 p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/q['object'],p)
print('RESTORED_ARCHIVE_MEMBERS',len(rows),'existing files. No research code executed.')
'''


if __name__=='__main__':main()
