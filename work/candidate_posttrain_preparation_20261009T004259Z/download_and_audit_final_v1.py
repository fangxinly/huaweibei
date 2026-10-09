"""One complete public archive; linear member verification, selected extraction."""
import argparse, hashlib, json, shutil, subprocess, sys, urllib.request, zipfile
from pathlib import Path
from types import SimpleNamespace
from common import sha, read, write, utc, verify

def run(a):
    p=read(a.plan); assert sha(a.plan)==a.plan_sha; verify(p,a.bundle,a.assets)
    publication=read(a.publication); cap=read(a.capture)
    assert publication['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED'
    assert cap['natural_exit']==0 and cap['ZIP_CRC_unique_all_members']
    ref=p['training_original_reference']; assert cap['archive_SHA']==ref['archive_SHA']
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid==p['GPU_UUID'][p['execution_node']] and p['execution_node']!=p['training_node']
    # Keep the complete immutable archive; extract only one full checkpoint.
    need=cap['archive_bytes']+p['final_checkpoint_bytes']+300_000_000+500_000_000
    assert shutil.disk_usage(a.out.parent).free>=need, (need,shutil.disk_usage(a.out.parent).free)
    a.out.mkdir(); archive=a.out/'complete_training_original.zip'
    rows=sorted([v for v in publication['assets'] if v['source_sha256']==cap['archive_SHA']],key=lambda v:v['source_offset'])
    assert rows and sum(v['bytes'] for v in rows)==cap['archive_bytes']
    with archive.open('xb') as target:
        for row in rows:
            assert target.tell()==row['source_offset']; h=hashlib.sha256(); count=0
            with urllib.request.urlopen(row['url'],timeout=180) as response:
                for block in iter(lambda:response.read(8*1024**2),b''):
                    target.write(block); h.update(block); count+=len(block)
            assert count==row['bytes'] and h.hexdigest()==row['sha256']
    assert archive.stat().st_size==cap['archive_bytes'] and sha(archive)==cap['archive_SHA']
    root=a.out/'original'; root.mkdir(); extracted=[]
    with zipfile.ZipFile(archive) as z:
        names=z.namelist(); assert len(names)==len(set(names))
        members=json.loads(z.read('member_SHA.json')); assert members==ref['member_SHA']
        assert set(names)==set(members)|{'member_SHA.json'}
        for name, digest in members.items():
            dest=root/name; assert dest.resolve().is_relative_to(root.resolve())
            size=z.getinfo(name).file_size
            extract=(size<=40_000_000 or name=='out/complete_final_and_selected.pt')
            if extract: dest.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256(); output=dest.open('xb') if extract else None
            try:
                with z.open(name) as source:
                    for block in iter(lambda:source.read(8*1024**2),b''):
                        h.update(block)
                        if output: output.write(block)
            finally:
                if output: output.close()
            assert h.hexdigest()==digest, name
            if extract: extracted.append(name)
    # Reading every complete member to EOF performs ZIP CRC validation too.
    write(root/'verified_extraction.json',dict(actual_UTC=utc(),archive_SHA=cap['archive_SHA'],all_member_SHA_CRC_unique_verified=True,extracted_members=extracted,large_rolling_checkpoint_verified_but_not_extracted=True))
    assert sha(root/'out/complete_final_and_selected.pt')==members['out/complete_final_and_selected.pt']
    from audit_official import run as audit
    audit(SimpleNamespace(plan=a.plan,plan_sha=a.plan_sha,bundle=a.bundle,assets=a.assets,input_root=root,out=a.out/'cpu_result'))
    # This newly extracted temporary PT is disposable after success. No old or
    # frozen original is removed; the complete public archive remains intact.
    temporary=root/'out/complete_final_and_selected.pt'
    assert temporary.resolve().is_relative_to(root.resolve())
    assert sha(temporary)==members['out/complete_final_and_selected.pt']
    temporary.unlink()
    write(a.out/'transport_receipt.json',dict(actual_UTC=utc(),archive=str(archive),archive_SHA=sha(archive),bytes=archive.stat().st_size,public_download=True,all_members_linear_SHA_CRC_unique=True,only_new_temporary_PT_removed_after_success=True,complete_original_retained=True))
    print((a.out/'cpu_result/audit_result.json').read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('publication','capture','plan','bundle','assets','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha',required=True); run(p.parse_args())

