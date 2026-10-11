"""Restore a newly preserved source ZIP on the new node; no scientific imports."""
import datetime as dt,hashlib,json,pathlib,shutil,urllib.request,zipfile
P=pathlib.Path
ROOT=P('/root/autodl-tmp/ni_node23668_20261011T114428Z/source_route_actual')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    ROOT.mkdir();archive=ROOT/'original_setup_from_GitHub.zip'
    url='https://github.com/fangxinly/huaweibei/releases/download/autonomous-flow-health-20261008/new-node23668-setup-a8b27ae51252.zip'
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Original-source-route-probe'}),timeout=60) as response,archive.open('xb') as out:shutil.copyfileobj(response,out,1024**2)
    assert archive.stat().st_size==6792023 and sha(archive)=='a8b27ae51252dc6c658c34ea5a8cbeb08cc58a3acf1cb176ecbd297ecfb0dd6e'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();manifest=json.loads(z.read('member_manifest.json'))
        assert len(names)==len(set(names)) and set(names)=={r['name'] for r in manifest}|{'member_manifest.json'}
        for row in manifest:
            b=z.read(row['name']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
        source=ROOT/'source';source.mkdir()
        for n in names:
            if n.startswith('careflow_source/'):
                rel=P(n).name;assert n=='careflow_source/'+rel
                (source/rel).write_bytes(z.read(n))
        (ROOT/'qualified_native_actual_stage_receipt.json').write_bytes(z.read('qualified_native_actual_stage_receipt.json'))
    receipt=dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),status='NEW_NODE_PUBLIC_SOURCE_FULL_RESTORE_SHA_CRC_MEMBER_VERIFIED',original_URL=url,whole_SHA=sha(archive),bytes=archive.stat().st_size,members=len(names),source_members=len(list(source.iterdir())),all_member_SHA_CRC_unique_exactset=True,task_arrays_numerically_decoded=False,scientific_imports_or_once_consumed=False,source_root=str(source),small_transport_route_only_not_full_model_preservation=True)
    (ROOT/'route_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if ROOT.exists():(ROOT/'failed_natural_exit.json').write_text(json.dumps({'actual_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'error_type':type(e).__name__,'message':str(e),'exit':1}))
        raise
