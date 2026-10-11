"""Verify complete original capture on D; never manufacture remote evidence."""
import argparse,datetime,hashlib,json,zipfile
from pathlib import Path

def sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
def run(folder,extract):
    receipt=json.loads((folder/'actual_capture_receipt.json').read_text());p=folder/'complete_actual_capture.zip'
    assert p.stat().st_size==receipt['archive_bytes'] and sha(p)==receipt['archive_SHA']
    with zipfile.ZipFile(p) as z:
        names=z.namelist();assert z.testzip() is None and len(names)==len(set(names))
        manifest=json.loads(z.read('member_SHA.json'))
        assert set(names)==set(manifest)|{'member_SHA.json'}
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
        natural=json.loads(z.read('natural_exit.json'));dispatch=json.loads(z.read('actual_dispatch.json'))
        assert natural['natural_exit']==receipt['child_natural_exit'] and natural['pid']==receipt['child_PID']
        assert natural['fullargv']==dispatch['fullargv']
        plans=[n for n in names if n.startswith('original_source/') and n.endswith('protocol.json')]
        assert len(plans)==1
        plan=json.loads(z.read(plans[0]));assert hashlib.sha256(z.read(plans[0])).hexdigest()==natural['plan_SHA']
        for name,h in plan['source_sha256'].items():assert hashlib.sha256(z.read('original_source/'+name)).hexdigest()==h
        physical=json.loads(z.read('actual_preflight.json'));assert physical['UUID']==plan['GPU_UUID'][receipt['node']] and not physical['compute'].strip()
        assert len(physical['fullargv'])>1 and dispatch['fullargv'][0]==physical['fullargv'][0]
        if extract:
            out=folder/'extracted';assert not out.exists();out.mkdir()
            for name in names:
                target=(out/name).resolve();assert target.is_relative_to(out.resolve())
            z.extractall(out)
    verdict=dict(actual_local_verification_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_SHA=receipt['archive_SHA'],receipt_SHA=sha(folder/'actual_capture_receipt.json'),member_count=len(names),natural_exit=natural['natural_exit'],PID=natural['pid'],source_and_argv_passed=True,physical_original_not_local_capture=True,all_original_member_SHA=True,ZIPCRC=True,unique=True)
    (folder/'actual_D_verification.json').write_text(json.dumps(verdict,indent=2),encoding='utf8');print(verdict)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--extract',action='store_true');a=p.parse_args();run(a.folder,a.extract)
