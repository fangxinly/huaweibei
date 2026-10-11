"""Local adversarial range qualification and actual small Release restoration."""
import copy
import datetime as dt
import io
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from group5_release_transport_v1 import digest,plan,restore,seal,upload,write
from group5_test_selected_contract_v1 import validate_release_parts


def main():
    root=Path('D:/CodexBackups/selective_flow_20261003_1105')/('g5transport_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir()
    fixture=root/'fixture';fixture.mkdir()
    # Contains no task arrays, labels, model tensors or credentials.
    (fixture/'synthetic.bin').write_bytes(bytes(range(256))*8192+b'exact-tail-137')
    write(fixture/'scope.json',dict(kind='SYNTHETIC_BYTES_ONLY',scientific_execution=False))
    proof=seal(fixture,fixture/'original.zip')
    m=plan(Path(proof['archive']),'group5-transport-fixture-'+proof['archive_SHA'][:12],512*1024)
    write(root/'manifest.json',m)
    raw=Path(proof['archive']).read_bytes()
    r=dict(whole_SHA=m['whole_SHA'],parts=[dict(name=p['name'],bytes=p['bytes'],digest='sha256:'+p['SHA'],
        state='uploaded',url=p['name'],offset=p['offset']) for p in m['parts']])
    ranges={p['name']:raw[p['offset']:p['offset']+p['bytes']] for p in m['parts']}
    local=restore(m,r,root/'local_restored.zip',opener=lambda url:io.BytesIO(ranges[url]))
    checks=[]
    mutations=[('missing',lambda x:x['parts'].pop()),('duplicate',lambda x:x['parts'].append(x['parts'][0])),
       ('wrong_SHA',lambda x:x['parts'][0].update(digest='sha256:'+'0'*64)),
       ('wrong_size',lambda x:x['parts'][0].update(bytes=1)),('starter',lambda x:x['parts'][0].update(state='starter'))]
    for name,mutation in mutations:
        bad=copy.deepcopy(r);mutation(bad)
        try:validate_release_parts(m,bad['parts'])
        except (ValueError,PermissionError):checks.append(name)
        else:raise AssertionError(name)
    for name,change in [('gap',1),('overlap',-1)]:
        bad=copy.deepcopy(m);bad['parts'][1]['offset']+=change
        try:validate_release_parts(bad,r['parts'])
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    corrupted=dict(ranges);corrupted[m['parts'][0]['name']]=b'X'+corrupted[m['parts'][0]['name']][1:]
    try:restore(m,r,root/'rejected_corruption.zip',opener=lambda url:io.BytesIO(corrupted[url]))
    except ValueError:checks.append('download_corruption')
    else:raise AssertionError('corruption accepted')
    write(root/'local_qualification.json',dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        local=local,negative_checks=checks,scientific_execution=False,real_network_qualified=False))
    os.environ['PATH']='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+os.environ['PATH']
    remote=upload(m,root/'remote_upload_receipt.json')
    restored=restore(m,remote,root/'actual_remote_restored.zip')
    write(root/'actual_remote_restoration.json',restored)
    write(root/'qualification.json',dict(status='SYNTHETIC_RELEASE_RANGES_FULL_DOWNLOAD_RESTORED',
        actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),original=proof,remote=remote,restored=restored,
        negative_checks=checks,part_bytes_for_real_checkpoints=512*1024**2,
        actual_large_checkpoint_or_CPU_qualification=False,scientific_execution=False))
    print(json.dumps(dict(root=str(root),status='SYNTHETIC_RELEASE_RANGES_FULL_DOWNLOAD_RESTORED',whole_SHA=digest(root/'actual_remote_restored.zip'))))


if __name__=='__main__':main()
