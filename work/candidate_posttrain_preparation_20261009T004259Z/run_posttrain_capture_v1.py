"""Native fresh preflight, natural stage exit, small original evidence capture.

Large training state stays at its immutable root and is referenced by SHA;
it is never copied into an inference or scoring capture.
"""
import argparse, datetime, hashlib, importlib.metadata, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from common import sha, read, write, utc, verify

def run(a):
    p = read(a.plan)
    assert sha(a.plan) == a.plan_sha
    assert p['status'] == 'ACTUAL_OFFICIAL_CANDIDATE_' + a.stage.upper() + '_STAGE_FROZEN'
    assert a.node==p['execution_node']; assert (a.stage=='audit' and a.node!=p['training_node']) or (a.stage in ('infer','score') and a.node==p['training_node'])
    assert not a.root.exists()
    assert sys.executable == str(a.assets / '.venv/bin/python')
    bundle = a.plan.parent
    verify(p, bundle, a.assets)
    for n, h in p['posttrain_native_CPU_qualification'].items():
        assert sha(bundle / n) == h
    native = read(bundle / 'candidate_posttrain_native_result.json')
    assert native['source_SHA'] == p['source_sha256']['qualify_candidate_posttrain_v1.py']
    assert read(bundle / 'candidate_posttrain_native_exit.json')['natural_exit'] == 0
    versions = {n: importlib.metadata.version(n) for n in p['runtime_versions']}
    assert versions == p['runtime_versions']
    uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
    assert uuid == p['GPU_UUID'][a.node] and not compute.strip()
    space = shutil.disk_usage(a.root.parent)
    assert space.free >= p['remote_free_floor_bytes']
    remaining = (datetime.datetime.fromisoformat(p['conservative_lease_end_UTC']) - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    assert remaining > p['posttrain_stage_budget_seconds'][a.stage] + 7200
    cmd = [sys.executable, str(bundle / {'audit': 'download_and_audit_final_v1.py', 'infer': 'infer_official.py', 'score': 'score_official.py'}[a.stage]), '--plan', str(a.plan), '--plan-sha', a.plan_sha, '--bundle', str(bundle), '--assets', str(a.assets), '--out', str(a.root / 'out')]
    if a.stage == 'audit':
        assert a.publication and a.capture
        cmd += ['--publication', str(a.publication), '--capture', str(a.capture)]
        assert a.received_archive is None, 'Only complete public source transport is qualified for this new audit'
    elif a.stage == 'infer':
        assert a.input_root
        cmd += ['--input-root', str(a.input_root)]
    else:
        assert a.prediction
        cmd += ['--prediction', str(a.prediction)]
    a.root.mkdir()
    write(a.root / 'fresh_preflight.json', dict(actual_UTC=utc(), UUID=uuid, compute=compute, processes=subprocess.check_output(['ps', '-eo', 'pid,ppid,args', '--width', '10000'], text=True), fullargv=[sys.executable] + sys.argv, source_SHA=sha(__file__), runtime_versions=versions, space=space._asdict(), remaining_seconds=remaining, conservative_not_platform_confirmed=True))
    with (a.root / 'stdout.log').open('wb') as out, (a.root / 'stderr.log').open('wb') as err:
        child = subprocess.Popen(cmd, stdout=out, stderr=err, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        write(a.root / 'dispatch.json', dict(actual_UTC=utc(), pid=child.pid, fullargv=cmd))
        code = child.wait()
    write(a.root / 'natural_exit.json', dict(actual_UTC=utc(), pid=child.pid, fullargv=cmd, natural_exit=code))
    source = a.root / 'original_source'
    source.mkdir()
    for n in list(p['source_sha256']) + list(p['posttrain_native_CPU_qualification']) + [a.plan.name]:
        shutil.copy2(bundle / n, source / n)
    if a.stage == 'audit':
        # Full original archive + extracted states remain on B. The proof is small.
        candidates = [f for f in a.root.rglob('*') if f.is_file() and not f.is_relative_to(a.root / 'out')]
        candidates += list((a.root / 'out/cpu_result').glob('*.json'))
        candidates += list((a.root / 'out').glob('transport_receipt.json'))
        refs = {'training': p['training_original_reference'], 'peer_original_archive': str(a.root / 'out/complete_training_original.zip'), 'B_extracted_original': str(a.root / 'out/original')}
        write(a.root / 'large_original_references.json', refs)
        candidates.append(a.root / 'large_original_references.json')
    else:
        candidates = [f for f in a.root.rglob('*') if f.is_file()]
    members = {f.relative_to(a.root).as_posix(): sha(f) for f in candidates}
    write(a.root / 'member_SHA.json', members)
    archive = a.root / 'complete_small_original.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for n in list(members) + ['member_SHA.json']:
            z.write(a.root / n, n)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
        for n, h in members.items():
            d = hashlib.sha256()
            with z.open(n) as f:
                for block in iter(lambda: f.read(8 * 1024**2), b''):
                    d.update(block)
            assert d.hexdigest() == h
    assert archive.stat().st_size <= 40_000_000
    receipt = dict(status='ACTUAL_POSTTRAIN_' + a.stage.upper() + ('_COMPLETE' if code == 0 else '_FAILED'), actual_UTC=utc(), child_PID=child.pid, natural_exit=code, archive=str(archive), archive_SHA=sha(archive), archive_bytes=archive.stat().st_size, ZIP_CRC_unique_all_members=True, small_capture_only_large_original_referenced=a.stage == 'audit')
    write(a.root / 'capture_receipt.json', receipt)
    print(receipt, flush=True)
    raise SystemExit(code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['audit', 'infer', 'score'], required=True)
    parser.add_argument('--node', choices=['A', 'B', 'C'], required=True)
    for n in ('plan', 'assets', 'root'):
        parser.add_argument('--' + n, type=Path, required=True)
    for n in ('input-root', 'prediction', 'publication', 'capture', 'received-archive'):
        parser.add_argument('--' + n, type=Path)
    parser.add_argument('--plan-sha', required=True)
    run(parser.parse_args())
