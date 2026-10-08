"""Root-specific completed original run/source/argv capsule. Large hashes are references."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write, plan_gate, stage_association, stable_file, zip_audit

def main():
    p = argparse.ArgumentParser()
    for n in ('root', 'bundle', 'dest'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--plan-sha', required=True)
    p.add_argument('--kind', choices=('gpu', 'cpu'), required=True)
    a = p.parse_args()
    plan = plan_gate(a.bundle, a.plan_sha)
    require(a.root.parent == Path('/data/coding') and a.root.name.startswith('paired_fulltrain_') and
            a.dest.parent == Path('/data/coding') and a.dest.name.startswith('capture_paired_fulltrain_') and
            not a.dest.exists(), 'Fresh complete paired capture root/destination required')
    r, ex = stage_association(a.root, a.plan_sha)
    expected = plan['original_CPU_gpu_UUID'] if a.kind == 'cpu' else plan['assigned_gpu_UUID'][r['method']]
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    require(gpu.strip() == expected, 'Root-specific actual GPU identity mismatch')
    raw = {'actual_utc': now(), 'GPU_UUID_raw': gpu}
    for name, command in (
        ('compute', ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader']),
        ('full_process_argv', ['ps', '-eo', 'pid,ppid,args']),
        ('filesystem_space', ['df', '-B1', str(a.root)])):
        raw[name] = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    a.dest.mkdir()
    write(a.dest / 'actual_capture_direct_evidence.json', raw)
    paths = [('run/' + f.relative_to(a.root).as_posix(), f) for f in sorted(a.root.rglob('*'))
             if f.is_file() and '__pycache__' not in f.parts]
    # Frozen source dependency list, not a long unbounded unrelated bundle traversal.
    for n in [*plan['source_sha256'], plan['orders_file'], 'paired_fulltrain_execution_plan.json']:
        paths.append(('source/' + n, a.bundle / n))
    paths.append(('actual_capture_direct_evidence.json', a.dest / 'actual_capture_direct_evidence.json'))
    small, large = {}, {}
    for n, path in paths:
        require(n not in small and n not in large, 'Duplicate capture name')
        record = stable_file(path)
        (small if record['bytes'] <= 16 * 1024**2 else large)[n] = record
    manifest = {'actual_utc': now(), 'pid': os.getpid(), 'argv': sys.argv, 'kind': a.kind,
                'root': str(a.root), 'source_bundle': str(a.bundle), 'plan_sha256': a.plan_sha,
                'original_receipt_sha256': sha(a.root / 'out/actual_stage_receipt.json'),
                'original_exit_sha256': sha(a.root / 'natural_exit.json'), 'small_members': small,
                'large_references_not_downloads': large,
                'actual_complete_original_models_must_be_downloaded_separately_and_joined': True}
    write(a.dest / 'member_manifest.json', manifest)
    archive = a.dest / 'snapshot.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for n, record in small.items():
            z.write(record['original_path'], n)
        z.write(a.dest / 'member_manifest.json', 'member_manifest.json')
    zip_audit(archive)
    with zipfile.ZipFile(archive) as z:
        import hashlib
        for n, record in small.items():
            require(hashlib.sha256(z.read(n)).hexdigest() == record['sha256'], 'Capture original member changed')
    result = {'status': 'ACTUAL_PAIRED_CAPTURE_COMPLETE', 'actual_utc': now(), 'pid': os.getpid(), 'argv': sys.argv,
              'kind': a.kind, 'method': r['method'], 'root': str(a.root), 'source_bundle': str(a.bundle),
              'plan_sha256': a.plan_sha, 'members': len(small) + 1, 'snapshot_sha256': sha(archive),
              'snapshot_bytes': archive.stat().st_size, 'large_references_not_downloads': large,
              'original_receipt_sha256': manifest['original_receipt_sha256'], 'original_exit_sha256': manifest['original_exit_sha256'],
              'full_checkpoint_new_download_this_capture': False}
    write(a.dest / 'capture_receipt.json', result)
    print('CAPTURE_COMPLETE ' + __import__('json').dumps(result), flush=True)

if __name__ == '__main__':
    main()
