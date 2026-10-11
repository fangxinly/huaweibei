"""Prepared launch wrapper: retain an actual child exit independently of its receipt."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--phase', choices=['precheck', 'execute'], required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    assert plan['version'] == 2 and plan['require_actual_phase_exit_receipts'] is True
    root = Path(plan['teacher_root']) / plan['collection_subdirectory']
    script = root / 'collect_group_teacher_fit_inner_v2.py'
    assert sha(script) == plan['collector_sha256']
    assert not (root / args.phase).exists()
    exit_path = root / (args.phase + '_exit.json')
    assert not exit_path.exists()
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    command = [sys.executable, str(script), '--plan', str(args.plan),
               '--fold', str(args.fold), '--phase', args.phase]
    started = now()
    with (root / (args.phase + '_process.log')).open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        code = child.wait()  # No timeout or termination of a healthy process.
    receipt = root / args.phase / 'receipt.json'
    proof = {'phase': args.phase, 'fold': args.fold, 'child_pid': child.pid,
             'child_argv': command[1:], 'exit_code': code, 'started_utc': started,
             'finished_utc': now(), 'plan_sha256': sha(args.plan),
             'collector_sha256': sha(script), 'wrapper_sha256': sha(__file__),
             'receipt_sha256': sha(receipt) if receipt.is_file() else None}
    temporary = exit_path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(proof, indent=2), encoding='utf-8')
    temporary.replace(exit_path)
    print('ACTUAL_COLLECTION_CHILD_EXIT', code, flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
