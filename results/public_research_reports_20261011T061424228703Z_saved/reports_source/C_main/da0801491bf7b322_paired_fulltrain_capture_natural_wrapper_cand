"""COMPLETE receipt followed by actual natural capture0; no task interruption."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write, plan_gate

def main():
    p = argparse.ArgumentParser()
    for n in ('root', 'bundle', 'dest', 'python'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--plan-sha', required=True)
    p.add_argument('--kind', choices=('gpu', 'cpu', 'identity'), required=True)
    a = p.parse_args()
    if a.kind=='identity':
        f=a.bundle/'paired_TRAIN_DEV_input_identity_plan.json';require(sha(f)==a.plan_sha,'Identity capture exact plan')
        plan=read(f)
        require(plan.get('status')=='PAIRED_TRAIN_DEV_IDENTITY_INPUT_ONLY_PROTOCOL_FROZEN' and
                plan.get('labels_enabled') is False and plan.get('model_or_training_enabled') is False and
                plan.get('TEST_entry_enabled') is False,'Identity capture frozen role protocol')
        for n,h in plan['source_sha256'].items():require(sha(a.bundle/n)==h,'Identity capture exact source')
    else:plan = plan_gate(a.bundle, a.plan_sha)
    require(not a.dest.exists(), 'Fresh capture destination required')
    source = a.bundle / 'paired_fulltrain_capture_candidate_v1.py'
    require(sha(source) == plan['source_sha256'][source.name], 'Frozen capture source mismatch')
    command = [str(a.python), str(source), '--root', str(a.root), '--bundle', str(a.bundle), '--dest', str(a.dest),
               '--plan-sha', a.plan_sha, '--kind', a.kind]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = child.communicate()
    record = {'actual_utc': now(), 'wrapper_pid': os.getpid(), 'child_pid': child.pid, 'child_full_argv': command,
              'wrapper_full_argv': sys.argv, 'exit_code': child.returncode, 'natural_wait_verified': True,
              'source_sha256': sha(source), 'stdout_sha256': hashlib.sha256(stdout.encode()).hexdigest(),
              'stderr_sha256': hashlib.sha256(stderr.encode()).hexdigest()}
    if child.returncode == 0:
        require('CAPTURE_COMPLETE ' in stdout and (a.dest / 'capture_receipt.json').is_file(), 'Capture natural0 without COMPLETE original receipt')
        record['original_receipt_sha256'] = sha(a.dest / 'capture_receipt.json')
        write(a.dest / 'capture_actual_exit.json', record)
        (a.dest / 'capture.stdout.log').write_text(stdout, encoding='utf-8')
        (a.dest / 'capture.stderr.log').write_text(stderr, encoding='utf-8')
    else:
        failure = a.root / ('actual_failed_capture_' + a.dest.name + '.json')
        require(not failure.exists(), 'Do not overwrite original failed capture')
        write(failure, dict(record, stdout=stdout, stderr=stderr))
    print(stdout, end='')
    print(stderr, file=sys.stderr, end='')
    print('ACTUAL_CAPTURE_NATURAL_EXIT ' + __import__('json').dumps(record), flush=True)
    raise SystemExit(child.returncode)

if __name__ == '__main__':
    main()
