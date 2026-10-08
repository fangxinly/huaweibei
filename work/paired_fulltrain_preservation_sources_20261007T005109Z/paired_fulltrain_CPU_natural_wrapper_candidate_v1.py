"""Original CPU audit: fresh root, exact physical arguments, natural child exit only."""
import argparse
from pathlib import Path
import subprocess
import sys
from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write, plan_gate

def main():
    p = argparse.ArgumentParser()
    for n in ('root', 'bundle', 'python', 'child-arguments'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--plan-sha', required=True)
    a = p.parse_args()
    plan = plan_gate(a.bundle, a.plan_sha)
    require(a.root.parent == Path('/data/coding') and a.root.name.startswith('paired_fulltrain_cpu_') and
            not (a.root / 'out').exists() and not (a.root / 'natural_exit.json').exists(), 'Fresh CPU root required')
    arguments = read(a.child_arguments)
    require(isinstance(arguments, list) and all(isinstance(x, str) for x in arguments), 'Physical CPU child argument list required')
    for flag, value in (('--root', str(a.root)), ('--bundle', str(a.bundle)), ('--plan-sha', a.plan_sha)):
        require(arguments.count(flag) == 1 and arguments[arguments.index(flag) + 1] == value, 'CPU child identity mismatch')
    source = a.bundle / 'paired_fulltrain_CPU_audit_candidate_v1.py'
    require(sha(source) == plan['source_sha256'][source.name], 'Frozen CPU source mismatch')
    command = [str(a.python), str(source), *arguments]
    write(a.root / 'wrapper_actual_start.json', {'actual_utc': now(), 'wrapper_argv': sys.argv, 'child_full_argv': command,
        'child_source_sha256': sha(source), 'child_args_sha256': sha(a.child_arguments)})
    with (a.root / 'child.stdout.log').open('w') as stdout, (a.root / 'child.stderr.log').open('w') as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        write(a.root / 'actual_child_launch.json', {'actual_utc': now(), 'pid': child.pid, 'full_argv': command})
        code = child.wait()
    receipt = a.root / 'out/actual_stage_receipt.json'
    write(a.root / 'natural_exit.json', {'actual_utc': now(), 'child_pid': child.pid, 'exit_code': code, 'natural_exit': True,
        'receipt_sha256': sha(receipt) if receipt.is_file() else None, 'full_argv': command})
    require(code != 0 or receipt.is_file(), 'CPU natural0 without original receipt')
    raise SystemExit(code)

if __name__ == '__main__':
    main()
