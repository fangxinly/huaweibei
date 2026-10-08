"""Original CPU audit: fresh root, exact physical arguments, natural child exit only."""
import argparse
from pathlib import Path
import subprocess
import sys
import json,hashlib,datetime,shutil
from repair_gate import repair_gate


def main():
    p = argparse.ArgumentParser()
    for n in ('root', 'bundle', 'python', 'child-arguments'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--plan-sha', required=True)
    p.add_argument('--repair-plan',type=Path,required=True)
    p.add_argument('--repair-plan-sha',required=True)
    a = p.parse_args()
    rp=repair_gate(a.repair_plan,a.repair_plan_sha)
    sys.path.insert(0,str(a.bundle))
    from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write, plan_gate
    plan = plan_gate(a.bundle,a.plan_sha)
    require(a.plan_sha==rp['original_training_plan_sha256'],'Original training protocol remains exact')
    import subprocess,datetime
    raw={'actual_utc':now(),'UUID':subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True),'compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True),'fullargv':subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True),'space':subprocess.check_output(['df','-B1',str(a.root)],text=True)}
    require(raw['UUID'].strip()==plan['original_CPU_gpu_UUID'] and not raw['compute'].strip(),'Fresh actual B identity/empty compute')
    require(shutil.disk_usage(a.root).free>=plan['remote_free_floor_bytes'],'Fresh actual space floor')
    require(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=8100)<datetime.datetime.fromisoformat(rp['conservative_lease_end_UTC']),'900s execution plus2h preservation margin')
    for n,h in plan['asset_sha256'].items():require(sha(Path(rp['public_assets'])/n)==h,'Fresh original public asset')
    write(a.root/'actual_repair_dispatch_fresh_direct.json',raw)
    require(a.root.parent == Path('/data/coding') and a.root.name.startswith('paired_fulltrain_cpu_') and
            not (a.root / 'out').exists() and not (a.root / 'natural_exit.json').exists(), 'Fresh CPU root required')
    arguments = read(a.child_arguments)
    require(isinstance(arguments, list) and all(isinstance(x, str) for x in arguments), 'Physical CPU child argument list required')
    for flag, value in (('--root', str(a.root)), ('--bundle', str(a.bundle)), ('--plan-sha', a.plan_sha)):
        require(arguments.count(flag) == 1 and arguments[arguments.index(flag) + 1] == value, 'CPU child identity mismatch')
    source=a.repair_plan.parent/'paired_CPU_history_repair_entry_v1.py'
    require(sha(source)==rp['source_sha256'][source.name],'Supplemental frozen CPU source mismatch')
    for n in (*rp['source_sha256'],'repair_plan.json'):
        target=a.root/'repair_source'/n;target.parent.mkdir(exist_ok=True);require(not target.exists(),'Fresh repair source copy');shutil.copyfile(a.repair_plan.parent/n,target)
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
