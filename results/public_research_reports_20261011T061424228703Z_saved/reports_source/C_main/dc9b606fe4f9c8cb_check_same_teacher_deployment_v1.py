"""Read-only asset/process verification before the frozen v2 GPU collector."""
from pathlib import Path
import argparse, datetime, hashlib, json, shutil, subprocess

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--fold', type=int, required=True)
    a = p.parse_args()
    plan = json.loads((a.root/'same_teacher_collection_plan_v2.json').read_text())
    assert sha(a.root/'same_teacher_collection_plan_v2.json') == '57b740fed4ee9ac2d9736956f2e6cbc3b9dcd30b102610abe49f563a493738f7'
    selected = plan['folds'][a.fold]
    parent = Path(plan['teacher_root'])
    checked = {}
    for folder, pins in [(a.root, plan['new_source_sha256']), (parent, plan['shared_teacher_pins']),
                         (parent, selected['original_small_pins'])]:
        for name, expected in pins.items():
            actual = sha(folder/name)
            assert actual == expected, name
            checked[str(folder/name)] = actual
    assert sha(a.root/'calibration_video_role_plan_v1.json') == plan['role_plan_sha256']
    assert sha(a.root/'run_same_teacher_collection_v2.py') == 'd0483173b6f3e9bba63389e5fd2d13e514a358ab43b103838672ab0d29842218'
    assert sha(a.root/'audit_same_teacher_collection_v2.py') == 'dcff3ad8965214682a9f13c6988f6123e04bc7cb6a79517c6092f680c1c188c3'
    for name, record in selected['role_files'].items():
        assert sha(a.root/name) == record['sha256']
    full = parent/'run_v1/selected_full_checkpoint.pt'
    assert full.stat().st_size == selected['full_checkpoint_bytes']
    assert sha(full) == selected['full_checkpoint_sha256']
    q = lambda args: subprocess.check_output(args, text=True).strip()
    gpu = q(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.total','--format=csv,noheader,nounits'])
    compute = q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'])
    assert gpu.split(',')[0].strip() == selected['expected_uuid'] and not compute
    processes = []
    for child in Path('/proc').iterdir():
        if not child.name.isdigit(): continue
        try: argv = (child/'cmdline').read_bytes().decode().split('\0')[:-1]
        except (OSError, UnicodeError): continue
        if argv: processes.append({'pid':int(child.name),'argv':argv})
    completion = json.loads((parent/'run_v1/completion.json').read_text())
    history = json.loads((parent/'run_v1/history.json').read_text())
    assert len(history) == completion['completed_epochs'] == 100
    assert completion['best_epoch'] == min(range(1,101),key=lambda e:history[e-1]['inner_sample_mse'])
    assert json.loads((parent/'formal_exit.json').read_text())['exit_code'] == 0
    result = {'status':'ACTUAL_SAME_TEACHER_DEPLOYMENT_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'fold':a.fold,'gpu':gpu,'compute':compute,'processes_full_argv':processes,
              'remote_free_bytes':shutil.disk_usage(parent).free,'checked_sha256':checked,
              'full_weight_sha256':selected['full_checkpoint_sha256'],'full_weight_new_downloaded':False,
              'best_epoch':completion['best_epoch'],'completed_epochs':100,'source_sha256':sha(__file__)}
    assert result['remote_free_bytes'] >= plan['minimum_remote_free_bytes']
    out=a.root/'deployment_verified.json'
    assert not out.exists()
    out.write_text(json.dumps(result,indent=2))
    print('ACTUAL_DEPLOYMENT_VERIFIED',a.fold,flush=True)

if __name__ == '__main__': main()
