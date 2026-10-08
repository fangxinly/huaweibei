"""Join genuinely saved D originals, two COMPLETE natural0 capsules and original other CPU.

No task, model, dataset, label access or remote connection. No large reference is a download.
"""
import argparse
from repair_gate import repair_gate
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile
from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write, safe_member, plan_gate, stage_association, capture_association, zip_audit, PRECHECK_JOINT, TRAIN_JOINT

def extract(base):
    base = Path(base)
    c = read(base / 'capture_receipt.json')
    zip_audit(base / 'snapshot.zip', c['snapshot_sha256'])
    with zipfile.ZipFile(base / 'snapshot.zip') as z:
        for n in z.namelist():
            safe_member(n)
            p = base / n
            require((p.resolve().is_relative_to(base.resolve())), 'Extraction target outside saved capsule')
            data = z.read(n)
            if p.exists():
                require(p.is_file() and not p.is_symlink() and sha(p) == hashlib.sha256(data).hexdigest(), 'Existing saved evidence differs; do not overwrite')
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--D-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=('precheck', 'train'), required=True)
    p.add_argument('--plan-sha', required=True)
    p.add_argument('--extract', action='store_true')
    p.add_argument('--repair-plan',type=Path,required=True)
    p.add_argument('--repair-plan-sha',required=True)
    a = p.parse_args()
    rp=repair_gate(a.repair_plan,a.repair_plan_sha)
    require(a.plan_sha==rp['original_training_plan_sha256'],'Exact unchanged training protocol')
    root = a.D_root.resolve()
    require(root.drive.upper() == 'D:' and a.out.resolve().is_relative_to(root) and not a.out.exists(), 'Fresh D joint target required')
    require(shutil.disk_usage(root).free >= 6 * 1024**3, 'Actual D preservation floor')
    if a.extract:
        for role in ('a', 'b'):
            extract(root / role)
    captures = {}
    manifests = {}
    for role in ('a', 'b'):
        c, m = capture_association(root / role)
        require(c['kind'] == ('gpu' if role == 'a' else 'cpu') and c['plan_sha256'] == a.plan_sha, 'Capture kind/plan association mismatch')
        captures[role] = {'receipt_sha256': sha(root / role / 'capture_receipt.json'),
                          'exit_sha256': sha(root / role / 'capture_actual_exit.json'), 'snapshot_sha256': c['snapshot_sha256'],
                          'members': c['members'], 'actual_capture_utc': c['actual_utc']}
        manifests[role] = m
    plan = plan_gate(root / 'a/source', a.plan_sha)
    require(sha(root / 'b/source/paired_fulltrain_execution_plan.json') == a.plan_sha, 'Different original CPU execution plan')
    for n, h in plan['source_sha256'].items():
        require(sha(root / 'b/source' / n) == h, 'Different original CPU frozen source')
    original, ex = stage_association(root / 'a/run', a.plan_sha)
    cpu, cpu_ex = stage_association(root / 'b/run', a.plan_sha, original['method'])
    require(cpu['stage'] == a.stage and cpu['status'] == 'ACTUAL_PAIRED_ORIGINAL_WHOLE_' + a.stage.upper() + '_CPU_AUDIT_PASSED_NO_MODEL_FORWARD', 'Original other CPU status mismatch')
    require(cpu['original_stage_receipt_sha256'] == sha(root / 'a/run/out/actual_stage_receipt.json') and
            cpu['original_stage_exit_sha256'] == sha(root / 'a/run/natural_exit.json') and
            cpu['original_root'] == original['root'] and cpu['original_CPU_GPU_UUID'] == plan['original_CPU_gpu_UUID'] and
            cpu['CPU_model_forward'] is False and cpu['final_TEST_access'] is False, 'Original other CPU exact evidence/node association')
    for role, source in (('a', 'paired_fulltrain_runtime_candidate_v1.py'), ('b', 'paired_fulltrain_CPU_audit_candidate_v1.py')):
        wrapper = read(root / role / 'run/wrapper_actual_start.json')
        expected=plan['source_sha256'][source] if role=='a' else rp['source_sha256']['paired_CPU_history_repair_entry_v1.py']
        require(wrapper['child_source_sha256']==expected,'Actual original GPU or supplemental CPU source mismatch')
        if role=='b':
            require(cpu['CPU_history_repair_plan_sha256']==a.repair_plan_sha and cpu['original_CPU_source_sha256']==plan['source_sha256'][source] and cpu['prior_failed_CPU_original_preserved'] is True,'Supplemental CPU exact parent/failure/source binding')
            for n,h in rp['source_sha256'].items():require(sha(root/'b/run/repair_source'/n)==h,'Captured actual supplemental CPU source')
            require(sha(root/'b/run/repair_source/repair_plan.json')==a.repair_plan_sha,'Captured supplemental protocol')
            checks=cpu['checks']['arrays_history_orders']['history_serialization_only']
            require(checks=={'rows':100,'sole_field':'dropped_TRAIN_rows','JSON_type':'list','Torch_type':'tuple','all_values_and_other_field_types_exact':True,'checkpoint_mutated':False},'Strict sole-field normalization proof')
    keys = ('clean_initial_full', 'precheck_after2_full') if a.stage == 'precheck' else ('complete_resume_full', 'selected_best_full')
    whole = {}
    for key in keys:
        item = original[key]
        file = root / 'a/run/out' / PurePosixPath(item['path']).name
        require(file.is_file() and file.stat().st_size == item['bytes'], 'Actual complete original D download missing')
        whole[key] = zip_audit(file, item['sha256'])
        ref = manifests['a']['large_references_not_downloads'].get('run/out/' + file.name)
        require(ref and ref['sha256'] == item['sha256'] and ref['bytes'] == item['bytes'], 'Original capture/whole D file mismatch')
        record = cpu['whole_checkpoint_records'][key]
        require(record['sha256'] == item['sha256'] and record['bytes'] == item['bytes'] and record['CRC'] is True, 'Original CPU loaded whole file differs from D')
        rel = PurePosixPath(record['original_saved_copy']).relative_to(PurePosixPath(cpu['root']))
        cpu_ref = manifests['b']['large_references_not_downloads'].get('run/' + str(rel))
        require(cpu_ref and cpu_ref['sha256'] == item['sha256'] and cpu_ref['bytes'] == item['bytes'], 'Original CPU capture/loaded whole file mismatch')
    result = {'status': PRECHECK_JOINT if a.stage == 'precheck' else TRAIN_JOINT,
              'actual_local_joint_utc': now(), 'argv': sys.argv, 'method': original['method'], 'original_root': original['root'],
              'plan_sha256': a.plan_sha, 'orders_sha256': plan['orders_sha256'],
              'official_row_ID_identity_sha256': plan['official_train_dev_ID_identity_sha256'],
              'original_stage_receipt_sha256': sha(root / 'a/run/out/actual_stage_receipt.json'),
              'original_stage_exit_sha256': sha(root / 'a/run/natural_exit.json'),
              'original_other_CPU_receipt_sha256': sha(root / 'b/run/out/actual_stage_receipt.json'),
              'original_other_CPU_exit_sha256': sha(root / 'b/run/natural_exit.json'), 'captures': captures,
              'whole_checkpoint_D_SHA_CRC_passed': True, 'original_complete_checkpoints_D': whole,
              'large_references_are_not_new_downloads': True, 'CPU_model_forward': False,
              'fresh_public_instance_selected_replay_pending': a.stage == 'train',
              'final_TEST_executed': False, 'overall_research_or_formal_five_metric_superiority_complete': False}
    result['CPU_history_repair_plan_sha256']=a.repair_plan_sha
    result['CPU_history_repair_source_sha256']=rp['source_sha256']['paired_CPU_history_repair_entry_v1.py']
    result['prior_CPU_natural1_failure_preserved']=True
    result['original_training_source_or_weights_changed']=False
    write(a.out, result)
    print(__import__('json').dumps(result), flush=True)

if __name__ == '__main__':
    main()
