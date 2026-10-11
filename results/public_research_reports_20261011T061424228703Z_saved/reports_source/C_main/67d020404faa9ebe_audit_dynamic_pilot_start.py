"""Qualify real dynamic ZIPs and new prefix bytes without deserializing labels."""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

root=Path(sys.argv[1]);clock=sys.argv[2]
workspace=Path.cwd();pilot=Path('D:/CodexBackups/selective_flow_20261003_1105/anchored_flow_pilot40_20261007T091746Z')
known=json.loads((workspace/'outputs/正式双方fullTRAIN100整保存与CPU待接续.json').read_text(encoding='utf-8-sig'))
prior=json.loads(Path('D:/CodexBackups/selective_flow_20261003_1105/group5_single_precheck_20261007T084136Z/actual_precheck_D_B_CPU_joint.json').read_text())
known_sha={x['sha256'] for method in known['methods'].values() for x in method['whole_D_files'].values()}
known_sha.add(prior['whole_checkpoint_sha256']);results={};latest=None
for node in ['A','B','C']:
    receipt=json.loads((root/(node+'_dynamic_receipt.json')).read_text())
    archive=root/(node+'_dynamic_capture.zip')
    if sha(archive)!=receipt['sha256'] or archive.stat().st_size!=receipt['bytes']:raise ValueError('Dynamic full ZIP bytes differ')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if z.testzip() or len(names)!=len(set(names)):raise ValueError('Dynamic ZIP CRC/unique failed')
        manifest=json.loads(z.read('dynamic_manifest.json'))
        if set(names)-{'dynamic_manifest.json'}!=set(manifest['member_sha256']):raise ValueError('Complete member inventory differs')
        for name,h in manifest['member_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('Dynamic original member bytes differ '+name)
        phys=json.loads(z.read('physical.json'))
        pinned=json.loads(z.read('pinned_asset_runtime_plan.json'))
        if phys['uuid']!=pinned['assigned_gpu_uuid'][node] or phys['native_runtime']!=pinned['runtime_exact_versions']:
            raise ValueError('Physical/runtime binding differs')
        if node=='A':
            child=json.loads(z.read('originals/anchored_flow_pilot40_execution_20261007T091746Z/actual_child.json'))
            construction=json.loads(z.read('originals/anchored_flow_pilot40_run_20261007T091746Z/out/actual_construction.json'))
            if child['pid']!=construction['pid'] or child['fullargv']!=construction['argv']:raise ValueError('Original child/constructor differs')
            if construction['protocol_sha256']!='82d2de9439766b2f981f92703d1c34ac6b30c2ef52911e603bf71c22c2a7330a':
                raise ValueError('Pilot protocol differs')
            plan=json.loads(z.read('sources/anchored_flow_pilot40_source_20261007T091746Z/pilot40_execution_protocol.json')) if 'sources/anchored_flow_pilot40_source_20261007T091746Z/pilot40_execution_protocol.json' in names else json.loads((pilot/'bundle/pilot40_execution_protocol.json').read_text())
            for name,h in plan['source_sha256'].items():
                if manifest['member_sha256'].get('sources/anchored_flow_pilot40_source_20261007T091746Z/'+name)!=h:
                    raise ValueError('Frozen worker source binding differs '+name)
            history=json.loads(z.read('originals/anchored_flow_pilot40_run_20261007T091746Z/out/history.json'))
            steps=z.read('originals/anchored_flow_pilot40_run_20261007T091746Z/out/actual_FIT_steps.jsonl').decode().splitlines()
            latest=dict(child=child,construction=construction,last_completed_epoch=history[-1]['epoch'],
                complete_step_prefix=len(steps),last_history=history[-1],snapshot_clock=manifest['actual_capture_finish_utc'],
                natural_completion_claimed=False)
        for ref in manifest['large_original_refs']:
            if not ref['new_large_original_requires_D_transfer']:
                if ref['sha256'] not in known_sha:raise ValueError('Unqualified old whole-state SHA reference '+ref['original_path'])
            else:
                p=root/'A_frozen_prefix_full.pt'
                if p.stat().st_size!=ref['bytes'] or sha(p)!=ref['sha256']:raise ValueError('New complete prefix state bytes differ')
                with zipfile.ZipFile(p) as state:
                    if state.testzip() or len(state.namelist())!=len(set(state.namelist())):raise ValueError('New full prefix CRC/unique differs')
                ref['D_path']=str(p);ref['whole_D_prefix_bytes_SHA_CRC_unique']=True
        results[node]=dict(original_capture_start_utc=manifest['actual_capture_start_utc'],original_capture_finish_utc=manifest['actual_capture_finish_utc'],
            full_ZIP_sha256=receipt['sha256'],members=len(names),uuid=phys['uuid'],compute=phys['compute'],source_files=len(manifest['source_file_sha256']),
            large_original_refs=manifest['large_original_refs'],real_snapshot_not_static_backfill=True)
result=dict(status='THREE_REAL_DYNAMIC_SNAPSHOTS_D_COMPLETE_PREFIX_NOT_TRAINING_COMPLETE',actualclock_D_audit_utc=clock,
    scheduled_window_UTC='2026-10-07T09:30:00+00:00',node_snapshots=results,pilot_original_prefix=latest,
    old_whole_states_referenced_not_retransferred=True,new_pilot_complete_prefix_D_saved=True,
    actual_D_free_bytes=shutil.disk_usage('D:/').free,final_pilot_max_state_reservation_bytes=4000000000,
    original_extra_saving_margin_partly_used_for_requested_dynamic_prefix=True,
    final_pilot_cpu_or_training_complete_claimed=False,remaining_dynamic_windows_UTC=['11:30','13:00'])
if result['actual_D_free_bytes']<4000000000:raise ValueError('Final single pilot complete saving reservation exhausted')
write(root/'actual_D_dynamic_joint.json',result)
print(json.dumps(dict(status=result['status'],joint_sha256=sha(root/'actual_D_dynamic_joint.json'),
    epoch=latest['last_completed_epoch'],updates=latest['complete_step_prefix'],D_free_bytes=result['actual_D_free_bytes']),ensure_ascii=False))
