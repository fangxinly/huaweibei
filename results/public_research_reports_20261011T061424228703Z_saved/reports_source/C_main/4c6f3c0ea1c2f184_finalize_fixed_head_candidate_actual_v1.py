"""Join original candidate GPU/other-node NumPy receipts and complete D capsules."""
import datetime, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np

base=Path('D:/CodexBackups/selective_flow_20261003_1105')
root=base/'fixed_head_candidate_actual_20261006T191103Z'
workspace=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
audits={}
for node in ('a','b'):
    d=root/node; rec=read(d/'capture_receipt.json'); ex=read(d/'capture_actual_exit.json'); pkg=d/'snapshot.zip'
    assert ex['exit_code']==0 and ex['natural_wait_verified'] and rec['pid']==ex['child_pid']
    assert rec['argv']==ex['child_full_argv'][1:]
    if 'original_receipt_sha256' in ex: assert ex['original_receipt_sha256']==sha(d/'capture_receipt.json')
    assert sha(pkg)==rec['snapshot_sha256'] and pkg.stat().st_size==rec['snapshot_bytes']
    with zipfile.ZipFile(pkg) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==rec['members']
        mf=json.loads(z.read('member_manifest.json'))
        assert set(z.namelist())==set(mf['small_members'])|{'member_manifest.json'}
        for n,it in mf['small_members'].items():
            content=z.read(n); assert hashlib.sha256(content).hexdigest()==it['sha256'] and len(content)==it['bytes']
        for n in z.namelist():
            rel=Path(n); assert not rel.is_absolute() and '..' not in rel.parts
            target=d/rel
            if target.exists(): assert target.read_bytes()==z.read(n)
            else: target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(z.read(n))
    assert rec['large_references']==mf['large_references_not_downloads']
    audits[node]={'capture_receipt_sha256':sha(d/'capture_receipt.json'),'capture_exit_sha256':sha(d/'capture_actual_exit.json'),
        'snapshot_sha256':sha(pkg),'members':rec['members'],'actual_capture_utc':rec['actual_utc'],
        'actual_capture_exit_utc':ex['actual_exit_utc'],'full_SHA_CRC_unique_all_members_passed':True,
        'large_references_not_downloads':rec['large_references']}
g=root/'a/run'; b=root/'b/run'
gpu=read(g/'out/actual_candidate_collection_receipt.json'); ge=read(g/'natural_exit.json')
cpu=read(b/'cpu_original_receipt.json'); ce=read(b/'cpu_original_receipt.actual_exit.json')
plan=read(g/'source/candidate_collection_plan.json')
assert ge['exit_code']==ce['exit_code']==0 and ge['natural_wait_verified'] and ce['natural_wait_verified']
assert ge['child_pid']==gpu['pid'] and ce['child_pid']==cpu['pid']
assert ge['child_full_argv'][1:]==gpu['argv'] and ce['child_full_argv'][1:]==cpu['argv']
assert sha(g/'out/actual_candidate_collection_receipt.json')==ge['original_receipt_sha256']==cpu['original_collection_receipt_sha256']
assert sha(b/'cpu_original_receipt.json')==ce['original_receipt_sha256']
assert sha(g/'natural_exit.json')==cpu['original_GPU_natural_exit_sha256']
assert sha(g/'out/original_headFIT232_label_free.npz')==gpu['original_array_sha256']==cpu['original_array_sha256']==sha(b/'out/original_headFIT232_label_free.npz')
assert sha(g/'source/candidate_collection_plan.json')==gpu['plan_sha256']==sha(b/'source/candidate_collection_plan.json')
for n,h in plan['source_and_role_sha256'].items(): assert sha(g/'source'/n)==sha(b/'source'/n)==h
assert not cpu['GPU_used'] and not cpu['CPU_model_forward'] and not cpu['task_labels_read'] and not cpu['new_performance_or_head_fit']
assert not gpu['task_labels_read'] and not gpu['head_fit_or_optimizer_updates'] and not gpu['headEVAL201_inputs_or_labels_used'] and not gpu['actual_performance_or_gain_measured']
assert gpu['rows']==cpu['rows']==232 and gpu['videos']==cpu['videos']==9 and cpu['max_repeat_error']==0
prior=base/'staged_reference100_complete_actual_20261006T183917Z'
prior_joint=prior/'complete_reference100_GPU_D_B_CPU_joint_audit.json'
assert sha(prior_joint)==plan['reference_joint_sha256']
full=prior/'a/run/out/selected_best_full.pt'; fullsha=sha(full)
assert fullsha==plan['checkpoint_file_sha256']==cpu['original_B_reference_full_checkpoint_fresh_sha256']
for node in audits:
    ref=audits[node]['large_references_not_downloads']['reference/selected_best_full.pt']
    assert ref['sha256']==fullsha and ref['bytes']==full.stat().st_size and ref['stable_before_after']
with np.load(g/'out/original_headFIT232_label_free.npz',allow_pickle=False) as z:
    d=z['pC'].astype(np.float64)-z['pF'].astype(np.float64)
    assert np.array_equal(d,z['delta']) and np.array_equal(4*d*d,z['W_raw'])
    assert np.max(np.abs(z['pF']-z['pF_restored']))==np.max(np.abs(z['pC']-z['pC_repeat']))==0
    assert len(np.unique(z['video_ids']))==9
record={'status':'ACTUAL_HEADFIT232_SINGLE_CANDIDATE_GPU_D_B_ARRAY_SOURCE_REFERENCE_JOINT_PASSED',
    'actual_local_joint_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':gpu['plan_sha256'],
    'original_GPU_receipt_sha256':sha(g/'out/actual_candidate_collection_receipt.json'),
    'original_GPU_natural_exit_sha256':sha(g/'natural_exit.json'),'GPU_actual_exit_utc':ge['actual_exit_utc'],
    'original_B_CPU_receipt_sha256':sha(b/'cpu_original_receipt.json'),'original_B_CPU_natural_exit_sha256':sha(b/'cpu_original_receipt.actual_exit.json'),
    'B_CPU_actual_exit_utc':ce['actual_exit_utc'],'original_array_sha256':gpu['original_array_sha256'],
    'captures':audits,'original_complete_reference_joint_sha256':sha(prior_joint),
    'existing_complete_D_best_full_fresh_sha256':fullsha,'existing_complete_D_best_bytes':full.stat().st_size,
    'full_reference_new_download_this_candidate':False,'new_CPU_Torch_model_forward':False,
    'rows':232,'videos':9,'diagnostics':gpu['diagnostics'],'GPU_budget':gpu['final_budget'],
    'task_labels_head_fit_201_inputs_or_performance':False,
    'B_capture_path_stamp_note':'Path suffix191902Z was a naming transcription error; actual start19:18:49.559629 and completion19:18:54.020423 are original receipt times. No receipt time changed or backfilled.',
    'next':'Separate complete label access, matched U/W fitting and one201 development evaluation protocol before any fitting or201 inference.'}
write(root/'complete_label_free_candidate_GPU_D_B_CPU_joint_audit.json',record)
write(workspace/'outputs/完整流单候选232零标签实际结果.json',record)
print(json.dumps({'status':record['status'],'joint_sha256':sha(root/'complete_label_free_candidate_GPU_D_B_CPU_joint_audit.json'),'diagnostics':record['diagnostics'],'budget':record['GPU_budget']},ensure_ascii=False))
