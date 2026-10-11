from pathlib import Path
import datetime, hashlib, json
r = Path(__file__).resolve().parents[1]
d = Path('D:/CodexBackups/selective_flow_20261003_1105/jacobian_completed_20261005')
targets = {'a': ('b','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'),
           'b': ('c','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'),
           'c': ('b','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067')}
receipts = {}
verifier_sha = hashlib.sha256((r/'work/verify_soft_preservation_cpu_v1.py').read_bytes()).hexdigest()
for n,(target,uuid) in targets.items():
    m = json.loads((d/n/'preservation_manifest.json').read_text(encoding='utf-8'))
    x = json.loads((d/n/'cpu_preservation_receipt.json').read_text(encoding='utf-8'))
    assert x['status'] == 'ROTATED_TRAINING_NODE_SEVEN_FILES_AND_EXTRAS_FULL_CPU_SHA_ZIP_TENSORS_VERIFIED'
    assert x['training_node'] == n and x['target_cpu_node'] == target and x['target_gpu_uuid'] == uuid
    assert x['assembly_node'] == 'newA' and target != 'a' and n != target
    assert not x['cuda_initialized'] and x['full_state_tensor_count'] == 457
    assert x['source_sha256'] == verifier_sha
    assert x['directory'] == '/data/coding/jacobian_preservation_20261005T1406Z/'+n
    assert len(x['required_seven_files']) == 7 and len(x['extra_files']) == 5
    for group in ['required_seven_files','extra_files']:
        assert x[group] == m[group]
        for filename,meta in m[group].items():
            f=d/n/filename
            assert f.stat().st_size==meta['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256']
    history=json.loads((d/n/'history.json').read_text(encoding='utf-8'))
    sel=json.loads((d/n/'selection.json').read_text(encoding='utf-8'))
    assert len(history)==100 and sel['best_epoch']==min(range(100),key=lambda i:history[i]['dev_author_batch_mean_mse'])+1
    assert sel['best_epoch']=={'a':41,'b':26,'c':41}[n]
    receipts[n]=x
out=dict(status='THREE_NEW_JACOBIAN_CPU_RECEIPTS_DOWNLOADED_AND_INDEPENDENTLY_AUDITED',
         utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), receipts=receipts,
         scope='A-to-B/B-to-C/C-to-B: every target differs from original training node and newA assembly. C target additionally strengthened outside assembly; this is not a distinct three-target cyclic rotation. All new receipts, not reused from seed91815.')
path=r/'outputs/Jacobian三臂七文件异训练与组装节点CPU保存核验.json'
with path.open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(out['status'])
