import hashlib,json,zipfile,sys,shutil
from pathlib import Path

base=Path('D:/CodexBackups/selective_flow_20261003_1105/anchored_flow_pilot40_complete_actual_20261007T103742Z')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
source=base/'A_original_capsule/source'
sys.path.insert(0,str(source))
from fold_evidence import verify_capsule,join
cap=json.loads((base/'A_capture_receipt.json').read_text())
a,r=verify_capsule(base/'A_complete_capture.zip',cap['sha256'])
pt=base/'resume_and_selected_full.pt'
assert pt.stat().st_size==r['complete_resume_and_selected']['bytes']
assert sha(pt)==r['complete_resume_and_selected']['sha256']
with zipfile.ZipFile(pt) as z:
    assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
    count=len(z.namelist())
if not (base/'B_complete_capture.zip').exists():
    result={'status':'PILOT40_COMPLETE_FULL_ORIGINAL_D_CPU_PENDING','original_receipt':r,
            'checkpoint_sha256':sha(pt),'bytes':pt.stat().st_size,'checkpoint_crc_unique_members':count,
            'original_capsule_sha256':cap['sha256'],'capture_sha_all_members_verified':True}
    target=base/'actual_D_original_verified.json'
    assert not target.exists()
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','checkpoint_sha256','bytes']}))
else:
    bcap=json.loads((base/'B_capture_receipt.json').read_text())
    b,cpu=verify_capsule(base/'B_complete_capture.zip',bcap['sha256'])
    with zipfile.ZipFile(base/'B_complete_capture.zip') as z:z.extractall(base/'B_original_capsule')
    paths={'original_capsule':'A_complete_capture.zip','CPU_capsule':'B_complete_capture.zip',
        'complete_checkpoint':'resume_and_selected_full.pt','prediction':'A_original_capsule/run/out/OUTER_prediction_only.npz',
        'original_receipt':'A_original_capsule/run/out/actual_stage_receipt.json',
        'CPU_receipt':'B_original_capsule/run/out/actual_stage_receipt.json'}
    binding={k:{'path':v,'sha256':sha(base/v)} for k,v in paths.items()}
    target=base/'actual_D_CPU_joint.json'
    join(binding,base,target)
    print(json.dumps({'status':json.loads(target.read_text())['status'],'joint_sha256':sha(target),
        'CPU':cpu},ensure_ascii=False))
