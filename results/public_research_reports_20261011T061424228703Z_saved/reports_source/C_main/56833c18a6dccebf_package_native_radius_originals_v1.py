from pathlib import Path
import argparse,datetime,hashlib,json,shutil,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;sha=lambda b:hashlib.sha256(b).hexdigest()
assert shutil.disk_usage(r).free>1073741824
assert json.loads((r/'mechanism/independent_audit.json').read_text())['status']=='CAL_NATIVE_RADIUS_ORIGINAL_GPU_MECHANISM_NUMPY_VERIFIED'
out=r/'preservation';out.mkdir(exist_ok=False)
names=['plan.json','authorized_cal_rows.npy','diagnose_native_radius_cal_v1.py','audit_native_radius_cal_v1.py','run_native_radius_cal_v1.py','run_same_teacher_capture_v1.py','mechanism_launch.json','mechanism_exit.json','mechanism.log']
names+=['mechanism/'+name for name in ['receipt.json','inventory.json','cal_candidate_arrays.npz','independent_audit.json']]
manifest={}
with zipfile.ZipFile(out/'snapshot.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in names:
        b=(r/name).read_bytes();manifest[name]={'bytes':len(b),'sha256':sha(b)};z.writestr(name,b)
    z.writestr('member_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out/'snapshot.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
b=(out/'snapshot.zip').read_bytes();receipt={'status':'CAL_NATIVE_RADIUS_ORIGINALS_PERMANENT_D_PACKAGED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bytes':len(b),'sha256':sha(b),'files':len(names),'true_labels_included':False,'no_EVAL_predictions':True}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
