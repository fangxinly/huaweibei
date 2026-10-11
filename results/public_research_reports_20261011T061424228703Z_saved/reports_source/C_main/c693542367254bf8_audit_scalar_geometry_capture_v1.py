from pathlib import Path
import argparse,datetime,hashlib,json,zipfile
p=argparse.ArgumentParser();p.add_argument("--snapshot",type=Path,required=True);p.add_argument("--prior",type=Path,required=True);p.add_argument("--originals",type=Path,required=True);p.add_argument("--remote-root",required=True);p.add_argument("--phase",choices=["precheck","execute"],required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((a.snapshot/"receipt.json").read_text())
ex=json.loads((a.snapshot/"actual_capture_exit.json").read_text())
assert ex["exit_code"]==0 and (a.snapshot/"exit_code.txt").read_text().strip()=="0"
data=(a.snapshot/"snapshot.zip").read_bytes();assert len(data)==r["bytes"] and sha(data)==r["sha256"]
skip={"inventory.json","finite_inventory.json","finite_c2_inventory.json","train_oracle_inventory.json","group_teacher_inventory.json","large_file_manifest.json"}
with zipfile.ZipFile(a.snapshot/"snapshot.zip") as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    m=json.loads(z.read("member_manifest.json"));assert set(z.namelist())==set(m)|{"member_manifest.json"}
    for name,item in m.items():
        b=z.read(name);assert len(b)==item["bytes"] and sha(b)==item["sha256"]
    assert sha(z.read("source/capture_soft_vector_v19.py"))==ex["capture_source_sha256"]=="6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad"
    inv=json.loads(z.read("inventory.json"));assert inv["gpu"].split(",")[0].strip()=="GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa" and not inv["compute"]
    prefix="group_teacher/"+Path(a.remote_root).name+"/"
    required=["plan.json","mu_input.npz","diagnose_scalar_direction_geometry_v1.py","run_scalar_geometry_phase_v1.py","audit_scalar_geometry_v1.py","precheck/receipt.json","precheck/independent_audit.json","precheck/inventory.json","precheck_exit.json","precheck_launch.json","precheck.log"]
    if a.phase=="execute":required+=["execution_authorization.json","execute/receipt.json","execute/independent_audit.json","execute/geometry_frozen.npz","execute/inventory.json","execute_exit.json","execute_launch.json","execute.log"]
    for name in required:assert z.read(prefix+name)==(a.originals/name).read_bytes(),name
    with zipfile.ZipFile(a.prior) as old:
        om=json.loads(old.read("member_manifest.json"));ol=json.loads(old.read("large_file_manifest.json"))
    count=0
    for name,meta in om.items():
        if name in skip:continue
        assert m[name]==meta,"Changed old scientific member: "+name
        count+=1
    large=json.loads(z.read("large_file_manifest.json"))
    for name,item in ol.items():assert large[name]["sha256"]==item["sha256"] and large[name]["bytes"]==item["bytes"],name
result=dict(status="ACTUAL_SCALAR_GEOMETRY_CAPTURE_ORIGINALS_AND_OLD_FROZEN_EVIDENCE_VERIFIED",phase=a.phase,actual_capture_utc=r["utc"],zip_sha256=r["sha256"],zip_bytes=r["bytes"],members=len(m),new_original_required_members=len(required),old_members_unchanged=count,old_large_weight_sha_references_unchanged=len(ol),fullweights_new_download=False,actual_capture_exit_code=0,source_sha256=sha(Path(__file__).read_bytes()))
a.out.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result))

