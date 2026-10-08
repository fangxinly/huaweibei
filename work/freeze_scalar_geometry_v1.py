from pathlib import Path
import ast,datetime,hashlib,json,shutil
import numpy as np
base=Path(__file__).resolve().parents[1];work=base/"work"
D=Path("D:/CodexBackups/selective_flow_20261003_1105")
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime("%Y%m%dT%H%M%SZ")
free={x:shutil.disk_usage(x).free for x in ("C:/","D:/")}
assert min(free.values())>=1024**3
out=D/("scalar_geometry_preparation_"+stamp);out.mkdir(exist_ok=False)
remote="/data/coding/group_teacher_v1_20261005T1650Z/scalar_geometry_diagnostic_v1_"+stamp
old=json.loads((work/"train_oracle_plan_v2.json").read_text(encoding="utf-8"))
mu_path=D/"group_teacher_oof_20261006T0308Z/train_scalar_oof.npz"
assert sha(mu_path)=="cff1a52f7e828c96cacf070ae4e051a5c34042ebb9435f44f5b3242cad91a7ea"
with np.load(mu_path,allow_pickle=False) as z:
    assert set(z.files)=={"row_ids","fold","mu"}
    np.savez_compressed(out/"mu_input.npz",**{k:z[k] for k in z.files})
names=["diagnose_scalar_direction_geometry_v1.py","run_scalar_geometry_phase_v1.py","audit_scalar_geometry_v1.py","run_same_teacher_capture_v1.py"]
for name in names:
    src=work/name
    ast.parse(src.read_text(encoding="utf-8"));shutil.copy2(src,out/name)
tree=ast.parse((out/names[0]).read_text(encoding="utf-8"))
for x in ast.walk(tree):
    if isinstance(x,ast.Subscript) and isinstance(x.slice,ast.Constant):
        assert not (x.slice.value=="y" and isinstance(x.value,ast.Name) and x.value.id=="cache")
pins=old["pinned_files"].copy()
pins["/data/coding/train_oracle_diagnostic_20261006T014605Z/diagnose_train_oracle_v2.py"]=old["diagnostic_source_sha256"]
pins["/data/coding/train_oracle_diagnostic_20261006T014605Z/out/diagnostics.npz"]="92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658"
pins["/data/coding/soft_vector_research_20261005T1220Z/capture_soft_vector_v19.py"]="6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad"
for file in out.iterdir():pins[remote+"/"+file.name]=sha(file)
plan=dict(status="FROZEN_SCALAR_FIRST_STEP_GEOMETRY_LABEL_FREE",frozen_utc=now.isoformat(),new_root=remote,
    base_root=old["base_root"],formal_root=old["formal_root"],mapping_path=old["mapping_path"],
    helper_source="/data/coding/train_oracle_diagnostic_20261006T014605Z/diagnose_train_oracle_v2.py",
    old_diagnostic_arrays="/data/coding/train_oracle_diagnostic_20261006T014605Z/out/diagnostics.npz",
    expected_uuid=old["expected_uuid"],source_sha256=sha(out/names[0]),pinned_files=pins,
    lambda_values=[0,.125,.25,.5,1],rows=1281,videos=52,batch_size=32,
    maximum_seconds=900,maximum_peak_allocated_bytes=1024**3,minimum_remote_free_bytes=2*1024**3,
    preservation_reserve_seconds=7200,estimated_lease_end_utc=old["estimated_lease_end_utc"],
    platform_lease_verified=False,
    labels="No task y read. Only label-free cached coordinates/p0/old trained scalar and OOF mu; poisoned dummy y ignored. No CAL fit, EVAL metric, DEV or TEST.",
    scope="Mechanism first-step Jacobian/sign/normalization audit, not a new controlled prediction or trained student. C2 is fullTRAINfit/DEVselected; not whole-pipeline crossfit.",
    precheck="Full32, TRAIN620 one-token plus621, tail1280; poisoned label identity0; original-F/rho replay1e-6; double clone true finite-difference HVP1e-4; parameters SHA unchanged; execute only after independent original precheck and actual capture.",
    reference_saving="No old fullweight transfer/retraining. New teacher subroot recursively inside frozen capture19.",
    independent_review_input="New review commentary suggested testing scalar/Jacobian collinearity and lambda normalization cancellation. Treat as a hypothesis until actual GPU arrays independently audited.")
(out/"plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()}
proof=dict(status="FROZEN_SCALAR_GEOMETRY_PREPARATION_NOT_DEPLOYED_OR_GPU_EXECUTED",actual_utc=now.isoformat(),remote_root=remote,local_directory=str(out),manifest=manifest,fresh_free_bytes=free,AST_no_cache_y=True)
(base/"outputs/首步标量方向几何短诊断冻结接续.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding="utf-8")
(out/"preparation_receipt.json").write_text(json.dumps(proof,indent=2),encoding="utf-8")
print(json.dumps(proof,ensure_ascii=True))

