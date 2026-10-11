"""Independent NumPy auditor; never runs a model or reads task outcome labels."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8"))
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--phase",choices=["precheck","execute"],required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    plan=read(a.root/"plan.json");receipt=read(a.root/a.phase/"receipt.json")
    launch=read(a.root/(a.phase+"_launch.json"));ex=read(a.root/(a.phase+"_exit.json"))
    assert ex["exit_code"]==0 and ex["child_pid"]==launch["child_pid"]
    assert launch["actual_proc_argv"]==launch["argv"]
    assert "--phase" in launch["argv"] and launch["argv"][-1]==a.phase
    assert sha(a.root/"diagnose_scalar_direction_geometry_v1.py")==plan["source_sha256"]==receipt["source_sha256"]
    assert receipt["plan_sha256"]==sha(a.root/"plan.json") and receipt["passed"]
    assert receipt["model_state_before"]==receipt["model_state_after"] and receipt["no_parameter_gradients"]
    assert receipt["optimizer_steps"]==0 and receipt["actual_peak_allocated_bytes"]<=plan["maximum_peak_allocated_bytes"]
    assert receipt["seconds"]<plan["maximum_seconds"]
    for e in receipt["evidence"]:assert e["poisoned_label_replacement_max_error"]==0 and e["lambda_zero_max_gradient"]==0
    assert any(620 in e["rows"] and e["valid_lengths"][e["rows"].index(620)]==1 for e in receipt["evidence"])
    assert all(e["relative_error"]<1e-4 for e in receipt["double_finite_difference_hvp"])
    result=dict(status="ORIGINAL_SCALAR_GEOMETRY_"+a.phase.upper()+"_INDEPENDENT_AUDIT_PASSED",
        phase=a.phase,source_sha256=receipt["source_sha256"],plan_sha256=sha(a.root/"plan.json"),
        original_gpu_receipt_sha256=sha(a.root/a.phase/"receipt.json"),
        original_natural_exit_sha256=sha(a.root/(a.phase+"_exit.json")),actual_child_pid=ex["child_pid"],
        no_cpu_model_forward=True,no_task_outcome_labels=True)
    if a.phase=="execute":
        path=a.root/"execute/geometry_frozen.npz"
        assert sha(path)==receipt["geometry_sha256"]
        with np.load(path,allow_pickle=False) as z:v={k:z[k] for k in z.files}
        assert np.array_equal(v["row"],np.arange(1281)) and len(np.unique(v["video"]))==52
        assert "y" not in v and all(np.isfinite(x).all() for k,x in v.items() if x.dtype.kind not in "US")
        n=1281;shape=(n,-1)
        j=v["jacobian"].astype(np.float64);go=v["old_gradient"].astype(np.float64);gt=v["teacher_gradient"].astype(np.float64)
        factor_old=2*float(v["weight"])*(v["pf"].astype(np.float64)-v["target_old"])
        factor_teacher=2*float(v["weight"])*(v["pf"].astype(np.float64)-v["mu"])
        errors={}
        for name,g,factor in [("old",go,factor_old),("teacher",gt,factor_teacher)]:
            expected=j*factor[:,None,None]
            rel=np.linalg.norm((g-expected).reshape(shape),axis=1)/(1+np.linalg.norm(expected.reshape(shape),axis=1))
            assert rel.max()<1e-5
            errors[name]=float(rel.max())
        lambda_err=0;unit_err=0
        assert np.max(np.abs(v["lambda_gradient"][0]))==0
        for i,lam in enumerate(v["lambda_values"]):
            expected=gt*lam
            g=v["lambda_gradient"][i].astype(np.float64)
            error=np.linalg.norm((g-expected).reshape(shape),axis=1)/(1+np.linalg.norm(expected.reshape(shape),axis=1))
            lambda_err=max(lambda_err,float(error.max()));assert error.max()<1e-5
            if lam>0:
                diff=np.max(np.abs(v["normalized_lambda_gradient"][i]-v["normalized_lambda_gradient"][-1]))
                unit_err=max(unit_err,float(diff));assert diff<1e-5
        oldnorm=np.linalg.norm(go.reshape(shape),axis=1);newnorm=np.linalg.norm(gt.reshape(shape),axis=1)
        valid=(oldnorm>1e-12)&(newnorm>1e-12)
        cosine=np.sum(go.reshape(shape)*gt.reshape(shape),axis=1)[valid]/(oldnorm[valid]*newnorm[valid])
        sign=np.sign(factor_old[valid]*factor_teacher[valid])
        assert np.max(np.abs(cosine-sign))<1e-5
        result.update(rows=1281,videos=52,arrays_sha256=sha(path),analytic_gradient_max_relative_errors=errors,
            scalar_gradient_relation_error=lambda_err,positive_lambda_normalized_direction_max_error=unit_err,
            nonzero_both_rows=int(valid.sum()),same_direction_rows=int((cosine>0).sum()),
            opposite_direction_rows=int((cosine<0).sum()),same_direction_fraction=float(np.mean(cosine>0)),
            signed_collinearity_max_error=float(np.max(np.abs(cosine-sign))),
            pf_minus_p0_rms=float(np.sqrt(np.mean((v["pf"].astype(np.float64)-v["p0"])**2))),
            interpretation="At the same base message point, scalar squared targets change Jacobian sign/magnitude, not a new normalized geometric direction; positive lambda normalization cancels magnitude. Multi-step nonlinear trajectories not tested here. No y-dependent quality/performance metric.")
    a.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))
if __name__=="__main__":main()

