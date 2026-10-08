"""Immutable C2 first-step scalar geometry: no labels, solver path or fitting."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
from pathlib import Path
import argparse, copy, datetime, hashlib, importlib.util, json, shutil, subprocess, sys, time
import numpy as np
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()
def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding="utf-8")
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def unit(value):
    norm=value.flatten(1).norm(dim=1)
    nonzero=norm>0
    out=torch.zeros_like(value)
    out[nonzero]=value[nonzero]/norm[nonzero,None,None]
    return out,norm
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--phase",choices=["precheck","execute"],required=True)
    a=p.parse_args();started=time.monotonic()
    plan=json.loads(a.plan.read_text());root=Path(plan["new_root"])
    assert plan["status"]=="FROZEN_SCALAR_FIRST_STEP_GEOMETRY_LABEL_FREE"
    assert sha(__file__)==plan["source_sha256"]
    for path,value in plan["pinned_files"].items():assert sha(path)==value,path
    out=root/a.phase;out.mkdir(exist_ok=False)
    q=lambda argv:subprocess.check_output(argv,text=True).strip()
    gpu=q(["nvidia-smi","--query-gpu=uuid,name,memory.total,memory.used","--format=csv,noheader,nounits"])
    compute=q(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader,nounits"])
    assert gpu.split(",")[0]==plan["expected_uuid"] and not compute
    now=datetime.datetime.now(datetime.timezone.utc)
    remaining=(datetime.datetime.fromisoformat(plan["estimated_lease_end_utc"])-now).total_seconds()
    assert remaining>plan["maximum_seconds"]+plan["preservation_reserve_seconds"]
    assert shutil.disk_usage(root).free>=plan["minimum_remote_free_bytes"]
    write(out/"inventory.json",dict(utc=now.isoformat(),argv=sys.argv,pid=os.getpid(),gpu=gpu,compute=compute,
        full_process_argv=q(["ps","-ww","-eo","pid,ppid,args"]),free_bytes=shutil.disk_usage(root).free,
        estimated_remaining_seconds=remaining,platform_lease_verified=False))
    if a.phase=="execute":
        auth=json.loads((root/"execution_authorization.json").read_text())
        assert auth["source_sha256"]==sha(__file__) and auth["plan_sha256"]==sha(a.plan)
        assert auth["precheck_sha256"]==sha(root/"precheck/receipt.json")
        assert auth["precheck_audit_sha256"]==sha(root/"precheck/independent_audit.json")
        assert auth["actual_precheck_capture_verified"] is True
        assert json.loads((root/"precheck_exit.json").read_text())["exit_code"]==0
    base=Path(plan["base_root"]);formal=Path(plan["formal_root"])
    sys.path[:0]=[str(formal),str(base)]
    from finite_task_risk_runtime_v2 import FrozenCoordinateLearner,PAIRS
    spec=importlib.util.spec_from_file_location("frozen_oracle_helpers",plan["helper_source"])
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    torch.set_num_threads(2);torch.manual_seed(91817);torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    selection=json.loads((formal/"run/selection.json").read_text())
    assert selection["epochs"]==100 and selection["best_epoch"]==37
    scales=json.loads((formal/"train_scales.json").read_text())
    model=FrozenCoordinateLearner(base,"finite_vector",scales).cuda()
    model.restore_addon(torch.load(formal/"run/best_addon.pt",map_location="cpu"))
    model.eval();model.requires_grad_(False)
    before=helpers.tensor_sha(model.state_dict());fb=model.feedback
    cache=np.load(base/"teacher_cache_v1/train_cache.npz",allow_pickle=False)
    inputs={k:cache[k] for k in ["state","mask","old_context","pooled_state","reference_prediction"]}
    with np.load(root/"mu_input.npz",allow_pickle=False) as z:
        assert set(z.files)=={"row_ids","fold","mu"}
        assert np.array_equal(z["row_ids"],np.arange(1281))
        mu=z["mu"].copy();fold=z["fold"].copy()
    with np.load(plan["old_diagnostic_arrays"],allow_pickle=False) as z:
        old_raw=z["raw_prediction"].copy()
        old_rho=z["estimated_residual"].copy()
        old_p0=z["p0"].copy()
    assert np.array_equal(old_p0,inputs["reference_prediction"])
    mapping=json.loads(Path(plan["mapping_path"]).read_text())
    videos=np.asarray([r["video_id"] for r in mapping])
    assert len(videos)==1281 and len(np.unique(videos))==52
    lambdas=np.asarray(plan["lambda_values"],dtype=np.float64)
    def batch(rows):
        return {k:torch.as_tensor(v[rows],device="cuda") for k,v in inputs.items()}
    def inspect(rows,poison=False):
        b=batch(rows)
        if poison:b["y"]=torch.arange(len(rows),device="cuda").float()+70
        pool=b["pooled_state"].detach()
        with torch.no_grad():
            messages=torch.stack([model.donor[i](torch.cat([pool[:,r],pool[:,d],
                .5*(pool[:,r]-pool[:,d]),.5*(pool[:,r]+pool[:,d])],-1))
                for i,(r,d) in enumerate(PAIRS)],1)
            rho=fb.estimate_residual(b["old_context"],pool,b["reference_prediction"])[1]
        scale=fb.train_message_rms[None,:,None]
        point=(messages/scale).detach().requires_grad_()
        pred=model.terminal(b["state"].detach(),b["mask"],fb.context(b["old_context"].detach(),point*scale))
        pf=pred.detach()
        target=torch.as_tensor(mu[rows],dtype=pred.dtype,device="cuda").double().detach()
        target_old=(b["reference_prediction"]-rho).double().detach()
        assert not target.requires_grad and not target_old.requires_grad and not pf.requires_grad
        jac=torch.autograd.grad(pred.sum(),point,retain_graph=True)[0]
        weight=(fb.finite_beta.double()/fb.train_residual_rms.double().square()).detach()
        prox=.5*(point-point.detach()).square().sum((1,2))
        proxgrad=torch.autograd.grad(prox.sum(),point,retain_graph=True)[0]
        assert float(proxgrad.abs().max())==0
        gradients=[]
        for lam in lambdas:
            anchored=(pf.double()+float(lam)*(target-pf.double())).detach()
            risk=prox.double()+weight*(pred.double()-anchored).square()
            gradients.append(torch.autograd.grad(risk.sum(),point,retain_graph=True)[0])
        old_loss=prox.double()+weight*(pred.double()-target_old).square()
        go=torch.autograd.grad(old_loss.sum(),point,retain_graph=True)[0]
        if len(rows)>1:
            altered=(pf.double()+(target-pf.double())).clone()
            altered[0]+=1
            ag=torch.autograd.grad((prox.double()+weight*(pred.double()-altered.detach()).square()).sum(),point)[0]
            assert float((ag[1:]-gradients[-1][1:]).abs().max())==0,"batch coupling"
        result=dict(row=np.asarray(rows),valid_lengths=b["mask"].sum(-1).cpu().numpy(),
            pf=pf.cpu().numpy(),p0=b["reference_prediction"].cpu().numpy(),mu=target.cpu().numpy(),
            target_old=target_old.cpu().numpy(),rho_old=rho.cpu().numpy(),jacobian=jac.cpu().numpy(),
            old_gradient=go.cpu().numpy(),teacher_gradient=gradients[-1].cpu().numpy(),
            lambda_gradient=torch.stack(gradients).cpu().numpy(),weight=np.asarray(float(weight)),
            normalized_lambda_gradient=torch.stack([unit(g)[0] for g in gradients]).cpu().numpy())
        assert np.isfinite(np.concatenate([v.reshape(-1) for v in result.values()])).all()
        for grad in [go,*gradients]:assert torch.isfinite(grad).all()
        assert np.max(np.abs(result["pf"]-old_raw[rows]))<=1e-6
        assert np.max(np.abs(result["rho_old"]-old_rho[rows]))<=1e-6
        return result
    evidence=[]
    for rows in [np.arange(32),np.asarray([620,621]),np.asarray([1280])]:
        t=time.monotonic();orig=inspect(rows);changed=inspect(rows,True)
        assert all(np.array_equal(orig[k],changed[k]) for k in orig)
        if 620 in rows:assert orig["valid_lengths"][list(rows).index(620)]==1
        norm=np.linalg.norm(orig["teacher_gradient"].reshape(len(rows),-1),axis=1)
        assert np.max(np.abs(orig["lambda_gradient"][0]))==0
        for i,lam in enumerate(lambdas):
            g=orig["lambda_gradient"][i];expected=orig["teacher_gradient"]*lam
            err=np.linalg.norm((g-expected).reshape(len(rows),-1),axis=1)/(1+np.linalg.norm(expected.reshape(len(rows),-1),axis=1))
            assert err.max()<1e-5
            if lam>0 and (norm>0).any():
                assert np.max(np.abs(orig["normalized_lambda_gradient"][i]-orig["normalized_lambda_gradient"][-1]))<1e-5
        evidence.append(dict(rows=rows.tolist(),valid_lengths=orig["valid_lengths"].tolist(),
            poisoned_label_replacement_max_error=0,lambda_zero_max_gradient=0,
            seconds=time.monotonic()-t))
    # Independent double clone only: original parameters never converted or updated.
    dm=copy.deepcopy(model).double();dm.requires_grad_(False);dm.eval()
    b=batch(np.asarray([620,621]))
    bd={k:v.double() if v.is_floating_point() else v for k,v in b.items()}
    with torch.no_grad():
        pool=bd["pooled_state"]
        m=torch.stack([dm.donor[i](torch.cat([pool[:,r],pool[:,d],.5*(pool[:,r]-pool[:,d]),
            .5*(pool[:,r]+pool[:,d])],-1)) for i,(r,d) in enumerate(PAIRS)],1)
    sc=dm.feedback.train_message_rms[None,:,None];u0=(m/sc).detach()
    tar=torch.as_tensor(mu[[620,621]],device="cuda",dtype=torch.float64).detach()
    v=torch.arange(1,u0.numel()+1,device="cuda",dtype=torch.float64).reshape_as(u0)
    v=v/v.norm()
    def dg(u,hessian=False):
        u=u.detach().requires_grad_()
        pred=dm.terminal(bd["state"],bd["mask"],dm.feedback.context(bd["old_context"],u*sc))
        loss=.5*(u-u0).square().sum()+dm.feedback.finite_beta*((pred-tar).square()/dm.feedback.train_residual_rms.square()).sum()
        g=torch.autograd.grad(loss,u,create_graph=hessian)[0]
        return (g,torch.autograd.grad((g*v).sum(),u)[0]) if hessian else g
    g,h=dg(u0,True);fd=[]
    for step in (1e-4,1e-5):
        estimate=(dg(u0+step*v)-dg(u0-step*v))/(2*step)
        error=float((estimate-h).norm()/(1+h.norm()))
        fd.append(dict(step=step,relative_error=error))
        assert torch.isfinite(estimate).all() and error<1e-4
    del dm
    check=dict(passed=True,utc=utc(),phase=a.phase,source_sha256=sha(__file__),plan_sha256=sha(a.plan),
        model_state_before=before,model_state_after=helpers.tensor_sha(model.state_dict()),
        no_parameter_gradients=all(x.grad is None for x in model.parameters()),optimizer_steps=0,
        evidence=evidence,double_finite_difference_hvp=fd,
        actual_peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started)
    assert check["model_state_before"]==check["model_state_after"] and check["no_parameter_gradients"]
    assert check["actual_peak_allocated_bytes"]<=plan["maximum_peak_allocated_bytes"]
    assert check["seconds"]<plan["maximum_seconds"]
    if a.phase=="precheck":
        write(out/"receipt.json",check);print("SCALAR_GEOMETRY_PRECHECK_COMPLETE",flush=True);return
    chunks={}
    for start in range(0,1281,32):
        rows=np.arange(start,min(start+32,1281));values=inspect(rows)
        for k,value in values.items():
            if k=="weight":continue
            axis=1 if "lambda_gradient" in k else 0
            chunks.setdefault(k,[]).append(value)
        assert time.monotonic()-started<plan["maximum_seconds"]
        assert torch.cuda.max_memory_allocated()<=plan["maximum_peak_allocated_bytes"]
    merged={k:np.concatenate(vals,axis=1 if "lambda_gradient" in k else 0) for k,vals in chunks.items()}
    assert np.array_equal(merged["row"],np.arange(1281))
    arrays=out/"geometry_frozen.npz"
    np.savez_compressed(arrays,**merged,lambda_values=lambdas,weight=np.asarray(float(weight if "weight" in locals() else model.feedback.finite_beta.double()/model.feedback.train_residual_rms.double().square())),fold=fold,video=videos)
    check.update(rows=1281,videos=52,batches=41,geometry_sha256=sha(arrays),labels_requested=False,
        model_state_after=helpers.tensor_sha(model.state_dict()),no_parameter_gradients=all(x.grad is None for x in model.parameters()),
        actual_peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
        scope="Fresh first-step gradients/Jacobian and label-free sign geometry only. No control path, labels, CAL fitting, EVAL metric, updated message inference, student training or whole-pipeline crossfit.")
    assert check["model_state_before"]==check["model_state_after"] and check["no_parameter_gradients"]
    write(out/"receipt.json",check)
    print("SCALAR_GEOMETRY_DIAGNOSTIC_COMPLETE",flush=True)
if __name__=="__main__":main()

