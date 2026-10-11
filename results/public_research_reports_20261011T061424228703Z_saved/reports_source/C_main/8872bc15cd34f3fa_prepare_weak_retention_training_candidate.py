import ast,hashlib,json,shutil,sys,zipfile
from pathlib import Path
clock=sys.argv[1];base=Path('D:/CodexBackups/selective_flow_20261003_1105');stamp=clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z';d=base/('weak_retention_training_candidate_'+stamp);d.mkdir();bundle=d/'source';bundle.mkdir()
src=Path('work/group5_flow_preparation_20261007T080856Z/source')
for p in src.iterdir():
 if p.is_file() and p.suffix=='.py':shutil.copy2(p,bundle/p.name)
original=(src/'anchored_flow.py').read_text(encoding='utf8');shutil.copy2(src/'anchored_flow.py',d/'original_anchored_flow.py')
old="    total=parts['final_mse']+.25*parts['final_mae']+.5*parts['source_mse']+.1*parts['nonzero_sign']\n    return total,parts"
new="""    weak=y.abs()<=1
    # Only FIT training labels identify weak samples; inference is unchanged.
    # Detach the comparator so this term cannot reduce itself by a direct
    # gradient that makes the base prediction worse.
    excess=(p-y).square()-(b.detach()-y).square()
    retention=F.relu(excess[weak]).mean() if weak.any() else p.sum()*0
    parts['weak_harm_excess']=retention
    total=parts['final_mse']+.25*parts['final_mae']+.5*parts['source_mse']+.1*parts['nonzero_sign']+retention
    return total,parts"""
assert original.count(old)==1;candidate=original.replace(old,new)
ast.parse(candidate);(bundle/'anchored_flow.py').write_text(candidate,encoding='utf8')
checks='''import json,torch
from anchored_flow import objective,synthetic_check
def extra(p,y,b):
 return objective(p,y,b)[1]['weak_harm_excess']
p=torch.tensor([1.],requires_grad=True);b=torch.tensor([.1],requires_grad=True);y=torch.tensor([0.]);v=extra(p,y,b)
gp,gb=torch.autograd.grad(v,(p,b),allow_unused=True)
assert abs(v.item()-.99)<1e-6 and abs(gp.item()-2)<1e-6 and gb is None
assert extra(torch.tensor([.1]),torch.tensor([0.]),torch.tensor([1.])).item()==0
assert extra(torch.tensor([3.]),torch.tensor([2.]),torch.tensor([2.])).item()==0
p=torch.tensor([3.,-3.],requires_grad=True);v=extra(p,torch.tensor([2.,-2.]),torch.tensor([0.,0.]));v.backward();assert torch.equal(p.grad,torch.zeros_like(p))
assert extra(torch.tensor([3.,-3.]),torch.tensor([1.,-1.]),torch.tensor([1.,-1.])).item()==4
result=synthetic_check();result.update(weak_harm_gradient_direction_passed=True,base_comparator_detached=True,strong_only_zero_penalty=True,boundary_abs1_included=True,no_real_data_training=True,no_GPU_used=True)
print(json.dumps(result))
'''
ast.parse(checks);(bundle/'check_weak_retention_native.py').write_text(checks,encoding='utf8')
wrapper='''import subprocess,json,sys,hashlib,datetime,zipfile,shutil,os
from pathlib import Path
bundle=Path(__file__).resolve().parent;plan=json.loads((bundle/'candidate_protocol.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert plan['execution_enabled'] is False
for n,h in plan['source_sha256'].items():assert sha(bundle/n)==h
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().strip();assert uuid=='GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f'
root=bundle.parent/'native_check';root.mkdir();argv=[sys.executable,str(bundle/'check_weak_retention_native.py')];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
(root/'preflight.json').write_text(json.dumps(dict(actual_utc=utc(),uuid=uuid,processes=subprocess.check_output(['ps','-eo','pid,args','--width','4000']).decode(),compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']).decode(),space=shutil.disk_usage('/data')._asdict())))
with (root/'stdout.log').open('wb') as out,(root/'stderr.log').open('wb') as err:
 child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,cwd=bundle);(root/'actual_child.json').write_text(json.dumps(dict(pid=child.pid,fullargv=argv,actual_start_utc=utc())));code=child.wait(timeout=60)
(root/'natural_exit.json').write_text(json.dumps(dict(pid=child.pid,fullargv=argv,natural_exit=code,actual_exit_utc=utc())))
if code==0:assert json.loads((root/'stdout.log').read_text())['weak_harm_gradient_direction_passed']
capture=bundle.parent/'complete_native_capture.zip'
files=[p for p in bundle.parent.rglob('*') if p.is_file() and p!=capture];manifest={p.relative_to(bundle.parent).as_posix():sha(p) for p in files};(bundle.parent/'capture_manifest.json').write_text(json.dumps(dict(actual_utc=utc(),member_sha256=manifest)))
with zipfile.ZipFile(capture,'x',zipfile.ZIP_DEFLATED) as z:
 for p in files+[bundle.parent/'capture_manifest.json']:z.write(p,p.relative_to(bundle.parent).as_posix())
with zipfile.ZipFile(capture) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
(bundle.parent/'capture_receipt.json').write_text(json.dumps(dict(actual_utc=utc(),sha256=sha(capture),natural_exit=code)))
print('NATIVE_CHECK_NATURAL_EXIT_'+str(code),flush=True);raise SystemExit(code)
'''
ast.parse(wrapper);(bundle/'run_native_synthetic_check.py').write_text(wrapper,encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=dict(status='WEAK_RETENTION_TRAINING_CANDIDATE_GPU_DISABLED',actualclock_UTC=clock,execution_enabled=False,human_request='直接针对流修正损害弱情绪继续优化',single_change='Add mean relu((p-y)^2-(stop_gradient(b)-y)^2) on FIT rows abs(y)<=1, coefficient fixed1. All inference and previous losses/optimizer unchanged.',source_sha256={p.name:sha(p) for p in bundle.iterdir() if p.is_file()},prior_anchored_source_SHA=sha(src/'anchored_flow.py'),no_true_labels_at_inference=True,no_TEST_or_OUTER_tuning=True,whole_video_folds_required=True,no_new_performance_claim=True,native_synthetic_check_only=True,full_training_not_started=True,resource_gate='Fresh lease/asset/source/space; full model optimizer states and at least2h saving; current lease time and D capacity insufficient',limitations=['Finite penalty is not a risk bound','Shared representation can change strong predictions','Detached comparison does not freeze the backbone','FIT/INNER history already explored; no fresh independent validation'])
(bundle/'candidate_protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf8')
with zipfile.ZipFile(d/'complete_candidate_source.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in bundle.iterdir():z.write(p,p.name)
with zipfile.ZipFile(d/'complete_candidate_source.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
rec=dict(D=str(d),bundle=str(bundle),source_zip_SHA=sha(d/'complete_candidate_source.zip'),plan_SHA=sha(bundle/'candidate_protocol.json'),stamp=stamp,actualclock=clock)
(d/'freeze_receipt.json').write_text(json.dumps(rec,indent=2),encoding='utf8');Path('work/weak_retention_current.json').write_text(json.dumps(rec,indent=2),encoding='utf8');print(json.dumps(rec))
