import ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1];info=json.loads((root/'outputs/第二租期预算计量v4本地候选最新.json').read_text());d=Path(info['directory']);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=(root/'work/launch_second_lease_fold0_precheck_v1.py').read_text(encoding='utf-8');name=d.name
new=old.replace('second_lease_fixed_execution_v1_20261006T1349Z',name).replace('deployment_execution_plan_v1.json','deployment_execution_plan_v2.json').replace('7cb0ed9e30f5189ba197831d3c86db1218dbb7bde88789e3afaced2ada3db0a3',info['execution_plan_sha256']).replace('second_lease_precheck_wrapper_v1.py','second_lease_precheck_wrapper_v2.py').replace('minimal_fixed_fold0_precheck_actual_','minimal_fixed_fold0_budget_trace_v4_actual_')
(d/'launch_second_lease_budget_trace_v4.py').write_text(new,encoding='utf-8');ast.parse(new)
for p in d.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
zpath=d.with_suffix('.zip');members={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in d.iterdir() if p.is_file()}
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for n in members:z.write(d/n,n)
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)
 for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
r={'actual_frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'directory':str(d),'package':str(zpath),'package_sha256':sha(zpath),'members':members,'launcher_sha256':sha(d/'launch_second_lease_budget_trace_v4.py'),'execution_plan_sha256':info['execution_plan_sha256'],'actual_GPU_executed':False,'science_and_budget_changed':False};(root/'outputs/第二租期预算计量v4执行包最新.json').write_text(json.dumps(r,indent=2)+'\n')
base=Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_precheck_failure_actual_20261006T141445Z');shutil.copy2(zpath,base/zpath.name);assert sha(base/zpath.name)==sha(zpath)
print(json.dumps({k:v for k,v in r.items() if k!='members'}))
