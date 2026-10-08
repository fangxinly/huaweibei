"""One fixed40 train dispatcher; fresh physical gates and natural-exit wrapper."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args()
plan=json.loads(a.plan.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
assert sha(__file__)==plan['operator_source_SHA'] and plan['stage']=='train'
archive=Path(plan['archive']);assert sha(archive)==plan['archive_SHA']
bundle=Path(plan['bundle']);assert not bundle.exists();bundle.mkdir()
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for n in z.namelist():
        q=(bundle/n).resolve();assert q.is_relative_to(bundle.resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(z.read(n))
assets=Path(plan['assets']);native=str(assets/'.venv/bin/python');assert sys.executable==native
protocol=bundle/'training_execution_protocol.json';assert sha(protocol)==plan['protocol_SHA']
os.chdir(bundle);sys.path.insert(0,str(bundle))
from fold_runtime import validate,write
args=SimpleNamespace(protocol=protocol,protocol_sha=plan['protocol_SHA'],bundle=bundle,
    assets=assets,root=Path(plan['root']),node='A',method='anchored_flow',fold=0,stage='train')
training,physical=validate(args)
record=Path(plan['operator_record']);record.mkdir()
shutil.copy2(__file__,record/Path(__file__).name);shutil.copy2(a.plan,record/a.plan.name)
write(record/'actual_physical_preflight.json',dict(physical,operator_source_SHA=sha(__file__),
    operator_fullargv=[sys.executable]+sys.argv,operator_pid=os.getpid()))
argv=[native,str(bundle/'fold_runtime.py'),'--protocol',str(protocol),'--protocol-sha',plan['protocol_SHA'],
    '--bundle',str(bundle),'--assets',str(assets),'--root',plan['root'],'--node','A','--method','anchored_flow','--fold','0','--stage','train']
write(record/'worker_argv.json',argv)
wrapper=[native,str(bundle/'fold_evidence.py'),'run','--bundle',str(bundle),'--protocol',str(protocol),
    '--protocol-sha',plan['protocol_SHA'],'--execution',plan['execution'],'--argv-json',str(record/'worker_argv.json')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (record/'wrapper_stdout.log').open('wb') as out,(record/'wrapper_stderr.log').open('wb') as err:
    child=subprocess.Popen(wrapper,stdout=out,stderr=err,cwd=bundle,env=env,start_new_session=True)
write(record/'actual_dispatch.json',dict(status='ACTUAL_SINGLE_FIXED40_NATURAL_TRAIN_WRAPPER_DISPATCHED_NOT_COMPLETE',
    actual_utc=utc(),wrapper_pid=child.pid,wrapper_fullargv=wrapper,worker_fullargv=argv,physical=physical))
print(json.dumps(dict(actual_utc=utc(),wrapper_pid=child.pid,worker_fullargv=argv)),flush=True)
