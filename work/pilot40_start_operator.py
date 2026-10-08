"""One-shot operator: bytes/native contracts/physical gate, then natural worker."""
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

stamp='20261007T091746Z'
base=Path('/data/coding')
bundle=base/('anchored_flow_pilot40_source_'+stamp)
archive=bundle.with_suffix('.zip')
plan_sha='82d2de9439766b2f981f92703d1c34ac6b30c2ef52911e603bf71c22c2a7330a'
zip_sha='a8b1fdb9aad61c5deb52a00c9d2581d659bc5fc38ae4dee4c5b607ff7c7be675'
assets=base/'multimodal_flow_public_20261006T1341Z'
python=assets/'.venv/bin/python'
if bundle.exists():raise ValueError('Fresh bundle root required')
if hashlib.sha256(archive.read_bytes()).hexdigest()!=zip_sha:raise ValueError('Uploaded payload differs')
with zipfile.ZipFile(archive) as z:
    if z.testzip() or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Source ZIP invalid')
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):raise ValueError('Unsafe ZIP member')
    z.extractall(bundle)
sys.path.insert(0,str(bundle))
from fold_runtime import validate,write
from types import SimpleNamespace
args=SimpleNamespace(protocol=bundle/'pilot40_execution_protocol.json',protocol_sha=plan_sha,
    bundle=bundle,assets=assets,root=base/('anchored_flow_pilot40_run_'+stamp),node='A',
    method='anchored_flow',fold=0,stage='train')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
tests=subprocess.run([str(python),'-B','-m','unittest','test_fold_contract','test_pilot_budget'],
                     cwd=bundle,env=env,capture_output=True,text=True)
checks=base/('anchored_flow_pilot40_operator_'+stamp)
checks.mkdir()
(checks/'native_tests.stdout.log').write_text(tests.stdout)
(checks/'native_tests.stderr.log').write_text(tests.stderr)
if tests.returncode:raise RuntimeError('Native contract tests failed: '+tests.stderr)
plan,physical=validate(args)
write(checks/'actual_native_preflight.json',dict(physical,protocol_sha256=plan_sha,native_tests_natural_exit=tests.returncode,
    fullargv=[sys.executable]+sys.argv,pid=os.getpid(),operator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
argv=[str(python),str(bundle/'fold_runtime.py'),'--protocol',str(args.protocol),'--protocol-sha',plan_sha,
    '--bundle',str(bundle),'--assets',str(assets),'--root',str(args.root),'--node','A','--method','anchored_flow','--fold','0','--stage','train']
write(bundle/'pilot_worker_argv.json',argv)
execution=base/('anchored_flow_pilot40_execution_'+stamp)
command=[str(python),str(bundle/'fold_evidence.py'),'run','--bundle',str(bundle),'--protocol',str(args.protocol),
    '--protocol-sha',plan_sha,'--execution',str(execution),'--argv-json',str(bundle/'pilot_worker_argv.json')]
with (checks/'wrapper.stdout.log').open('wb') as out,(checks/'wrapper.stderr.log').open('wb') as err:
    wrapper=subprocess.Popen(command,stdout=out,stderr=err,stdin=subprocess.DEVNULL,cwd=bundle,env=env,start_new_session=True)
write(checks/'actual_wrapper_dispatch.json',dict(actual_dispatch_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    wrapper_pid=wrapper.pid,wrapper_fullargv=command,expected_child_fullargv=argv,protocol_sha256=plan_sha))
print(json.dumps({'status':'PILOT40_WRAPPER_DISPATCHED_WORKER_STATUS_MUST_BE_OBSERVED','wrapper_pid':wrapper.pid,
                  'run_root':str(args.root),'execution_root':str(execution)},ensure_ascii=False),flush=True)
