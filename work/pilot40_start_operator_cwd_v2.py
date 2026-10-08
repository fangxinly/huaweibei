"""Limited correction: original operator used data/coding instead of bundle cwd."""
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

stamp='20261007T091746Z';base=Path('/data/coding')
bundle=base/('anchored_flow_pilot40_source_'+stamp);assets=base/'multimodal_flow_public_20261006T1341Z'
python=assets/'.venv/bin/python';archive=bundle.with_suffix('.zip')
plan_sha='82d2de9439766b2f981f92703d1c34ac6b30c2ef52911e603bf71c22c2a7330a'
if hashlib.sha256(archive.read_bytes()).hexdigest()!='a8b1fdb9aad61c5deb52a00c9d2581d659bc5fc38ae4dee4c5b607ff7c7be675':
    raise ValueError('Original payload differs')
with zipfile.ZipFile(archive) as z:
    if z.testzip():raise ValueError('Original source ZIP CRC differs')
    for name in z.namelist():
        if hashlib.sha256(z.read(name)).digest()!=hashlib.sha256((bundle/name).read_bytes()).digest():
            raise ValueError('Original extracted member differs '+name)
prior=base/('anchored_flow_pilot40_operator_'+stamp)
tests=(prior/'native_tests.stderr.log').read_text()
if 'Ran 32 tests' not in tests or not tests.rstrip().endswith('OK'):
    raise PermissionError('Original completed native tests absent')
args=SimpleNamespace(protocol=bundle/'pilot40_execution_protocol.json',protocol_sha=plan_sha,
    bundle=bundle,assets=assets,root=base/('anchored_flow_pilot40_run_'+stamp),node='A',method='anchored_flow',fold=0,stage='train')
execution=base/('anchored_flow_pilot40_execution_'+stamp)
if args.root.exists() or execution.exists():raise PermissionError('Existing training root forbids redispatch')
sys.path.insert(0,str(bundle));os.chdir(bundle)
from fold_runtime import validate,write
plan,physical=validate(args)
checks=base/('anchored_flow_pilot40_operator_cwd_v2_'+stamp);checks.mkdir()
write(checks/'actual_native_preflight.json',dict(physical,protocol_sha256=plan_sha,
    original_native_tests_sha256=hashlib.sha256((prior/'native_tests.stderr.log').read_bytes()).hexdigest(),
    limited_operator_cwd_correction=True,training_model_source_unchanged=True,
    fullargv=[sys.executable]+sys.argv,pid=os.getpid(),operator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
argv=[str(python),str(bundle/'fold_runtime.py'),'--protocol',str(args.protocol),'--protocol-sha',plan_sha,
    '--bundle',str(bundle),'--assets',str(assets),'--root',str(args.root),'--node','A','--method','anchored_flow','--fold','0','--stage','train']
write(bundle/'pilot_worker_argv.json',argv)
command=[str(python),str(bundle/'fold_evidence.py'),'run','--bundle',str(bundle),'--protocol',str(args.protocol),
    '--protocol-sha',plan_sha,'--execution',str(execution),'--argv-json',str(bundle/'pilot_worker_argv.json')]
with (checks/'wrapper.stdout.log').open('wb') as out,(checks/'wrapper.stderr.log').open('wb') as err:
    wrapper=subprocess.Popen(command,stdout=out,stderr=err,stdin=subprocess.DEVNULL,cwd=bundle,env=env,start_new_session=True)
write(checks/'actual_wrapper_dispatch.json',dict(actual_dispatch_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    wrapper_pid=wrapper.pid,wrapper_fullargv=command,expected_child_fullargv=argv,protocol_sha256=plan_sha))
print(json.dumps({'status':'PILOT40_WRAPPER_DISPATCHED_WORKER_STATUS_MUST_BE_OBSERVED','wrapper_pid':wrapper.pid,
                  'run_root':str(args.root),'execution_root':str(execution)},ensure_ascii=False),flush=True)
