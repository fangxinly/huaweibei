import subprocess,json,pathlib,datetime
r=pathlib.Path('/data/coding/g5_N1_public_runtime_20261010T112547Z')
a=['/data/miniconda/envs/torch/bin/python', '-B', '/data/coding/g5_N1_public_runtime_20261010T112547Z/group5_public_runtime_v1.py', '--plan', '/data/coding/g5_N1_public_runtime_20261010T112547Z/plan.json', '--plan-sha', 'b6c4ae0ddbc015cc61c29614eecad5ecbf9a5145ec874f5ce50f9e3fc5935d07', '--root', '/data/coding/g5_N1_public_runtime_20261010T112547Z/runtime']
with (r/'stdout.log').open('xb') as f:
 c=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT)
 (r/'dispatch.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=c.pid,argv=a)))
 code=c.wait()
(r/'wrapper_exit.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=c.pid,natural_exit=code)))
raise SystemExit(code)
