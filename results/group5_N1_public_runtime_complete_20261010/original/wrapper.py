import subprocess,json,pathlib,datetime
r=pathlib.Path('/data/coding/g5_N1_prestaged_runtime_20261010T113812Z')
a=['/data/miniconda/envs/torch/bin/python', '-B', '/data/coding/g5_N1_prestaged_runtime_20261010T113812Z/group5_public_runtime_prestaged_v2.py', '--plan', '/data/coding/g5_N1_prestaged_runtime_20261010T113812Z/plan.json', '--plan-sha', 'cc5f553da512562ce2f90e1c2c55d23b1f4afb85a3da4f639499e64c9d6533a2', '--root', '/data/coding/g5_N1_prestaged_runtime_20261010T113812Z/runtime']
with (r/'stdout.log').open('xb') as f:
 c=subprocess.Popen(a,stdout=f,stderr=subprocess.STDOUT)
 (r/'dispatch.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=c.pid,argv=a)))
 code=c.wait()
(r/'wrapper_exit.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=c.pid,natural_exit=code)))
raise SystemExit(code)
