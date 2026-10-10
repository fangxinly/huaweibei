import subprocess,json,datetime,pathlib,os
r=pathlib.Path('/data/coding/g5careflow1_native_20261010T112228Z')
e=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
a=['/data/coding/batch5_continue_actual_20261009T163550Z/N2_public_runtime_actual_20261009T174820Z/.venv/bin/python', '-B', '/data/coding/g5careflow1_native_20261010T112228Z/source/group5_direct_CPU_audit_v2.py', '--root', '/data/coding/g5careflow1_native_20261010T112228Z', '--plan-sha', '02d0e24c1f9bc385160468bd21f448926b6435e5949ed466d9fd67d1f6a41da3', '--out', '/data/coding/g5careflow1_native_20261010T112228Z/independent_CPU_audit.json']
p=subprocess.run(a,env=e)
(r/'CPU_exit_observed.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),natural_exit=p.returncode)))
