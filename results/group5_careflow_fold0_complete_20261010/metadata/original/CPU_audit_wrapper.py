import subprocess,json,datetime,pathlib,os
r=pathlib.Path('/data/coding/g5careflow100_20261010T043810Z')
e=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
c=['/data/coding/batch5_continue_actual_20261009T163550Z/N2_public_runtime_actual_20261009T174820Z/.venv/bin/python', '-B', '/data/coding/g5careflow100_20261010T043810Z/source/group5_direct_CPU_audit_v2.py', '--root', '/data/coding/g5careflow100_20261010T043810Z', '--plan-sha', '443b6eb01ba6f4f3f2e14119d93715abe8a8a6a2590019a42dc05a33f91e6845', '--out', '/data/coding/g5careflow100_20261010T043810Z/independent_CPU_audit.json']
p=subprocess.run(c,env=e)
(r/'CPU_exit_observed.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),natural_exit=p.returncode)))
