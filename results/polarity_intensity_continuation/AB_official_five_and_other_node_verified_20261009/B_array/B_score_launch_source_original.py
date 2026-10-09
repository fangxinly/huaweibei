import pathlib,hashlib,json,zipfile,subprocess,os,datetime
p=pathlib.Path('/data/coding/batch5_continue_actual_20261009T163550Z/B_score_payload_20261009T180428Z.zip');assert hashlib.sha256(p.read_bytes()).hexdigest()=='23debd75c156fce7fc3a2004ea5edb819a3d1450dc3752f26032e9a67f1d5813'
b=p.parent/'N2_B_score_actual_20261009T180550Z';b.mkdir();payload=b/'payload';payload.mkdir()
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n in z.namelist():assert (payload/n).resolve().is_relative_to(payload.resolve())
 z.extractall(payload)
plan=json.loads((payload/'plan.json').read_bytes());assert not pathlib.Path(plan['author_once_token']).exists();root=b/'actual';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');argv=['/data/coding/batch5_continue_actual_20261009T163550Z/N2_public_runtime_actual_20261009T174820Z/.venv/bin/python','-B',str(payload/'B_array_checks_v1.py'),'--payload',str(payload),'--stage','score','--root',str(root)]
with (b/'wrapper.log').open('wb') as f:child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
print(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid,root=str(root),argv=argv)))
