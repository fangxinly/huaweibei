import pathlib,hashlib,json,zipfile,subprocess,os,datetime
p=pathlib.Path('/data/coding/B_peer_payload_20261009T180428Z.zip');assert hashlib.sha256(p.read_bytes()).hexdigest()=='a6d0d3ec2f91d661b15c19aefb41a581f45c4ac0c5dd7570565a5e6d6b553c9e'
b=pathlib.Path('/data/coding/B_peer_checks_actual_20261009T180720Z');b.mkdir();payload=b/'payload';payload.mkdir()
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n in z.namelist():assert (payload/n).resolve().is_relative_to(payload.resolve())
 z.extractall(payload)
root=b/'N3_peer';assert not root.exists();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');argv=['/data/miniconda/envs/torch/bin/python','-B',str(payload/'B_array_checks_v1.py'),'--payload',str(payload),'--stage','peer','--root',str(root)]
with (b/'N3_peer_wrapper.log').open('wb') as f:child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
print(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid,root=str(root),argv=argv)))
