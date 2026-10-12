import pathlib,subprocess,json,datetime as dt,sys,hashlib
r=pathlib.Path(__file__).parent;p=json.loads((r/'plan.json').read_bytes())
argv=[sys.executable,'-B',str(r/'runtime.py'),'--plan',str(r/'plan.json'),'--plan-sha',hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),'--root',p['new_runtime_root']]
with (r/'runtime_stdout.log').open('wb') as o,(r/'runtime_stderr.log').open('wb') as e:
 child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=o,stderr=e)
 (r/'runtime_dispatch.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),pid=child.pid,argv=argv)))
 code=child.wait()
(r/'runtime_natural_exit.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),pid=child.pid,natural_exit=code)))
sys.exit(code)
