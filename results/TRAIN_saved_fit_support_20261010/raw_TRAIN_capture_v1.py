"""Capture actual raw TRAIN driver exit and preserve complete original outputs."""
import datetime,hashlib,json,pathlib,subprocess,sys,zipfile,os
P=pathlib.Path
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def seal(root,name):
 target=root/name;members=[]
 with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(root.rglob('*')):
   if not p.is_file() or p==target:continue
   b=p.read_bytes();n=p.relative_to(root).as_posix();z.writestr(n,b);members.append(dict(name=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
  z.writestr('capture_member_manifest.json',raw(members))
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())=={v['name'] for v in members}|{'capture_member_manifest.json'}
  for v in members:
   b=z.read(v['name']);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 return dict(archive=str(target),archive_SHA=sha(target),all_member_SHA_CRC_unique_exact_set_passed=True)
if __name__=='__main__':
 planpath=P(sys.argv[1]);expected=sys.argv[2];assert sha(planpath)==expected
 plan=json.loads(planpath.read_bytes());base=P(plan['root']).parent
 assert not (base/'natural_exit.json').exists()
 env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (base/'driver.out').open('xb') as out:
  p=subprocess.Popen([plan['interpreter'],'-B',str(planpath.parent/'TRAIN_raw_probe_driver_v1.py'),'--plan',str(planpath),'--plan-sha',expected],stdout=out,stderr=subprocess.STDOUT,env=env)
  (base/'dispatch.json').write_bytes(raw(dict(actual_UTC=started,child=p.pid,plan_SHA=expected,thread_env={k:env[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},no_AB_reexecution=True)))
  code=p.wait()
 exitrec=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=p.pid,natural_exit=code,plan_SHA=expected)
 (base/'natural_exit.json').write_bytes(raw(exitrec))
 if code==0:
  result=json.loads((P(plan['root'])/'prediction_result.json').read_bytes())
  assert result['status']=='RAW_TRAIN_FIXED_FIVEFOLD_PREDICTIONS_COMPLETE_METRICS_NOT_YET_COMPUTED' and not result['VAL_TEST_numeric_decode']
 receipt=seal(base,'complete_actual_raw_TRAIN_capture.zip');receipt.update(exitrec)
 (base/'wrapper_capture_receipt.json').write_bytes(raw(receipt));print(json.dumps(receipt),flush=True)
 sys.exit(code)
