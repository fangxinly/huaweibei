from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,subprocess,time,zipfile
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['none','fixed','predicted'],required=True);a.add_argument('--stamp',required=True);a.add_argument('--uuid',required=True);c=a.parse_args();assert c.stamp.isalnum()
base=Path('/data/coding/selective_flow');src=base/'inflow_counterfactual_v5_deployment_20261005T0520Z'/('run_'+c.mode);out=base/('counterfactual_provisional_'+c.stamp)/('run_'+c.mode);out.mkdir(parents=True,exist_ok=False)
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().strip();assert uuid==c.uuid
assert shutil.disk_usage(out).free>2_000_000_000
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
attempts=[]
for i in range(3):
 before=(src/'best.pt').stat()
 if before.st_size<700_000_000 or (src/'history.json').stat().st_mtime_ns<before.st_mtime_ns:
  attempts.append({'attempt':i+1,'stable':False,'reason':'Checkpoint still incomplete or empty','bytes':before.st_size});time.sleep(2);continue
 hist=(src/'history.json').read_bytes();h=json.loads(hist);assert 1<=len(h)<=100
 best=min(h,key=lambda x:x['valid_mse']);source_sha_before=sha(src/'best.pt');target=out/('attempt_'+str(i+1)+'.pt');shutil.copyfile(src/'best.pt',target)
 after=(src/'best.pt').stat();source_sha_after=sha(src/'best.pt');target_sha=sha(target);newhist=json.loads((src/'history.json').read_bytes());newbest=min(newhist,key=lambda x:x['valid_mse'])
 stable=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and source_sha_before==source_sha_after==target_sha and (src/'history.json').stat().st_mtime_ns>=after.st_mtime_ns and best['epoch']==newbest['epoch'] and best['valid_mse']==newbest['valid_mse']
 attempts.append({'attempt':i+1,'stable':stable,'source_sha_before':source_sha_before,'source_sha_after':source_sha_after,'copy_sha':target_sha})
 if stable and target.stat().st_size>700_000_000 and zipfile.is_zipfile(target):
  with zipfile.ZipFile(target) as z:
   assert z.testzip() is None and len(z.namelist())>100
  break
else:raise RuntimeError('No stable provisional weight; failed attempts retained')
os.link(target,out/'best.pt');(out/'source_history.json').write_bytes(hist)
for name in ['protocol.json','batch_orders.npy','shared_phase.json']:
 if (src/name).exists():shutil.copyfile(src/name,out/('source_'+name))
files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.iterdir() if p.is_file() and not p.name.startswith('attempt_')}
report={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'STABLE_PROVISIONAL_FULL_WEIGHT_AND_MATCHING_BEST_SO_FAR_HISTORY_SHA_VERIFIED','mode':c.mode,'gpu_uuid':uuid,'epochs_observed':len(h),'best_epoch_so_far':best['epoch'],'best_dev_batch_mse_so_far':best['valid_mse'],'source':str(src),'directory':str(out),'files':files,'attempts':attempts,'training_finished':False,'limits':'Provisional backup; not100-epoch completion or final model selection. No optimizer recovery, no inference or TEST. Training source untouched; failed attempts retained.'}
with (out/'provisional_receipt.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report))
