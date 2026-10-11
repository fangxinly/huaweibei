"""Operator transfer gate for already authorized original CPU audit, no model/labels.

Wait for the two specific whole files, then obtain fresh direct node/source/asset
evidence and launch only the already frozen natural CPU wrapper once.
"""
import argparse,datetime,json,os,subprocess,sys,time,traceback,hashlib,shutil
from pathlib import Path

def utc():return datetime.datetime.now(datetime.timezone.utc)
def stamp():return utc().isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
 p=argparse.ArgumentParser()
 for n in ('root','bundle','assets'):p.add_argument('--'+n,type=Path,required=True)
 for n in ('plan-sha','args-sha','method'):p.add_argument('--'+n,required=True)
 a=p.parse_args();sys.path.insert(0,str(a.bundle))
 from paired_fulltrain_evidence_candidate_v1 import plan_gate,stage_association,zip_audit,require
 plan=plan_gate(a.bundle,a.plan_sha)
 require(a.root.parent==Path('/data/coding') and a.root.name.startswith('paired_fulltrain_cpu_') and
         not (a.root/'out').exists() and not (a.root/'natural_exit.json').exists() and
         not (a.root/'operator_transfer_gate_start.json').exists(),'Fresh single CPU transfer gate required')
 argsfile=a.root/'actual_CPU_child_arguments.json';require(sha(argsfile)==a.args_sha,'Actual D argument SHA')
 args=read(argsfile)
 for flag,value in (('--root',str(a.root)),('--bundle',str(a.bundle)),('--plan-sha',a.plan_sha),('--method',a.method),('--stage','train')):
  require(args.count(flag)==1 and args[args.index(flag)+1]==value,'Frozen CPU physical argument identity')
 original=a.root/'original/run';r,e=stage_association(original,a.plan_sha,a.method)
 require(r['status']=='ACTUAL_PAIRED_METHOD_FULLTRAIN100_COMPLETE_PRESERVATION_PENDING_NO_FINAL_TEST','Original completed fullTRAIN required')
 write(a.root/'operator_transfer_gate_start.json',{'actual_utc':stamp(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),
       'whole_D_argument_SHA':a.args_sha,'original_receipt_SHA':sha(original/'out/actual_stage_receipt.json'),
       'status':'WAITING_FOR_WHOLE_TRANSFER_NO_CPU_AUDIT_LAUNCHED','no_model_or_labels':True})
 started=time.monotonic();deadline=datetime.datetime(2026,10,7,13,30,tzinfo=datetime.timezone.utc)
 while True:
  require(time.monotonic()-started<2700,'Transfer did not finish within45min; no CPU started')
  require((deadline-utc()).total_seconds()>2*3600+900,'Conservative lease execution plus2h preservation gate')
  items={k:original/'out'/Path(r[k]['path']).name for k in ('complete_resume_full','selected_best_full')}
  if all(f.is_file() and f.stat().st_size==r[k]['bytes'] for k,f in items.items()):
   before={k:(f.stat().st_size,f.stat().st_mtime_ns) for k,f in items.items()};time.sleep(5)
   if before=={k:(f.stat().st_size,f.stat().st_mtime_ns) for k,f in items.items()}:break
  time.sleep(15)
 whole={k:zip_audit(f,r[k]['sha256']) for k,f in items.items()}
 # Direct fresh evidence is produced here, rather than replaying preparation JSON.
 plan=plan_gate(a.bundle,a.plan_sha)
 assets={n:sha(a.assets/n) for n in plan['asset_sha256']}
 require(assets==plan['asset_sha256'],'Actual CPU public assets byte SHA gate')
 raw={'actual_utc':stamp(),'whole_file_SHA_CRC_unique':whole,'source_plan_SHA':a.plan_sha,'all_source_SHA_match':True,
      'actual_public_assets_SHA':assets,'conservative_lease_not_platform_confirmation':True,
      'CPU_execution_budget_seconds':900,'minimum_preservation_seconds':7200}
 for name,cmd in (('GPU_UUID',['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']),
                  ('compute',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader']),
                  ('full_process_argv',['ps','-eo','pid,ppid,args']),('space',['df','-B1',str(a.root)])):
  raw[name]=subprocess.run(cmd,capture_output=True,text=True,check=True).stdout
 require(raw['GPU_UUID'].strip()==plan['original_CPU_gpu_UUID'] and not raw['compute'].strip(),'Fresh original B CPU node/empty GPU compute')
 require(shutil.disk_usage(a.root).free>=plan['remote_free_floor_bytes'],'Actual CPU preservation space floor')
 require((deadline-utc()).total_seconds()>2*3600+900 and not (a.root/'out').exists() and not (a.root/'natural_exit.json').exists(),
         'Fresh execution and lease gate, no duplicate CPU run')
 write(a.root/'operator_transfer_gate_fresh_direct.json',raw)
 executable=a.assets/'.venv/bin/python'
 command=[str(executable),str(a.bundle/'paired_fulltrain_CPU_natural_wrapper_candidate_v1.py'),'--root',str(a.root),
          '--bundle',str(a.bundle),'--python',str(executable),'--child-arguments',str(argsfile),'--plan-sha',a.plan_sha]
 with (a.root/'operator_CPU_wrapper.stdout.log').open('x') as out,(a.root/'operator_CPU_wrapper.stderr.log').open('x') as err:
  child=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
  write(a.root/'operator_actual_CPU_wrapper_launch.json',{'actual_utc':stamp(),'wrapper_pid':child.pid,'full_argv':command,
           'source_sha256':sha(a.bundle/'paired_fulltrain_CPU_natural_wrapper_candidate_v1.py'),'fresh_direct_SHA':sha(a.root/'operator_transfer_gate_fresh_direct.json')})
  code=child.wait()
 write(a.root/'operator_CPU_wrapper_natural_exit.json',{'actual_utc':stamp(),'wrapper_pid':child.pid,'exit_code':code,'natural_wait':True,
       'original_CPU_child_exit_SHA':sha(a.root/'natural_exit.json') if (a.root/'natural_exit.json').exists() else None})
 require(code==0,'Actual original CPU audit failed; inspect preserved original failure')
 print('ACTUAL_OPERATOR_CPU_WRAPPER_NATURAL_ZERO '+json.dumps(read(a.root/'natural_exit.json')),flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();raise
