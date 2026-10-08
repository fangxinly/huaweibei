import argparse,pathlib,subprocess,sys,json,hashlib,datetime,os
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--worker',type=pathlib.Path,required=True);p.add_argument('--args',type=pathlib.Path,required=True);a=p.parse_args()
 args=json.loads(a.args.read_text());dest=pathlib.Path(args[args.index('--dest')+1]);assert not dest.exists()
 cmd=[a.python,str(a.worker),*args];child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);o,e=child.communicate()
 record={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wrapper_pid':os.getpid(),'wrapper_argv':sys.argv,'child_pid':child.pid,'child_full_argv':cmd,'worker_sha256':sha(a.worker),'args_sha256':sha(a.args),'natural_wait':True,'exit_code':child.returncode,'stdout':o,'stderr':e}
 if child.returncode==0:
  assert 'COMPLETE ' in o and (dest/'actual_receipt.json').exists();record['receipt_sha256']=sha(dest/'actual_receipt.json')
  (dest/'natural_exit.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps(record),flush=True);raise SystemExit(child.returncode)
if __name__=='__main__':main()
