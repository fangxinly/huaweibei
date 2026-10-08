"""Natural-wait wrapper candidate; does not stop any child or reuse another training root."""
import argparse,datetime,hashlib,json,subprocess,sys
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--python',type=Path,required=True);p.add_argument('--child-arguments',type=Path,required=True);a=p.parse_args()
    if not str(a.root).startswith('/data/coding/paired_fulltrain_') or (a.root/'out').exists() or (a.root/'natural_exit.json').exists():
        raise PermissionError('Fresh paired root required; no duplicate launch')
    args=json.loads(a.child_arguments.read_text(encoding='utf-8'))
    if not isinstance(args,list) or any(not isinstance(x,str) for x in args):raise ValueError('Physical child argument list required')
    if args.count('--root')!=1 or args[args.index('--root')+1]!=str(a.root):raise ValueError('Child root mismatch')
    command=[str(a.python),str(a.bundle/'paired_fulltrain_runtime_candidate_v1.py'),*args]
    write(a.root/'wrapper_actual_start.json',{'actual_utc':now(),'wrapper_argv':sys.argv,'child_full_argv':command,
           'child_source_sha256':sha(a.bundle/'paired_fulltrain_runtime_candidate_v1.py'),'child_args_sha256':sha(a.child_arguments)})
    with (a.root/'child.stdout.log').open('w') as stdout,(a.root/'child.stderr.log').open('w') as stderr:
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr)
        write(a.root/'actual_child_launch.json',{'actual_utc':now(),'pid':child.pid,'full_argv':command})
        code=child.wait()
    receipt=a.root/'out/actual_stage_receipt.json'
    write(a.root/'natural_exit.json',{'actual_utc':now(),'child_pid':child.pid,'exit_code':code,'natural_exit':True,
                                    'receipt_sha256':sha(receipt) if receipt.is_file() else None,'full_argv':command})
    if code==0 and not receipt.is_file():raise RuntimeError('Natural0 without actual stage receipt')
    raise SystemExit(code)
if __name__=='__main__':main()
