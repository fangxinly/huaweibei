import argparse,datetime,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pid',type=int);a=p.parse_args();r=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(a.root),free_bytes=shutil.disk_usage(a.root).free)
for n in ('capture_receipt.json','natural_exit.json','out/progress.json','out/complete_recovery400_restore.json','out/mutable_recovery_checkpoint_ready.json'):
 f=a.root/n
 if f.exists():r[n]=json.loads(f.read_text())
for n in ('out/peer_interrupted_original.zip','complete_interrupted_original.zip'):
 f=a.root/n
 if f.exists():r[n]=f.stat().st_size
for f in (a.root/'stderr.log',Path(str(a.root)+'_launcher.err')):
 if f.exists() and f.stat().st_size:r[f.name]=f.read_text()[-2000:]
if a.pid:r['process']=subprocess.run(['ps','-p',str(a.pid),'-o','pid,ppid,etimes,time,rss,args','--width','10000'],capture_output=True,text=True).stdout
print(json.dumps(r),flush=True)
