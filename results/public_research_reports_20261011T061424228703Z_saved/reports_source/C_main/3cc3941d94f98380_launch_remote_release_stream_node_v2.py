"""Explicit authorized A/B stream upload, headers only in encrypted stdin."""
import argparse,json,subprocess
from pathlib import Path
from publish_release_assets_v1 import credential_headers

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--node',choices=['A','B'],required=True);p.add_argument('--remote-plan',required=True);p.add_argument('--remote-receipt',required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--publish',action='store_true');a=p.parse_args()
 if not a.publish:raise SystemExit('Explicit authorized --publish required')
 for v in (a.remote_plan,a.remote_receipt):
  if not v.startswith('/data/coding/') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-' for c in v):raise ValueError('Unsafe remote path')
 port,host={'A':('53491','root@REDACTED_SERVER_HOST.invalid'),'B':('53495','root@REDACTED_SERVER_HOST.invalid')}[a.node]
 headers=credential_headers();cmd=['ssh','-T','-p',port,'-o','PreferredAuthentications=password','-o','PubkeyAuthentication=no','-o','NumberOfPasswordPrompts=1','-o','ConnectTimeout=20',host,'/data/miniconda/envs/torch/bin/python','/data/coding/remote_release_stream_v1.py','--plan',a.remote_plan,'--receipt',a.remote_receipt]
 child=subprocess.Popen(cmd,stdin=subprocess.PIPE);child.stdin.write(json.dumps({'headers':headers}).encode());child.stdin.close();del headers
 code=child.wait();a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(dict(status='SSH_REMOTE_PUBLISHER_NATURAL_EXIT',node=a.node,natural_exit=code,fullargv=cmd)),encoding='utf8');raise SystemExit(code)
