"""Receive a GitHub header over encrypted stdin, upload explicit SHA files.

Token is process memory only: no file, command argument, URL, receipt or log.
Repository and upload/API hosts are fixed. This script does not read SSH secrets.
"""
import argparse,datetime,hashlib,http.client,json,os,subprocess,sys,urllib.parse,urllib.request
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def run(a):
 p=json.loads(a.plan.read_text());secret=json.loads(sys.stdin.read());headers=secret['headers']
 assert p['repository']=='fangxinly/huaweibei'
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==p['UUID']
 assert digest(__file__)==p['uploader_SHA']
 base='https://api.github.com/repos/fangxinly/huaweibei'
 def request(url,payload=None,method=None):
  assert urllib.parse.urlsplit(url).hostname in ('api.github.com','uploads.github.com')
  body=None if payload is None else json.dumps(payload).encode()
  req=urllib.request.Request(url,data=body,headers=dict(headers,**({'Content-Type':'application/json'} if body else {})),method=method)
  with urllib.request.urlopen(req,timeout=180) as r:return json.load(r)
 releases=request(base+'/releases?per_page=100')
 release=next((r for r in releases if r['tag_name']==p['tag']),None)
 if release is None:release=request(base+'/releases',dict(tag_name=p['tag'],target_commitish=p['commit'],name=p['title'],body=p['body'],draft=True))
 state=dict(status='REMOTE_RELEASE_UPLOADING',source_SHA=digest(__file__),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,release_id=release['id'],repository=p['repository'],tag=p['tag'],assets=[])
 save=lambda:a.receipt.write_text(json.dumps(state,indent=2),encoding='utf8')
 save()
 for spec in p['assets']:
  path=Path(spec['path']).resolve();assert path.is_relative_to(Path('/data/coding'))
  assert path.stat().st_size==spec['bytes'] and digest(path)==spec['sha256']
  offset=0;chunk=1_600_000_000
  with path.open('rb') as f:
   while offset<spec['bytes']:
    length=min(chunk,spec['bytes']-offset);name=spec['name'] if spec['bytes']<=chunk else spec['name']+'.part%02d'%(offset//chunk)
    existing=next((x for x in request(release['assets_url']+'?per_page=100') if x['name']==name),None)
    h=hashlib.sha256();sent=0
    if existing is None:
     u=urllib.parse.urlsplit(release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(name));assert u.hostname=='uploads.github.com'
     conn=http.client.HTTPSConnection(u.hostname,timeout=180);conn.putrequest('POST',u.path+'?'+u.query)
     for k,v in dict(headers,**{'Content-Type':'application/octet-stream','Content-Length':str(length)}).items():conn.putheader(k,v)
     conn.endheaders()
    while sent<length:
     b=f.read(min(8*1024**2,length-sent));assert b;h.update(b)
     if existing is None:conn.send(b)
     sent+=len(b)
     if sent%(128*1024**2)==0:print(json.dumps(dict(asset=name,sent=sent,total=length)),flush=True)
    if existing is None:
     response=conn.getresponse();assert response.status==201,'GitHub HTTP '+str(response.status)
     uploaded=json.load(response);conn.close()
    else:uploaded=existing
    assert uploaded['state']=='uploaded' and uploaded['size']==length and uploaded.get('digest')=='sha256:'+h.hexdigest()
    state['assets'].append(dict(name=name,bytes=length,sha256=h.hexdigest(),source_sha256=spec['sha256'],source_offset=offset,id=uploaded['id'],url=uploaded['browser_download_url']));save();offset+=length
 release=request(base+'/releases/'+str(release['id']),dict(draft=False),method='PATCH')
 fresh={x['id']:x for x in request(release['assets_url']+'?per_page=100')}
 for row in state['assets']:
  r=fresh[row['id']];assert r['size']==row['bytes'] and r.get('digest')=='sha256:'+row['sha256'];row['url']=r['browser_download_url']
 state.update(status='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),release_url=release['html_url']);save();print(json.dumps(state),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for k in ('plan','receipt'):parser.add_argument('--'+k,type=Path,required=True)
 args=parser.parse_args()
 try:run(args)
 except Exception as e:print(json.dumps({'status':'REMOTE_RELEASE_FAILED','error_type':type(e).__name__}),flush=True);raise SystemExit(1)
