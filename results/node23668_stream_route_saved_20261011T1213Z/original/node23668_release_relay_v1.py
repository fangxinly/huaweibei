"""Loopback-only stream relay; GitHub credentials remain on the Windows host.

Only operator-frozen manifests in this relay's registration directory can be
uploaded. Range files are never staged locally. No retries or deletion occur.
"""
import argparse,datetime as dt,hashlib,http.server,json,os,pathlib,re,sys,urllib.parse,urllib.request
P=pathlib.Path;BLOCK=1024**2;REPO='fangxinly/huaweibei';TAG='autonomous-flow-health-20261008'
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(o,indent=2)+'\n');tmp.replace(p)
def registered(root,key):
    if not re.fullmatch('[A-Za-z0-9_-]{1,80}',key):raise ValueError('Registration key invalid')
    m=json.loads((root/'registered'/f'{key}.json').read_bytes())
    if m['repository']!=REPO or m['tag']!=TAG or not m['publication_authorized']:raise PermissionError('Publication scope differs')
    if not re.fullmatch('[0-9a-f]{64}',m['whole_SHA']) or not m['parts'] or len(m['parts'])>1000:raise ValueError('Manifest invalid')
    offset=0
    for n,p in enumerate(m['parts']):
        if p['offset']!=offset or not 0<p['bytes']<=512*1024**2 or not re.fullmatch('[0-9a-f]{64}',p['SHA']):raise ValueError('Frozen range invalid')
        if not re.fullmatch('[A-Za-z0-9_.-]+',p['name']) or len(p['name'])>220:raise ValueError('Frozen asset name invalid')
        offset+=p['bytes']
    if offset!=m['bytes'] or len({p['name'] for p in m['parts']})!=len(m['parts']):raise ValueError('Frozen range coverage invalid')
    return m
def start(root):
    sys.path.insert(0,str(P(__file__).resolve().parent))
    import publish_release_assets_v1 as pub
    headers=pub.credential_headers()
    release=pub.request(f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}',headers)
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version='HTTP/1.0'
        def log_message(self,*args):pass
        def response(self,code,o):
            data=json.dumps(o).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def identity(self,verb):
            pieces=self.path.strip('/').split('/')
            if len(pieces)!=3 or pieces[0]!=verb or not pieces[2].isdigit():raise ValueError('Unknown route')
            key=pieces[1];m=registered(root,key);index=int(pieces[2]);part=m['parts'][index]
            return key,m,index,part
        def do_POST(self):
            key='unknown';index=-1
            try:
                key,m,index,part=self.identity('upload')
                if int(self.headers.get('Content-Length','0'))!=part['bytes']:raise ValueError('Upload length differs')
                receipt=root/'receipts'/f'{key}_{index:04d}.json'
                if receipt.exists():raise FileExistsError('Range already attempted; inspect original receipt before explicit next action')
                write(receipt,dict(actual_UTC=now(),status='STREAM_UPLOAD_STARTED',whole_SHA=m['whole_SHA'],part=part))
                inventory=[];page=1
                while True:
                    rows=pub.request(release['assets_url']+f'?per_page=100&page={page}',headers);inventory.extend(rows)
                    if len(rows)<100:break
                    page+=1
                if any(x['name']==part['name'] for x in inventory) or len(inventory)>=1000:raise PermissionError('Fresh asset name or Release capacity unavailable')
                h=hashlib.sha256();sent=0
                def body():
                    nonlocal sent
                    while sent<part['bytes']:
                        b=self.rfile.read(min(BLOCK,part['bytes']-sent))
                        if not b:raise EOFError('Remote original range ended early')
                        sent+=len(b);h.update(b);yield b
                url=release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(part['name'])
                req=urllib.request.Request(url,data=body(),method='POST',headers=dict(headers,**{'Content-Type':'application/octet-stream','Content-Length':str(part['bytes'])}))
                with urllib.request.urlopen(req,timeout=1800) as response:asset=json.load(response)
                if sent!=part['bytes'] or h.hexdigest()!=part['SHA'] or asset['size']!=part['bytes'] or asset['state']!='uploaded' or asset.get('digest')!='sha256:'+part['SHA']:raise ValueError('Sent bytes or GitHub digest differs')
                row=dict(actual_UTC=now(),status='REMOTE_RANGE_DIGEST_VERIFIED',whole_SHA=m['whole_SHA'],part=part,streamed_bytes=sent,streamed_SHA=h.hexdigest(),id=asset['id'],url=asset['browser_download_url'],remote_digest=asset['digest'],local_payload_disk_bytes=0,credential_remained_local=True)
                write(receipt,row);self.response(200,row)
            except Exception as exc:
                row=dict(actual_UTC=now(),status='STREAM_UPLOAD_FAILED_ORIGINAL_PRESERVED',error_type=type(exc).__name__)
                if hasattr(exc,'code'):row['HTTP_status']=exc.code
                if key!='unknown' and index>=0:write(root/'failures'/f'{key}_{index:04d}.json',row)
                self.response(500,row)
        def do_GET(self):
            if self.path=='/health':self.response(200,dict(status='LOCAL_ONLY_RELEASE_RELAY_READY',actual_UTC=now(),repository=REPO));return
            begun=False
            try:
                key,m,index,part=self.identity('download')
                row=json.loads((root/'receipts'/f'{key}_{index:04d}.json').read_bytes())
                if row['status']!='REMOTE_RANGE_DIGEST_VERIFIED' or row['part']!=part:raise ValueError('Range not published')
                if not row['url'].startswith(f'https://github.com/{REPO}/releases/download/{TAG}/'):raise PermissionError('Download scope differs')
                request=urllib.request.Request(row['url'],headers={'User-Agent':'Original-whole-restoration-via-local-relay'})
                with urllib.request.urlopen(request,timeout=1800) as source:
                    self.send_response(200);self.send_header('Content-Type','application/octet-stream');self.send_header('Content-Length',str(part['bytes']));self.end_headers();begun=True
                    total=0;h=hashlib.sha256()
                    while True:
                        b=source.read(BLOCK)
                        if not b:break
                        total+=len(b)
                        if total>part['bytes']:raise ValueError('Public range exceeds frozen size')
                        h.update(b);self.wfile.write(b)
                if total!=part['bytes'] or h.hexdigest()!=part['SHA']:raise ValueError('Public range restoration differs')
                write(root/'downloads'/f'{key}_{index:04d}.json',dict(actual_UTC=now(),status='PUBLIC_RANGE_DOWNLOAD_SHA_VERIFIED',bytes=total,SHA=h.hexdigest(),original_url=row['url'],local_payload_disk_bytes=0))
            except Exception as exc:
                write(root/'failures'/('download_'+now().replace(':','_')+'.json'),dict(actual_UTC=now(),error_type=type(exc).__name__))
                if not begun:self.response(500,dict(error_type=type(exc).__name__))
    root.mkdir(parents=True,exist_ok=True)
    server=http.server.HTTPServer(('127.0.0.1',0),Handler)
    write(root/'ready.json',dict(actual_UTC=now(),pid=os.getpid(),port=server.server_port,bind='127.0.0.1',repository=REPO,tag=TAG,source_SHA=hashlib.sha256(P(__file__).read_bytes()).hexdigest(),credentials_in_memory_only=True))
    server.serve_forever(poll_interval=1)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=P,required=True);args=a.parse_args();start(args.root)
