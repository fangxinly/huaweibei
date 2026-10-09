import datetime, hashlib, json, os, pathlib, sys, urllib.request, urllib.parse, urllib.error
sys.path.insert(0,'C:/Users/21234/Documents/Codex/2026-10-05/ni/work')
import publish_release_assets_v1 as p
os.environ['PATH']='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'+os.pathsep+os.environ['PATH']
r=pathlib.Path(__file__).resolve().parent
receipt=r/'N2_score_Release_publication_receipt.json'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(v):receipt.write_text(json.dumps(v,indent=2)+'\n',encoding='utf8')
proof=json.loads((r/'N2_score_D_preservation_receipt.json').read_bytes()); f=pathlib.Path(proof['archive']); raw=f.read_bytes()
assert len(raw)==proof['archive_bytes'] and hashlib.sha256(raw).hexdigest()==proof['archive_SHA'] and proof['all_member_SHA_CRC_unique_passed']
state=dict(status='B_N2_AUTHOR_FIVE_ORIGINAL_RELEASE_STARTED',actual_UTC=utc(),source=str(f),source_SHA=proof['archive_SHA'],source_bytes=len(raw),pid=os.getpid(),fullargv=[sys.executable]+sys.argv)
write(state)
try:
 h=p.credential_headers(); release=p.request('https://api.github.com/repos/fangxinly/huaweibei/releases/tags/autonomous-flow-health-20261008',h)
 name='B-original-selected92-author-five-once-N2-20261009T180550Z.zip'
 assets=p.request(release['assets_url']+'?per_page=100',h); found=next((a for a in assets if a['name']==name),None)
 if found is None:
  req=urllib.request.Request(release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(name),data=raw,headers=dict(h,**{'Content-Type':'application/octet-stream','Content-Length':str(len(raw))}),method='POST')
  with urllib.request.urlopen(req,timeout=45) as response:found=json.load(response)
 assert found['state']=='uploaded' and found['size']==len(raw) and found.get('digest')=='sha256:'+proof['archive_SHA']
 state.update(status='B_N2_AUTHOR_FIVE_ORIGINAL_RELEASE_REMOTE_DIGEST_VERIFIED',actual_UTC=utc(),name=name,url=found['browser_download_url'],id=found['id'],remote_digest_verified=True)
 write(state);print(json.dumps(state))
except Exception as exc:
 state.update(status='B_N2_AUTHOR_FIVE_ORIGINAL_PUBLICATION_FAILED_PRESERVED',actual_UTC=utc(),error_type=type(exc).__name__)
 if isinstance(exc,urllib.error.HTTPError):state['HTTP_status']=exc.code
 write(state);raise RuntimeError(state['status']) from None
